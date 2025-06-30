import json
import time

import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
)

from backend.manager.base_manager import BaseManager
from backend.manager.file_manager import FileManager


class TensorBoardManager(BaseManager):
    """Manages TensorBoard event data loading and caching."""

    def __init__(self):
        super().__init__()
        self._manager_tracking_time = 0
        self._event_data = {}

    def initialize(self, file_manager: FileManager):
        self._initialized = True
        self.file_manager = file_manager
        self._manager_tracking_time = time.time()

    def cleanup(self):
        """Cleanup method called by atexit."""
        self._manager_tracking_time = 0
        self._event_data = {}

    def reset_training_time(self):
        """Reset the training start time to current time."""
        self._manager_tracking_time = time.time()
        self._event_data = {}

    def examine_event_file(self, file_path: str) -> dict:
        """Examine a specific event file and return its contents."""
        try:
            event_acc = EventAccumulator(
                file_path,
                size_guidance={"scalars": 0, "tensors": 0},
            )
            event_acc.Reload()

            tags = event_acc.Tags()
            result = {"file_path": file_path, "tags": tags, "tensor_data": {}}

            # Process tensor metrics
            for tag in tags.get("tensors", []):
                try:
                    events = event_acc.Tensors(tag)
                    result["tensor_data"][tag] = {
                        "num_events": len(events),
                        "events": [],
                    }

                    for i, event in enumerate(
                        events[:5]
                    ):  # Show first 5 events
                        value = self._parse_tensor_value(event.tensor_proto)
                        result["tensor_data"][tag]["events"].append(
                            {
                                "index": i,
                                "step": event.step,
                                "wall_time": event.wall_time,
                                "value": value,
                            }
                        )

                except Exception as e:
                    result["tensor_data"][tag] = {"error": str(e)}

            return result

        except Exception as e:
            return {"file_path": file_path, "error": str(e)}

    def _find_latest_event_file(self) -> str | None:
        """Find the latest event file using the FileManager."""
        # The ignore_timing logic is now handled by passing since_time=0
        since = 0 if self.ignore_timing else self._manager_tracking_time
        return self.file_manager.tensorboard_file.find_latest_event_file(
            since_time=since
        )

    def _load_event_data(self) -> dict[str, pd.DataFrame]:
        """Load and parse event data from TensorBoard logs."""
        data = {}

        try:
            # Find latest event file
            latest_event_file = self._find_latest_event_file()
            if not latest_event_file:
                return data

            # Always create new EventAccumulator and reload to get latest data
            event_acc = EventAccumulator(
                latest_event_file,
                size_guidance={"scalars": 0, "tensors": 0},
            )
            event_acc.Reload()

            # Get available tags
            tags = event_acc.Tags()

            # Process tensor metrics
            for tag in tags.get("tensors", []):
                try:
                    events = event_acc.Tensors(tag)

                    # Handle different types of tensor data
                    if tag in [
                        "parameters",
                        "num_params",
                        "element_spec",
                        "context_spec",
                    ]:
                        # Metadata tensors (single values)
                        if events:
                            event = events[0]
                            value = self._parse_tensor_value(event.tensor_proto)

                            data[tag] = pd.DataFrame(
                                [
                                    {
                                        "wall_time": event.wall_time,
                                        "step": event.step,
                                        "value": value,
                                        "description": f"Model metadata: {tag}",
                                    }
                                ],
                                columns=[
                                    "wall_time",
                                    "step",
                                    "value",
                                    "description",
                                ],
                            )

                    elif tag.startswith("losses/") or tag.startswith(
                        "perf_stats/"
                    ):
                        # Training metrics (time series)
                        parsed_events = []
                        for e in events:
                            val = self._parse_tensor_value(e.tensor_proto)
                            parsed_events.append((e.wall_time, e.step, val))

                        data[tag] = pd.DataFrame(
                            parsed_events,
                            columns=["wall_time", "step", "value"],
                        )

                    else:
                        # Other tensor types
                        data[tag] = pd.DataFrame(
                            [
                                (
                                    e.wall_time,
                                    e.step,
                                    f"Tensor data: {e.tensor_proto.dtype}",
                                )
                                for e in events
                            ],
                            columns=["wall_time", "step", "value"],
                        )

                except Exception as e:
                    continue

        except Exception as e:
            pass

        return data

    def _parse_tensor_value(self, tensor_proto) -> any:
        """Parse tensor value based on its type."""
        if hasattr(tensor_proto, "float_val") and tensor_proto.float_val:
            return float(tensor_proto.float_val[0])
        elif hasattr(tensor_proto, "int64_val") and tensor_proto.int64_val:
            return int(tensor_proto.int64_val[0])
        elif hasattr(tensor_proto, "string_val") and tensor_proto.string_val:
            return tensor_proto.string_val[0].decode("utf-8")
        elif (
            hasattr(tensor_proto, "tensor_content")
            and tensor_proto.tensor_content
        ):
            # Handle TensorProto format
            if tensor_proto.dtype == 1:  # DT_FLOAT
                tensor_data = np.frombuffer(
                    tensor_proto.tensor_content, dtype=np.float32
                )
                return float(tensor_data[0]) if len(tensor_data) > 0 else 0.0
            elif tensor_proto.dtype == 3:  # DT_INT32
                tensor_data = np.frombuffer(
                    tensor_proto.tensor_content, dtype=np.int32
                )
                return int(tensor_data[0]) if len(tensor_data) > 0 else 0
            elif tensor_proto.dtype == 9:  # DT_INT64
                tensor_data = np.frombuffer(
                    tensor_proto.tensor_content, dtype=np.int64
                )
                return int(tensor_data[0]) if len(tensor_data) > 0 else 0
            elif tensor_proto.dtype == 7:  # DT_STRING
                try:
                    return tensor_proto.tensor_content.decode("utf-8")
                except:
                    return f"String tensor: {len(tensor_proto.tensor_content)} bytes"
            else:
                return f"Shape: {list(tensor_proto.tensor_shape.dim)}, Dtype: {tensor_proto.dtype}"
        else:
            return f"Shape: {list(tensor_proto.tensor_shape.dim)}, Dtype: {tensor_proto.dtype}"

    def _parse_parameter_summary(self, param_text: str) -> dict:
        """Parse parameter summary text and extract key information."""
        if not param_text or not isinstance(param_text, str):
            return {}

        result = {
            "total_params": None,
            "total_bytes": None,
            "layers": [],
            "parameter_count": 0,
        }

        lines = param_text.split("\n")

        for line in lines:
            line = line.strip()
            if not line or line.startswith("|--") or line.startswith("==="):
                continue

            # Extract total information
            if "Total:" in line:
                # Format: "Total: 999,885,952 -- 1,999,771,904 bytes"
                parts = line.split("Total:")[1].strip()
                if "--" in parts:
                    params_part, bytes_part = parts.split("--")
                    result["total_params"] = int(
                        params_part.strip().replace(",", "")
                    )
                    result["total_bytes"] = int(
                        bytes_part.strip()
                        .replace(",", "")
                        .replace(" bytes", "")
                    )

            # Extract layer information
            elif "|" in line and "layer_" in line:
                # Format: "| layer_9/pre_attention_norm/scale   | (1152,)           | bfloat16 | 1,152       | 7.97      | 7.44    | ()       |"
                parts = [p.strip() for p in line.split("|") if p.strip()]
                if len(parts) >= 4:
                    layer_info = {
                        "name": parts[0],
                        "shape": parts[1],
                        "dtype": parts[2],
                        "params": (
                            int(parts[3].replace(",", ""))
                            if parts[3].replace(",", "").isdigit()
                            else 0
                        ),
                    }
                    result["layers"].append(layer_info)
                    result["parameter_count"] += layer_info["params"]

        return result

    def _parse_element_spec(self, spec_text: str) -> dict:
        """Parse element spec JSON format."""
        if not spec_text or not isinstance(spec_text, str):
            return {}

        try:
            # Extract JSON from the text (remove the ```python wrapper)
            json_start = spec_text.find("{")
            json_end = spec_text.rfind("}") + 1
            if json_start != -1 and json_end != -1:
                json_str = spec_text[json_start:json_end]
                spec_data = json.loads(json_str)

                result = {
                    "input_shape": spec_data.get("input", {}).get("shape", []),
                    "input_dtype": spec_data.get("input", {}).get("dtype", ""),
                    "loss_mask_shape": spec_data.get("loss_mask", {}).get(
                        "shape", []
                    ),
                    "target_shape": spec_data.get("target", {}).get(
                        "shape", []
                    ),
                    "batch_size": (
                        spec_data.get("input", {}).get("shape", [0])[0]
                        if spec_data.get("input", {}).get("shape")
                        else 0
                    ),
                    "sequence_length": (
                        spec_data.get("input", {}).get("shape", [0, 0])[1]
                        if len(spec_data.get("input", {}).get("shape", [])) > 1
                        else 0
                    ),
                }
                return result
        except (json.JSONDecodeError, KeyError, IndexError):
            pass

        return {}

    def _parse_context_spec(self, spec_text: str) -> dict:
        """Parse context spec table format."""
        if not spec_text or not isinstance(spec_text, str):
            return {}

        result = {"batch_specs": [], "grad_specs": [], "total_specs": 0}

        lines = spec_text.split("\n")

        for line in lines:
            line = line.strip()
            if (
                not line
                or line.startswith("|--")
                or line.startswith("===")
                or "Path" in line
                or "Spec" in line
            ):
                continue

            # Format: "| `batch.input` | `i32[4 200]` |"
            if "|" in line and "`" in line:
                parts = [p.strip() for p in line.split("|") if p.strip()]
                if len(parts) >= 2:
                    path = parts[0].strip("`")
                    spec = parts[1].strip("`")

                    spec_info = {"path": path, "spec": spec}

                    if path.startswith("batch."):
                        result["batch_specs"].append(spec_info)
                    elif path.startswith("grads."):
                        result["grad_specs"].append(spec_info)

                    result["total_specs"] += 1

        return result

    def _get_data(self) -> dict[str, pd.DataFrame]:
        """Get current event data (always loads fresh data)."""
        self._event_data = self._load_event_data()
        return self._event_data

    def _get_metadata(self) -> dict[str, any]:
        """Get metadata tensors only."""
        data = self._get_data()
        return {
            k: v.iloc[0]["value"] if not v.empty else None
            for k, v in data.items()
            if k in ["num_params", "parameters", "element_spec", "context_spec"]
        }

    def _get_training_metrics(self) -> dict[str, pd.DataFrame]:
        """Get training metrics only."""
        data = self._get_data()
        return {
            k: v
            for k, v in data.items()
            if k.startswith(("losses/", "perf_stats/"))
        }

    def _get_latest_values(self) -> dict[str, any]:
        """Get latest values for all training metrics."""
        training_data = self._get_training_metrics()
        latest_values = {}

        for metric_name, metric_df in training_data.items():
            if not metric_df.empty:
                latest_values[metric_name] = metric_df.iloc[-1]["value"]

        return latest_values

    def get_parsed_metadata(self) -> dict:
        """Get parsed metadata with all the parsed information."""
        metadata = self._get_metadata()
        parsed_data = {}

        if "parameters" in metadata and metadata["parameters"]:
            parsed_data["parameters"] = self._parse_parameter_summary(
                str(metadata["parameters"])
            )

        if "element_spec" in metadata and metadata["element_spec"]:
            parsed_data["element_spec"] = self._parse_element_spec(
                str(metadata["element_spec"])
            )

        if "context_spec" in metadata and metadata["context_spec"]:
            parsed_data["context_spec"] = self._parse_context_spec(
                str(metadata["context_spec"])
            )

        # Add raw metadata
        parsed_data["raw"] = metadata

        return parsed_data

    def get_loss_metrics(self) -> dict[str, pd.DataFrame]:
        """Get loss metrics only."""
        data = self._get_data()
        return {k: v for k, v in data.items() if k.startswith("losses/")}

    def get_performance_metrics(self) -> dict[str, pd.DataFrame]:
        """Get performance metrics only."""
        data = self._get_data()
        return {k: v for k, v in data.items() if k.startswith("perf_stats/")}

    def get_current_step(self) -> int:
        """Get current training step."""
        training_data = self._get_training_metrics()
        if (
            "losses/loss" in training_data
            and not training_data["losses/loss"].empty
        ):
            return training_data["losses/loss"].iloc[-1]["step"]
        return 0

    def get_current_loss(self) -> float:
        """Get current loss value."""
        latest_values = self._get_latest_values()
        return latest_values.get("losses/loss", 0.0)

    def get_training_speed(self) -> float:
        """Get current training speed (steps/sec)."""
        latest_values = self._get_latest_values()
        return latest_values.get("perf_stats/steps_per_sec", 0.0)

    def get_training_time(self) -> float:
        """Get total training time (hours)."""
        latest_values = self._get_latest_values()
        return latest_values.get("perf_stats/total_training_time_hours", 0.0)

    def get_data_throughput(self) -> float:
        """Get data throughput (points/sec)."""
        latest_values = self._get_latest_values()
        return latest_values.get("perf_stats/data_points_per_sec_global", 0.0)

    def get_avg_step_time(self) -> float:
        """Get average step time (seconds)."""
        latest_values = self._get_latest_values()
        return latest_values.get("perf_stats/train/avg_time_sec", 0.0)

    def get_avg_eval_time(self) -> float:
        """Get average evaluation time (seconds)."""
        latest_values = self._get_latest_values()
        return latest_values.get(
            "perf_stats/evals_along_train/avg_time_sec", 0.0
        )

    def get_eta_str(self, total_steps: int) -> str:
        """Get the estimated time remaining as a formatted string."""
        training_speed = self.get_training_speed()
        latest_step = self.get_current_step()

        if training_speed > 0 and total_steps > 0:
            remaining_steps = total_steps - latest_step
            if remaining_steps > 0:
                eta_seconds = remaining_steps / training_speed
                return time.strftime("%H:%M:%S", time.gmtime(eta_seconds))

        return "N/A"
