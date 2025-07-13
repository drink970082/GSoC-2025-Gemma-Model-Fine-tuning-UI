import os
from typing import Optional, Any, List, Dict
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
)
import numpy as np
import json

# Constants
EVENT_FILE_PATTERN = "events.out.tfevents"
DT_FLOAT = 1
DT_INT32 = 3
DT_INT64 = 9
DT_STRING = 7


class EventFileParser:
    """Handles TensorBoard event file parsing."""

    def __init__(self, log_dir: str):
        """Initialize the event file parser."""
        self.log_dir = log_dir

    def load_event_data(
        self,
    ) -> tuple[Dict[str, Any], Dict[str, pd.DataFrame], Dict[str, Any]]:
        """Load and parse event data from TensorBoard logs."""
        data = {}
        try:
            latest_event_file = self._find_event_file()
            if not latest_event_file:
                return data
            event_acc = EventAccumulator(
                latest_event_file,
                size_guidance={"scalars": 0, "tensors": 0},
            )
            event_acc.Reload()
            tags = event_acc.Tags()
            for tag in tags.get("tensors", []):
                try:
                    events = event_acc.Tensors(tag)
                    data[tag] = self._process_tensor_events(tag, events)
                except Exception:
                    continue
        except Exception:
            return data
        metadata = {
            k: v.iloc[0]["value"] if not v.empty else None
            for k, v in data.items()
            if k in ["num_params", "parameters", "element_spec", "context_spec"]
        }
        training_metrics = {
            k: v
            for k, v in data.items()
            if k.startswith(("losses/", "perf_stats/"))
        }
        latest_training_metrics = {
            k: v.iloc[-1]["value"] if not v.empty else None
            for k, v in training_metrics.items()
        }
        return (
            self._get_parsed_metadata(metadata),
            training_metrics,
            latest_training_metrics,
        )

    def _find_event_file(self) -> Optional[str]:
        """Find the event file in the log directory."""
        for root, _, files in os.walk(self.log_dir):
            for file in files:
                if EVENT_FILE_PATTERN in file:
                    return os.path.join(root, file)
        return None

    def _get_parsed_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Get parsed metadata with all the parsed information."""
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
        parsed_data["raw"] = metadata
        return parsed_data

    def _process_tensor_events(
        self, tag: str, events: list[Any]
    ) -> pd.DataFrame:
        """Process tensor events based on tag type."""
        if tag in ["num_params", "parameters", "element_spec", "context_spec"]:
            return self._process_metadata_tensor(tag, events)
        elif tag.startswith("losses/") or tag.startswith("perf_stats/"):
            return self._process_training_metric(tag, events)
        else:
            return self._process_other_tensor(tag, events)

    def _process_metadata_tensor(
        self, tag: str, events: list[Any]
    ) -> pd.DataFrame:
        """Process metadata tensors (single values)."""
        if events:
            event = events[0]
            value = self._parse_tensor_value(event.tensor_proto)
            return pd.DataFrame(
                [
                    {
                        "wall_time": event.wall_time,
                        "step": event.step,
                        "value": value,
                        "description": f"Model metadata: {tag}",
                    }
                ],
                columns=["wall_time", "step", "value", "description"],
            )
        return pd.DataFrame()

    def _process_training_metric(self, events: List[Any]) -> pd.DataFrame:
        """Process training metrics (time series)."""
        parsed_events = []
        for e in events:
            val = self._parse_tensor_value(e.tensor_proto)
            parsed_events.append((e.wall_time, e.step, val))

        return pd.DataFrame(
            parsed_events,
            columns=["wall_time", "step", "value"],
        )

    def _process_other_tensor(self, events: List[Any]) -> pd.DataFrame:
        """Process other tensor types."""
        return pd.DataFrame(
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

    def _parse_tensor_value(self, tensor_proto: Any) -> Any:

        default_value = f"Shape: {list(tensor_proto.tensor_shape.dim)}, Dtype: {tensor_proto.dtype}"
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
            if tensor_proto.dtype == DT_FLOAT:
                tensor_data = np.frombuffer(
                    tensor_proto.tensor_content, dtype=np.float32
                )
                return float(tensor_data[0]) if len(tensor_data) > 0 else 0.0
            elif tensor_proto.dtype == DT_INT32:
                tensor_data = np.frombuffer(
                    tensor_proto.tensor_content, dtype=np.int32
                )
                return int(tensor_data[0]) if len(tensor_data) > 0 else 0
            elif tensor_proto.dtype == DT_INT64:
                tensor_data = np.frombuffer(
                    tensor_proto.tensor_content, dtype=np.int64
                )
                return int(tensor_data[0]) if len(tensor_data) > 0 else 0
            elif tensor_proto.dtype == DT_STRING:
                try:
                    return tensor_proto.tensor_content.decode("utf-8")
                except:
                    return f"String tensor: {len(tensor_proto.tensor_content)} bytes"
            else:
                return default_value
        else:
            return default_value

    def _parse_parameter_summary(self, param_text: str) -> Dict[str, Any]:
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

    def _parse_element_spec(self, spec_text: str) -> Dict[str, Any]:
        """Parse element spec JSON format."""
        if not spec_text or not isinstance(spec_text, str):
            return {}

        try:
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

    def _parse_context_spec(self, spec_text: str) -> Dict[str, Any]:
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

    def set_work_dir(self, work_dir: str) -> None:
        """Set the work directory for the event file parser."""
        self.log_dir = work_dir