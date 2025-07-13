from collections import deque
import pandas as pd
import psutil
import pynvml
from typing import Dict, Any, Deque

from backend.manager.base_manager import BaseManager

DEFAULT_HISTORY_LENGTH = 100


class SystemManager(BaseManager):
    """Manages polling and history of system resource usage."""

    def __init__(self) -> None:
        """Initialize the SystemManager."""
        super().__init__()
        self.history: Dict[str, Deque[float]] = {
            "gpu_util": deque(maxlen=DEFAULT_HISTORY_LENGTH),
            "gpu_mem": deque(maxlen=DEFAULT_HISTORY_LENGTH),
            "gpu_temp": deque(maxlen=DEFAULT_HISTORY_LENGTH),
            "cpu_util": deque(maxlen=DEFAULT_HISTORY_LENGTH),
        }
        try:
            pynvml.nvmlInit()
            self._is_nvml_initialized: bool = True
        except pynvml.NVMLError:
            self._is_nvml_initialized: bool = False
            print("NVML library not found. GPU monitoring will be disabled.")

    def cleanup(self) -> None:
        """Shutdown the NVML library."""
        if self._is_nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except pynvml.NVMLError:
                pass

    def poll_system_usage(self) -> None:
        """Polls CPU and GPU usage and updates the history deques."""
        self.history["cpu_util"].append(psutil.cpu_percent())
        if not self._is_nvml_initialized:
            self._append_default_gpu_values()
            return
        try:
            self._poll_gpu_usage()
        except pynvml.NVMLError:
            self._append_default_gpu_values()

    def get_history_as_dataframes(self) -> Dict[str, pd.DataFrame]:
        """Returns the history as a dictionary of Pandas DataFrames for charting."""
        return {
            "CPU Utilization (%)": pd.DataFrame(list(self.history["cpu_util"])),
            "GPU Utilization (%)": pd.DataFrame(list(self.history["gpu_util"])),
            "GPU Memory (GB)": pd.DataFrame(list(self.history["gpu_mem"])),
            "GPU Temperature (°C)": pd.DataFrame(
                list(self.history["gpu_temp"])
            ),
        }

    def has_gpu(self) -> bool:
        """Returns True if NVML was initialized successfully."""
        return self._is_nvml_initialized

    def _append_default_gpu_values(self) -> None:
        """Append default values when GPU monitoring is unavailable."""
        self.history["gpu_util"].append(0)
        self.history["gpu_mem"].append(0)
        self.history["gpu_temp"].append(0)

    def _poll_gpu_usage(self) -> None:
        """Poll GPU usage from all available devices."""
        device_count = pynvml.nvmlDeviceGetCount()
        gpu_util, gpu_temp = [], []
        total_used_mem = 0

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_util.append(util_info.gpu)
            total_used_mem += mem_info.used
            gpu_temp.append(
                pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
            )
        self.history["gpu_util"].append(
            sum(gpu_util) / len(gpu_util) if gpu_util else 0
        )
        self.history["gpu_mem"].append(total_used_mem / (1024**3))
        self.history["gpu_temp"].append(
            sum(gpu_temp) / len(gpu_temp) if gpu_temp else 0
        )
