from collections import deque

import pandas as pd
import psutil
import pynvml

from backend.manager.base_manager import BaseManager


class SystemManager(BaseManager):
    """Manages polling and history of system resource usage."""

    def __init__(self, history_length: int = 100):
        super().__init__()
        self._history_length = history_length
        self.history = {
            "gpu_util": deque(maxlen=self._history_length),
            "gpu_mem": deque(maxlen=self._history_length),
            "gpu_temp": deque(maxlen=self._history_length),
            "cpu_util": deque(maxlen=self._history_length),
        }
        try:
            pynvml.nvmlInit()
            self._is_nvml_initialized = True
        except pynvml.NVMLError:
            self._is_nvml_initialized = False
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
            self.history["gpu_util"].append(0)
            self.history["gpu_mem"].append(0)
            self.history["gpu_temp"].append(0)
            return

        try:
            device_count = pynvml.nvmlDeviceGetCount()
            gpu_util, gpu_mem, gpu_temp = [], [], []
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
            self.history["gpu_mem"].append(
                total_used_mem / (1024**3)
            )  # Total memory in GB
            self.history["gpu_temp"].append(
                sum(gpu_temp) / len(gpu_temp) if gpu_temp else 0
            )

        except pynvml.NVMLError:
            self.history["gpu_util"].append(0)
            self.history["gpu_mem"].append(0)
            self.history["gpu_temp"].append(0)

    def get_history_as_dataframes(self) -> dict:
        """Returns the history as a dictionary of Pandas DataFrames for charting."""
        return {
            "CPU Utilization (%)": pd.DataFrame(self.history["cpu_util"]),
            "GPU Utilization (%)": pd.DataFrame(self.history["gpu_util"]),
            "GPU Memory (GB)": pd.DataFrame(self.history["gpu_mem"]),
            "GPU Temperature (°C)": pd.DataFrame(self.history["gpu_temp"]),
        }

    def has_gpu(self) -> bool:
        """Returns True if NVML was initialized successfully."""
        return self._is_nvml_initialized
