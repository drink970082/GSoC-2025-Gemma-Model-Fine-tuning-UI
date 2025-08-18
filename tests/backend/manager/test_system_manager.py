# tests/backend/manager/test_system_manager.py
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

import backend.manager.system_manager as sm_mod
from backend.manager.system_manager import SystemManager


@pytest.fixture
def psutil_stub(monkeypatch):
	ps = SimpleNamespace(
		cpu_percent=lambda interval=None: 42.0,
	)
	monkeypatch.setattr(sm_mod, "psutil", ps)
	return ps


def _make_nvml_ok(device_specs):
	# device_specs: list of dicts with keys gpu, used_bytes, temp
	class Util:
		def __init__(self, gpu):
			self.gpu = gpu

	class Mem:
		def __init__(self, used):
			self.used = used

	class NVML:
		class NVMLError(Exception):
			pass

		NVML_TEMPERATURE_GPU = 0

		def __init__(self):
			self.shutdown_called = 0

		def nvmlInit(self):
			return None

		def nvmlShutdown(self):
			self.shutdown_called += 1

		def nvmlDeviceGetCount(self):
			return len(device_specs)

		def nvmlDeviceGetHandleByIndex(self, i):
			return i

		def nvmlDeviceGetUtilizationRates(self, handle):
			return Util(device_specs[handle]["gpu"])

		def nvmlDeviceGetMemoryInfo(self, handle):
			return Mem(device_specs[handle]["used_bytes"])

		def nvmlDeviceGetTemperature(self, handle, kind):
			return device_specs[handle]["temp"]

	return NVML()


def _make_nvml_missing():
	class NVML:
		class NVMLError(Exception):
			pass

		def nvmlInit(self):
			raise NVML.NVMLError("missing")

		def nvmlShutdown(self):
			pass
	return NVML()


def test_init_without_nvml_sets_flag_false(psutil_stub, monkeypatch):
	monkeypatch.setattr(sm_mod, "pynvml", _make_nvml_missing())
	sm = SystemManager()
	assert sm.has_gpu() is False


def test_cleanup_calls_nvml_shutdown_when_initialized(psutil_stub, monkeypatch):
	nvml = _make_nvml_ok([{"gpu": 10, "used_bytes": 1_000_000_000, "temp": 55}])
	monkeypatch.setattr(sm_mod, "pynvml", nvml)
	sm = SystemManager()
	assert sm.has_gpu() is True
	sm.cleanup()
	assert nvml.shutdown_called == 1


def test_get_history_as_dataframes_cpu_only(psutil_stub, monkeypatch):
	monkeypatch.setattr(sm_mod, "pynvml", _make_nvml_missing())
	sm = SystemManager()
	df_map = sm.get_history_as_dataframes()
	assert set(df_map.keys()) == {
		"CPU Utilization (%)",
		"GPU Utilization (%)",
		"GPU Memory (GB)",
		"GPU Temperature (°C)",
	}
	assert isinstance(df_map["CPU Utilization (%)"], pd.DataFrame)
	assert float(df_map["CPU Utilization (%)"].iloc[0, 0]) == 42.0
	assert float(df_map["GPU Utilization (%)"].iloc[0, 0]) == 0.0
	assert float(df_map["GPU Memory (GB)"].iloc[0, 0]) == 0.0
	assert float(df_map["GPU Temperature (°C)"].iloc[0, 0]) == 0.0


def test_poll_gpu_usage_aggregates_devices(psutil_stub, monkeypatch):
	nvml = _make_nvml_ok(
		[
			{"gpu": 20, "used_bytes": 2_000_000_000, "temp": 50},
			{"gpu": 40, "used_bytes": 3_000_000_000, "temp": 60},
		]
	)
	monkeypatch.setattr(sm_mod, "pynvml", nvml)
	sm = SystemManager()
	sm._poll_system_usage()
	assert sm.history["cpu_util"][-1] == 42.0
	assert sm.history["gpu_util"][-1] == 30
	gb = sm.history["gpu_mem"][-1]
	assert 4.5 <= gb <= 5.0
	assert sm.history["gpu_temp"][-1] == 55


def test_poll_gpu_usage_handles_nvml_errors(psutil_stub, monkeypatch):
	class BadNVML:
		class NVMLError(Exception):
			pass
		NVML_TEMPERATURE_GPU = 0
		def nvmlInit(self): return None
		def nvmlDeviceGetCount(self): raise BadNVML.NVMLError("boom")
		def nvmlShutdown(self): pass

	monkeypatch.setattr(sm_mod, "pynvml", BadNVML())
	sm = SystemManager()
	sm._poll_system_usage()
	assert sm.history["gpu_util"][-1] == 0
	assert sm.history["gpu_mem"][-1] == 0
	assert sm.history["gpu_temp"][-1] == 0