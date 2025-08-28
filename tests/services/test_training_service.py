# tests/services/test_training_service.py
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import services.training_service as ts_mod
from services.training_service import TrainingService


@pytest.fixture(autouse=True)
def stub_streamlit(monkeypatch):
    ts_mod.st = MagicMock()
    return ts_mod.st


@pytest.fixture
def svc(monkeypatch, tmp_path):
    pm = MagicMock()
    tb = MagicMock()
    sysm = MagicMock()

    # Point CHECKPOINT_FOLDER to tmp
    ts_mod.config.CHECKPOINT_FOLDER = str(tmp_path)

    s = TrainingService(pm, tb, sysm)
    return s, pm, tb, sysm


def _tc(epochs=3, name="m"):
    mc = SimpleNamespace(epochs=epochs)
    return SimpleNamespace(model_config=mc, model_name=name)


def test_start_training_already_running_warns_and_returns_false(svc):
    s, pm, tb, sysm = svc
    s.is_training_running = MagicMock(return_value="RUNNING")

    out = s.start_training(_tc())
    assert out is False
    ts_mod.st.warning.assert_called()


def test_start_training_happy_path(svc, monkeypatch):
    s, pm, tb, sysm = svc
    s.is_training_running = MagicMock(side_effect=["IDLE", "RUNNING"])
    monkeypatch.setattr(TrainingService, "_wait_for_state_file", lambda self, timeout=10: None)

    out = s.start_training(_tc(name="modelX"))

    assert out is True
    pm.update_config.assert_called_once()
    pm.start_training.assert_called_once()
    # work_dir set and applied to managers
    assert s.work_dir is not None and "modelX-" in s.work_dir
    pm.set_work_dir.assert_called_with(s.work_dir)
    tb.set_work_dir.assert_called_with(s.work_dir)


def test_start_training_error_path_returns_false(svc):
    s, pm, tb, sysm = svc
    s.is_training_running = MagicMock(return_value="IDLE")
    pm.start_training.side_effect = RuntimeError("boom")

    out = s.start_training(_tc())
    assert out is False
    ts_mod.st.error.assert_called()


def test_stop_training_calls_pm_and_resets_workdir(svc, monkeypatch):
    s, pm, tb, sysm = svc
    spy = MagicMock()
    monkeypatch.setattr(TrainingService, "_reset_work_dir", spy)

    out = s.stop_training(mode="force")
    assert out == pm.terminate_process.return_value
    pm.terminate_process.assert_called_once_with(mode="force", delete_checkpoint=True)
    spy.assert_called_once()


def test_is_training_running_orphaned_triggers_cleanup(svc, monkeypatch):
    s, pm, tb, sysm = svc
    pm.get_status.return_value = "ORPHANED"
    monkeypatch.setattr(ts_mod.time, "sleep", lambda *_: None)

    out = s.is_training_running()
    assert out == "IDLE"
    pm.force_cleanup.assert_called_once()
    ts_mod.st.warning.assert_called()
    ts_mod.st.success.assert_called()


def test_is_training_running_finished_resets_state(svc):
    s, pm, tb, sysm = svc
    pm.get_status.return_value = "FINISHED"
    out = s.is_training_running()
    assert out == "FINISHED"
    pm.reset_state.assert_called_once_with()


def test_is_training_running_failed_resets_and_clears_checkpoint(svc):
    s, pm, tb, sysm = svc
    pm.get_status.return_value = "FAILED"
    out = s.is_training_running()
    assert out == "FAILED"
    pm.reset_state.assert_called_once_with(delete_checkpoint=True)


def test_is_training_running_passthrough(svc):
    s, pm, tb, sysm = svc
    pm.get_status.return_value = "RUNNING"
    assert s.is_training_running() == "RUNNING"


def test_getters_and_pass_throughs(svc):
    s, pm, tb, sysm = svc
    s.training_config = _tc(epochs=5)
    tb.get_kpi_data.return_value = {"ok": 1}
    tb.get_training_metrics.return_value = {"df": "x"}
    sysm.get_history_as_dataframes.return_value = {"cpu": "df"}
    sysm.has_gpu.return_value = True
    pm.read_stdout_log.return_value = "out"
    pm.read_stderr_log.return_value = "err"

    assert s.get_training_config() is s.training_config
    assert s.get_kpi_data() == {"ok": 1}
    tb.get_kpi_data.assert_called_once_with(5)
    assert s.get_training_metrics() == {"df": "x"}
    assert s.get_system_usage_history() == {"cpu": "df"}
    assert s.has_gpu() is True
    assert s.get_log_contents() == ("out", "err")


def test_set_and_reset_work_dir(svc, tmp_path):
    s, pm, tb, sysm = svc
    ts_mod.config.CHECKPOINT_FOLDER = str(tmp_path)

    s._set_work_dir("myModel")
    assert s.work_dir is not None
    assert s.work_dir.startswith(str(tmp_path))
    pm.set_work_dir.assert_called_with(s.work_dir)
    tb.set_work_dir.assert_called_with(s.work_dir)

    s._reset_work_dir()
    assert s.work_dir is None
    pm.set_work_dir.assert_called_with(None)
    tb.set_work_dir.assert_called_with(None)