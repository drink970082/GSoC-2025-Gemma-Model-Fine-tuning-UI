# tests/backend/manager/test_process_manager.py
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from backend.manager.process_manager import ProcessManager
from backend.manager.training_state_manager import TrainingStateManager
from backend.trainer_main import parse_config
from config.app_config import TrainingStatus


def create_test_config(method="Standard", parameters=None):
    return {
        "model_name": "test_model",
        "model_config": {
            "model_variant": "gemma-2b",
            "epochs": 1,
            "learning_rate": 0.001,
            "method": method,
            **({"parameters": parameters} if parameters else {}),
        },
        "data_config": {
            "source": "json",
            "dataset_name": "test_dataset.jsonl",
            "split": "train",
            "shuffle": True,
            "batch_size": 2,
            "seq2seq_in_prompt": "prompt",
            "seq2seq_in_response": "response",
            "seq2seq_max_length": 128,
            "seq2seq_truncate": True,
            "config": None,
        },
    }


@pytest.fixture
def pm_setup(monkeypatch, tmp_path: Path):
    # Stub streamlit
    import backend.manager.process_manager as pm_mod

    pm_mod.st = MagicMock()

    # Stub config values used by ProcessManager
    pm_mod.config.TRAINER_MAIN_PATH = "/tmp/fake_trainer_main.py"
    pm_mod.config.TRAINER_STDOUT_LOG = "stdout.log"
    pm_mod.config.TRAINER_STDERR_LOG = "stderr.log"

    # Speed up sleep
    monkeypatch.setattr(pm_mod.time, "sleep", lambda *_args, **_kwargs: None)

    # Fake TrainingStateManager
    tsm = MagicMock(spec=TrainingStateManager)
    # Default state: IDLE
    tsm.get_state.return_value = {"status": TrainingStatus.IDLE.value}

    pm = ProcessManager(tsm)
    pm.set_work_dir(str(tmp_path / "work"))
    return pm, tsm, pm_mod


def _make_training_config():
    return parse_config(create_test_config())


def test_set_work_dir_sets_log_paths(pm_setup, tmp_path: Path):
    pm, _tsm, pm_mod = pm_setup
    assert pm.work_dir is not None
    assert pm.stdout_log_path == os.path.join(
        pm.work_dir, pm_mod.config.TRAINER_STDOUT_LOG
    )
    assert pm.stderr_log_path == os.path.join(
        pm.work_dir, pm_mod.config.TRAINER_STDERR_LOG
    )
    assert os.path.isdir(pm.work_dir)


def test_start_training_happy_path(pm_setup, monkeypatch):
    pm, tsm, pm_mod = pm_setup
    pm.update_config(_make_training_config())

    class FakeProc:
        def __init__(self):
            self.pid = 1234
            self.returncode = None

        def poll(self):
            return None

    popen_mock = MagicMock(return_value=FakeProc())
    monkeypatch.setattr(pm_mod.subprocess, "Popen", popen_mock)

    pm.start_training()

    # Subprocess started with expected argv (basic shape)
    assert popen_mock.called
    args, kwargs = popen_mock.call_args
    cmd = args[0]
    assert cmd[0] == "python"
    assert cmd[1] == pm_mod.config.TRAINER_MAIN_PATH
    # JSON arg is parseable and includes model_config
    assert cmd[2] == "--config"
    json.loads(cmd[3])
    assert cmd[4] == "--work_dir"
    assert cmd[5] == pm.work_dir

    # State marked running
    tsm.mark_running.assert_called_once()
    # model_config.json written
    model_cfg_path = os.path.join(pm.work_dir, "model_config.json")
    assert os.path.exists(model_cfg_path)


def test_start_training_when_already_running(pm_setup, monkeypatch):
    pm, tsm, pm_mod = pm_setup
    pm.update_config(_make_training_config())

    # Simulate already running
    tsm.get_state.return_value = {"status": TrainingStatus.RUNNING.value}

    # Ensure we can assert not-called safely
    monkeypatch.setattr(pm_mod.subprocess, "Popen", MagicMock())

    pm.start_training()

    # No subprocess spawn
    assert not pm_mod.subprocess.Popen.called
    # Warned via streamlit
    pm_mod.st.warning.assert_called_once()


def test_start_training_without_work_dir_raises(monkeypatch, tmp_path: Path):
    # Fresh instance with no work_dir set
    tsm = MagicMock(spec=TrainingStateManager)
    tsm.get_state.return_value = {"status": TrainingStatus.IDLE.value}
    pm = ProcessManager(tsm)
    pm.set_work_dir(None)

    with pytest.raises(ValueError, match="Work directory is not set"):
        pm.start_training()


def test_start_training_missing_config_raises(pm_setup):
    pm, _tsm, _pm_mod = pm_setup
    pm.training_config = None

    with pytest.raises(
        ValueError, match="Data or model configuration is not set"
    ):
        pm.start_training()


def test_start_training_subprocess_exits_early_marks_failed(
    pm_setup, monkeypatch
):
    pm, tsm, pm_mod = pm_setup
    pm.update_config(_make_training_config())

    class FakeProc:
        def __init__(self):
            self.pid = 11
            self.returncode = 0

        def poll(self):
            return self.returncode

    monkeypatch.setattr(
        pm_mod.subprocess, "Popen", MagicMock(return_value=FakeProc())
    )

    # Make stderr empty so it falls back to generic error
    pm.start_training()

    # Failed path taken
    assert pm.work_dir is None  # reset + remove work dir
    assert tsm.mark_failed.called
    pm_mod.st.error.assert_called()


def test_terminate_process_graceful_success(
    pm_setup, monkeypatch, tmp_path: Path
):
    pm, tsm, pm_mod = pm_setup

    class FakeProc:
        def __init__(self):
            self.pid = 999
            self.returncode = None
            self._killed = False
            self._sent_signal = None
            self._wait_calls = 0

        def poll(self):
            return None

        def send_signal(self, sig):
            self._sent_signal = sig

        def wait(self, timeout=None):
            self._wait_calls += 1
            self.returncode = 0
            return 0

        def kill(self):
            self._killed = True

    pm.training_process = FakeProc()
    pm._log_stdout_handle = MagicMock(closed=False)
    pm._log_stderr_handle = MagicMock(closed=False)

    prev_dir = pm.work_dir
    ok = pm.terminate_process(mode="graceful", delete_checkpoint=False)

    assert ok is True
    assert pm.training_process is None
    # reset_state clears work_dir, but dir remains on disk when delete_checkpoint=False
    assert pm.work_dir is None
    assert os.path.isdir(prev_dir)
    assert tsm.mark_idle.called
    assert pm._log_stdout_handle is None
    assert pm._log_stderr_handle is None


def test_terminate_process_graceful_timeout_then_force_success(
    pm_setup, monkeypatch, tmp_path: Path
):
    pm, tsm, pm_mod = pm_setup

    class FakeProc:
        def __init__(self):
            self.pid = 1000
            self.returncode = None
            self.killed = False

        def poll(self):
            return None

        def send_signal(self, sig):
            pass

        def wait(self, timeout=None):
            raise pm_mod.subprocess.TimeoutExpired(cmd="x", timeout=timeout)

        def kill(self):
            self.killed = True

        def wait_force(self, timeout=None):
            self.returncode = -9
            return -9

    fp = FakeProc()

    # First wait raises, then we monkeypatch wait to force path success
    def wait_side_effect(timeout=None):
        # first call from graceful raises; subsequent force call uses wait_force
        pm.training_process.wait = fp.wait_force
        raise pm_mod.subprocess.TimeoutExpired(cmd="x", timeout=timeout)

    fp.wait = wait_side_effect
    pm.training_process = fp

    ok = pm.terminate_process(mode="graceful", delete_checkpoint=True)

    assert ok is True
    assert fp.killed is True
    # delete_checkpoint=True removes work_dir
    assert pm.work_dir is None
    assert tsm.mark_idle.called


def test_terminate_process_force_failure_marks_orphaned(pm_setup, monkeypatch):
    pm, tsm, pm_mod = pm_setup

    class FakeProc:
        def __init__(self):
            self.pid = 1001
            self.returncode = None

        def poll(self):
            return None

        def kill(self):
            pass

        def wait(self, timeout=None):
            raise pm_mod.subprocess.TimeoutExpired(cmd="x", timeout=timeout)

    pm.training_process = FakeProc()
    ok = pm.terminate_process(mode="force", delete_checkpoint=False)

    assert ok is False
    tsm.mark_orphaned.assert_called_once()
    # Should not mark idle on failure
    assert not tsm.mark_idle.called


def test_force_cleanup_variants(pm_setup, monkeypatch):
    pm, tsm, pm_mod = pm_setup

    called = {"run": 0}

    def run_ok(args, check=False):
        called["run"] += 1
        return MagicMock()

    # OK path
    monkeypatch.setattr(pm_mod.subprocess, "run", run_ok)
    pm.reset_state = MagicMock()
    assert pm.force_cleanup() is True
    tsm.mark_idle.assert_called()
    pm.reset_state.assert_called()

    # FileNotFoundError path
    def run_missing(args, check=False):
        raise FileNotFoundError()

    monkeypatch.setattr(pm_mod.subprocess, "run", run_missing)
    pm.reset_state = MagicMock()
    tsm.mark_idle.reset_mock()
    assert pm.force_cleanup() is True
    tsm.mark_idle.assert_called()
    pm.reset_state.assert_called()

    # Generic Exception path
    def run_err(args, check=False):
        raise RuntimeError("boom")

    monkeypatch.setattr(pm_mod.subprocess, "run", run_err)
    tsm.mark_idle.reset_mock()
    pm.reset_state = MagicMock()
    assert pm.force_cleanup() is False
    tsm.mark_idle.assert_not_called()
    pm.reset_state.assert_not_called()


def test_get_status_handles_dead_process_success(pm_setup, tmp_path: Path):
    pm, tsm, pm_mod = pm_setup

    class DeadProcOK:
        def __init__(self):
            self.pid = 2
            self.returncode = 0

        def poll(self):
            return self.returncode

    # RUNNING but process finished with code 0
    tsm.get_state.return_value = {"status": TrainingStatus.RUNNING.value}
    pm.training_process = DeadProcOK()

    status = pm.get_status()
    assert status == TrainingStatus.FINISHED.value
    tsm.mark_finished.assert_called_once()


def test_get_status_handles_dead_process_failure_with_stderr(
    pm_setup, tmp_path: Path
):
    pm, tsm, pm_mod = pm_setup

    class DeadProcFail:
        def __init__(self):
            self.pid = 3
            self.returncode = 1

        def poll(self):
            return self.returncode

    err_path = tmp_path / "stderr.log"
    err_path.write_text("boom", encoding="utf-8")
    pm.stderr_log_path = str(err_path)

    tsm.get_state.return_value = {"status": TrainingStatus.RUNNING.value}
    pm.training_process = DeadProcFail()

    status = pm.get_status()
    assert status == TrainingStatus.FAILED.value
    tsm.mark_failed.assert_called_once()
    # ensure our error content was used
    called_args = tsm.mark_failed.call_args[0]
    assert "boom" in called_args[0]


def test_get_status_running_without_process(pm_setup):
    pm, tsm, pm_mod = pm_setup
    tsm.get_state.return_value = {"status": TrainingStatus.RUNNING.value}
    pm.training_process = None

    status = pm.get_status()
    assert status == TrainingStatus.RUNNING.value
    tsm.mark_finished.assert_not_called()
    tsm.mark_failed.assert_not_called()
