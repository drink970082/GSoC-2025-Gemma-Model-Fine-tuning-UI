# tests/services/test_di_container.py
import atexit
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import services.di_container as di_mod
from services.di_container import DIContainer


@pytest.fixture(autouse=True)
def stub_atexit(monkeypatch):
    calls = {"args": []}
    monkeypatch.setattr(atexit, "register", lambda fn: calls["args"].append(fn))
    return calls


def test_get_registers_services_once_and_returns(monkeypatch, stub_atexit):
    # Stubs for managers
    PM = MagicMock(name="ProcessManager")
    TSM = MagicMock(name="TrainingStateManager")
    TBM = MagicMock(name="TensorBoardManager")
    SYSM = MagicMock(name="SystemManager")
    monkeypatch.setattr(di_mod, "ProcessManager", PM)
    monkeypatch.setattr(di_mod, "TrainingStateManager", TSM)
    monkeypatch.setattr(di_mod, "TensorBoardManager", TBM)
    monkeypatch.setattr(di_mod, "SystemManager", SYSM)

    # Stub TrainingService where it's imported from
    class _TS:
        def __init__(
            self, process_manager, tensorboard_manager, system_manager
        ):
            self.process_manager = process_manager
            self.tensorboard_manager = tensorboard_manager
            self.system_manager = system_manager

    import services.training_service as ts_mod

    monkeypatch.setattr(ts_mod, "TrainingService", _TS)

    c = DIContainer()

    # First get triggers setup
    ts = c.get("training_service")
    assert isinstance(ts, _TS)

    # Managers constructed and registered (check identity against mock return_values)
    pm_inst = c.get("process_manager")
    tsm_inst = c.get("training_state_manager")
    tbm_inst = c.get("tensorboard_manager")
    sysm_inst = c.get("system_manager")

    assert pm_inst is PM.return_value
    assert tsm_inst is TSM.return_value
    assert tbm_inst is TBM.return_value
    assert sysm_inst is SYSM.return_value

    PM.assert_called_once_with(tsm_inst)
    TSM.assert_called_once()
    TBM.assert_called_once()
    SYSM.assert_called_once()

    # atexit registered exactly once
    assert len(stub_atexit["args"]) == 1

    # TrainingService wired up with same instances
    assert ts.process_manager is pm_inst
    assert ts.tensorboard_manager is tbm_inst
    assert ts.system_manager is sysm_inst


def test_get_unknown_raises_keyerror():
    c = DIContainer()
    # Simulate setup already done without services
    c._setup_done = True
    c._services = {}
    with pytest.raises(KeyError):
        c.get("nope")


def test_cleanup_all_calls_cleanup_and_handles_exceptions(capfd):
    c = DIContainer()
    ok = MagicMock()
    err = MagicMock()
    err.cleanup.side_effect = RuntimeError("boom")
    c._services = {"ok": ok, "err": err}
    c._setup_done = True

    c._cleanup_all()

    ok.cleanup.assert_called_once()
    err.cleanup.assert_called_once()
    # Prints an error but does not raise
    out = capfd.readouterr()
    assert "Error during cleanup" in out.out
