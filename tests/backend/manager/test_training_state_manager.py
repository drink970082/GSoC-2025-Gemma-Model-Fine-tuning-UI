import json
import os
from pathlib import Path

import pytest

from backend.manager.training_state_manager import TrainingStateManager
from config.app_config import TrainingStatus


def test_get_state_returns_idle_when_missing(tmp_path: Path):
    tsm = TrainingStateManager()
    tsm.state_file = str(tmp_path / "training_state.json")
    assert tsm.get_state() == {"status": TrainingStatus.IDLE.value}


def test_get_state_handles_corrupt_json(tmp_path: Path):
    state_path = tmp_path / "training_state.json"
    state_path.write_text("{not: json", encoding="utf-8")
    tsm = TrainingStateManager()
    tsm.state_file = str(state_path)
    assert tsm.get_state() == {"status": TrainingStatus.IDLE.value}


def test_mark_running_sets_values(tmp_path: Path):
    tsm = TrainingStateManager()
    tsm.state_file = str(tmp_path / "training_state.json")
    tsm.mark_running(pid=1234, start_time="2024-01-01T00:00:00")

    state = tsm.get_state()
    assert state["status"] == TrainingStatus.RUNNING.value
    assert state["pid"] == 1234
    assert state["start_time"] == "2024-01-01T00:00:00"
    assert state["end_time"] is None
    assert state["error"] is None


def test_mark_finished_preserves_pid_and_start_time(tmp_path: Path):
    tsm = TrainingStateManager()
    tsm.state_file = str(tmp_path / "training_state.json")

    # Seed with running state
    tsm.mark_running(pid=42, start_time="2024-01-01T00:00:00")
    tsm.mark_finished(end_time="2024-01-01T01:00:00")

    state = tsm.get_state()
    assert state["status"] == TrainingStatus.FINISHED.value
    assert state["pid"] == 42
    assert state["start_time"] == "2024-01-01T00:00:00"
    assert state["end_time"] == "2024-01-01T01:00:00"
    assert state["error"] is None


def test_mark_failed_sets_error_and_preserves_context(tmp_path: Path):
    tsm = TrainingStateManager()
    tsm.state_file = str(tmp_path / "training_state.json")

    tsm.mark_running(pid=9001, start_time="2024-02-01T12:00:00")
    tsm.mark_failed(error="boom", end_time="2024-02-01T12:01:00")

    state = tsm.get_state()
    assert state["status"] == TrainingStatus.FAILED.value
    assert state["pid"] == 9001
    assert state["start_time"] == "2024-02-01T12:00:00"
    assert state["end_time"] == "2024-02-01T12:01:00"
    assert state["error"] == "boom"


def test_mark_idle_resets_to_idle(tmp_path: Path):
    tsm = TrainingStateManager()
    tsm.state_file = str(tmp_path / "training_state.json")

    tsm.mark_running(pid=1, start_time="t0")
    tsm.mark_idle()

    state = tsm.get_state()
    assert state["status"] == TrainingStatus.IDLE.value
    # Other fields present but None
    assert state["pid"] is None
    assert state["error"] is None
    assert state["start_time"] is None
    assert state["end_time"] is None


def test_mark_orphaned_sets_error_and_preserves_context(tmp_path: Path):
    tsm = TrainingStateManager()
    tsm.state_file = str(tmp_path / "training_state.json")

    tsm.mark_running(pid=77, start_time="t0")
    tsm.mark_orphaned(error="Process not found")

    state = tsm.get_state()
    assert state["status"] == TrainingStatus.ORPHANED.value
    assert state["pid"] == 77
    assert state["start_time"] == "t0"
    assert state["error"] == "Process not found"


def test_cleanup_removes_state_file(tmp_path: Path):
    state_path = tmp_path / "training_state.json"
    state_path.write_text(
        json.dumps({"status": TrainingStatus.IDLE.value}), encoding="utf-8"
    )
    assert state_path.exists()

    tsm = TrainingStateManager()
    tsm.state_file = str(state_path)
    tsm.cleanup()

    assert not os.path.exists(tsm.state_file)


def test_atomic_write_overwrites(tmp_path: Path):
    # Ensure atomic write replaces previous content fully
    tsm = TrainingStateManager()
    tsm.state_file = str(tmp_path / "training_state.json")

    tsm.mark_running(pid=1, start_time="t0")
    first = tsm.get_state()
    assert first["status"] == TrainingStatus.RUNNING.value

    tsm.mark_idle()
    second = tsm.get_state()
    assert second["status"] == TrainingStatus.IDLE.value
