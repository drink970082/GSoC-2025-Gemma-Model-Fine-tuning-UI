# tests/backend/manager/test_base_manager.py
from pathlib import Path

from backend.manager.base_manager import BaseManager


class _Dummy(BaseManager):
	def cleanup(self) -> None:
		pass


def test_set_work_dir_creates_directory(tmp_path: Path):
	m = _Dummy()
	target = tmp_path / "w"
	assert not target.exists()
	m.set_work_dir(str(target))
	assert m.work_dir == str(target)
	assert target.exists() and target.is_dir()


def test_set_work_dir_none_resets():
	m = _Dummy()
	m.set_work_dir(None)
	assert m.work_dir is None