"""Per-user, same-machine lock paths for experiment resource files."""

import getpass
import os
import stat
import tempfile
from pathlib import Path

import pytest
from zcu_tools.resources.file_lock import lock_path_for


def test_runtime_lock_identity_and_private_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
    data = tmp_path / "data"
    data.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(data, target_is_directory=True)
    monkeypatch.chdir(tmp_path)
    root = runtime / "qubit-measure" / "locks"
    assert not root.exists()

    lock = lock_path_for(Path("data/setup.yaml"))

    assert lock.parent == root
    assert root.is_dir()
    if os.name == "posix":
        assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert lock.suffix == ".lock"
    assert len(lock.stem) == 64
    assert all(character in "0123456789abcdef" for character in lock.stem)
    assert not lock.exists(), "choosing a path must not acquire or create a lock file"
    assert lock_path_for(data / "setup.yaml") == lock
    assert lock_path_for(alias / "setup.yaml") == lock
    assert lock_path_for(data / "../data/setup.yaml") == lock
    assert lock_path_for(data / "point.yaml") != lock
    assert lock_path_for(tmp_path / "setup.yaml") != lock


@pytest.mark.parametrize("runtime_state", ["unset", "missing", "file", "empty"])
def test_unavailable_runtime_uses_per_user_temporary_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runtime_state: str
) -> None:
    monkeypatch.chdir(tmp_path)
    temporary = tmp_path / "temp"
    temporary.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temporary))
    runtime = tmp_path / "runtime"
    if runtime_state == "unset":
        monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    elif runtime_state == "empty":
        monkeypatch.setenv("XDG_RUNTIME_DIR", "")
    else:
        if runtime_state == "file":
            runtime.touch()
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))

    lock = lock_path_for(tmp_path / "setup.yaml")

    root = temporary / f"qubit-measure-{getpass.getuser()}" / "locks"
    assert lock.parent == root
    assert root.is_dir()
    if os.name == "posix":
        assert stat.S_IMODE(root.stat().st_mode) == 0o700
