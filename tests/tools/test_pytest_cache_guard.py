"""Exercise cache protection at pytest's public plugin/session boundary."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pytest_cache_guard import GuiCacheGuard


def run_session(
    tmp_path: Path, source: str, plugin: GuiCacheGuard
) -> int | pytest.ExitCode:
    """Run an isolated in-process pytest session without this suite's conftest."""
    config = tmp_path / "pytest.ini"
    config.write_text("[pytest]\naddopts = --import-mode=importlib\n")
    test_file = tmp_path / "test_cache_client.py"
    test_file.write_text(source)
    return pytest.main(
        [
            "-c",
            str(config),
            "--confcutdir",
            str(tmp_path),
            "-p",
            "no:cacheprovider",
            "-q",
            str(test_file),
        ],
        plugins=[plugin],
    )


def test_default_caretaker_writes_only_to_session_cache(tmp_path: Path) -> None:
    previous_cache = os.environ.get("XDG_CACHE_HOME")
    isolated_root = tmp_path / "session-cache"
    real_cache = tmp_path / "synthetic-user-cache" / "gui"
    plugin = GuiCacheGuard(real_cache=real_cache, cache_root=isolated_root)
    result = run_session(
        tmp_path,
        f"""
from pathlib import Path
from zcu_tools.gui.session.persistence import SingleFileCaretaker

class Originator:
    def capture_persisted_state(self) -> str:
        return "saved"
    def restore_persisted_state(self, state: str) -> None:
        pass

class Codec:
    def default(self) -> str:
        return "default"
    def decode(self, raw: object) -> str:
        return str(raw)
    def encode(self, state: str) -> object:
        return state

def test_default_save():
    caretaker = SingleFileCaretaker(
        Originator(), codec=Codec(), filename="gui_state_v1.json"
    )
    caretaker.flush()
    expected = Path({str(isolated_root)!r}) / "zcu_tools/gui/gui_state_v1.json"
    assert expected.read_text().strip() == '"saved"'
    assert caretaker.state_path == expected
""",
        plugin,
    )
    assert result == pytest.ExitCode.OK, "default GUI saves must use session cache"
    assert not real_cache.exists()
    assert os.environ.get("XDG_CACHE_HOME") == previous_cache
