"""Exercise cache protection at pytest's public plugin/session boundary."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from uuid import uuid4

import pytest
from pytest_cache_guard import GuiCacheGuard


def run_session(
    tmp_path: Path, source: str, plugin: GuiCacheGuard
) -> int | pytest.ExitCode:
    """Run an isolated in-process pytest session without this suite's conftest."""
    config = tmp_path / "pytest.ini"
    config.write_text("[pytest]\naddopts = --import-mode=importlib\n")
    # pytest's importlib mode keeps collected modules in sys.modules. A unique
    # package lets repeated sessions run fresh source without replacing imports.
    package_name = f"cache_session_{uuid4().hex}"
    session_dir = tmp_path / package_name
    session_dir.mkdir()
    test_file = session_dir / "test_cache_client.py"
    test_file.write_text(source)
    previous_path = sys.path[:]
    previous_current_test = os.environ.get("PYTEST_CURRENT_TEST")
    try:
        return pytest.main(
            [
                "-c",
                str(config),
                "--confcutdir",
                str(tmp_path),
                "-p",
                "no:cacheprovider",
                "-v",
                str(test_file),
            ],
            plugins=[plugin],
        )
    finally:
        sys.path[:] = previous_path
        if previous_current_test is None:
            os.environ.pop("PYTEST_CURRENT_TEST", None)
        else:
            os.environ["PYTEST_CURRENT_TEST"] = previous_current_test
        for name in list(sys.modules):
            if name == package_name or name.startswith(package_name + "."):
                del sys.modules[name]


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


@pytest.mark.parametrize("phase", ["collection", "setup", "call", "teardown"])
@pytest.mark.parametrize("mutation", ["create", "rewrite", "touch", "remove"])
def test_protected_cache_mutation_fails_with_polluting_node(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], phase: str, mutation: str
) -> None:
    previous_cache = os.environ.get("XDG_CACHE_HOME")
    real_cache = tmp_path / "synthetic-user-cache" / "gui"
    state_file = real_cache / "gui_state_v1.json"
    if mutation != "create":
        real_cache.mkdir(parents=True)
        state_file.write_text("user state")
    actions = {
        "create": "cache.mkdir(parents=True); state.write_text('polluted')",
        "rewrite": "state.write_text('changed state')",
        "touch": "os.utime(state, ns=(1_000_000_000, 1_000_000_000))",
        "remove": "state.unlink()",
    }
    action = actions[mutation]
    source = f"""
import os
from pathlib import Path
import pytest
cache = Path({str(real_cache)!r})
state = cache / "gui_state_v1.json"
"""
    if phase == "collection":
        source += f"\n{action}\n"
    source += f"""
@pytest.fixture(autouse=True)
def cache_client():
    {action if phase == "setup" else "pass"}
    yield
    {action if phase == "teardown" else "pass"}

def test_polluter():
    {action if phase == "call" else "pass"}
"""
    result = run_session(
        tmp_path,
        source,
        GuiCacheGuard(real_cache=real_cache, cache_root=tmp_path / "session-cache"),
    )
    output = capsys.readouterr().out
    assert result != pytest.ExitCode.OK, "protected cache mutations must fail pytest"
    assert "GUI cache pollution" in output
    assert str(real_cache) in output
    assert "test_cache_client.py" in output
    assert phase in output
    if phase != "collection":
        assert "test_polluter" in output
    assert os.environ.get("XDG_CACHE_HOME") == previous_cache
