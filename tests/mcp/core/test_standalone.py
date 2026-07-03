from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from zcu_tools.mcp._standalone import bootstrap_standalone_server


def _fake_server_path(tmp_path: Path) -> Path:
    return tmp_path / "repo" / "lib" / "zcu_tools" / "mcp" / "app" / "server.py"


def test_bootstrap_standalone_server_inserts_lib_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server_file = _fake_server_path(tmp_path)
    lib_dir = server_file.parents[3]
    monkeypatch.setattr(sys, "path", [])

    resolved = bootstrap_standalone_server(server_file)

    assert resolved == lib_dir
    assert sys.path == [str(lib_dir)]


def test_bootstrap_standalone_server_fails_fast_for_missing_module(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def missing_spec(name: str):
        assert name == "missing_dep"
        return None

    monkeypatch.setattr(importlib.util, "find_spec", missing_spec)

    with pytest.raises(SystemExit) as exc_info:
        bootstrap_standalone_server(
            _fake_server_path(tmp_path),
            required_modules=(
                ("missing_dep", "server requires {module}; install extras\n"),
            ),
        )

    assert exc_info.value.code == 1
    assert "server requires missing_dep" in capsys.readouterr().err
