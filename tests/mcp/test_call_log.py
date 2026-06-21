"""Tests for mcp.core.call_log — Phase 166.

All tests use a temporary directory as the log root and manipulate the module's
internal state directly (via monkeypatching) to keep the tests hermetic.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Helpers for resetting the lazy-open module state between tests
# ---------------------------------------------------------------------------


def _reload_call_log(monkeypatch, tmp_path: Path, enabled: bool = True):
    """Reload call_log with a patched _REPO_ROOT pointing to tmp_path.

    Also sets / clears ZCU_MCP_CALL_LOG env var so the kill-switch is
    deterministic, and resets the lazy-open singleton state.
    """
    if enabled:
        monkeypatch.delenv("ZCU_MCP_CALL_LOG", raising=False)
    else:
        monkeypatch.setenv("ZCU_MCP_CALL_LOG", "0")

    # Remove cached module so re-import resets module globals.
    sys.modules.pop("zcu_tools.mcp.core.call_log", None)

    import zcu_tools.mcp.core.call_log as mod

    # Point _REPO_ROOT at tmp_path so log files land there.
    monkeypatch.setattr(mod, "_REPO_ROOT", tmp_path)
    # Reset lazy-open state so the first _get_log_file() call opens a fresh file.
    monkeypatch.setattr(mod, "_log_file", None)
    monkeypatch.setattr(mod, "_open_attempted", False)
    monkeypatch.setattr(mod, "_internal_error_reported", False)

    return mod


def _collect_jsonl_entries(log_dir: Path) -> list[dict[str, Any]]:
    """Parse all *-calls.jsonl files in log_dir and return a flat list of entries."""
    entries: list[dict[str, Any]] = []
    for path in sorted(log_dir.glob("*-calls.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_success_entry_written(monkeypatch, tmp_path):
    """A successful handler call writes one JSONL entry with expected fields."""
    mod = _reload_call_log(monkeypatch, tmp_path)

    def my_handler(arguments: dict) -> dict:
        return {"value": 42}

    wrapped = mod.wrap_handler("gui_foo", my_handler)
    result = wrapped({"x": 1})

    assert result == {"value": 42}

    log_dir = tmp_path / "logs" / "mcp" / "measure"
    entries = _collect_jsonl_entries(log_dir)
    assert len(entries) == 1

    e = entries[0]
    assert e["tool"] == "gui_foo"
    assert e["status"] == "success"
    assert e["input"] == {"x": 1}
    assert e["output"] == {"value": 42}
    assert isinstance(e["duration_ms"], float | int)
    assert "ts" in e


def test_error_entry_written_and_reraises(monkeypatch, tmp_path):
    """A failing handler: error entry is written and the original exception is re-raised."""
    mod = _reload_call_log(monkeypatch, tmp_path)

    class MyError(RuntimeError):
        pass

    def bad_handler(arguments: dict) -> dict:
        raise MyError("something went wrong")

    wrapped = mod.wrap_handler("gui_bar", bad_handler)

    with pytest.raises(MyError, match="something went wrong"):
        wrapped({})

    log_dir = tmp_path / "logs" / "mcp" / "measure"
    entries = _collect_jsonl_entries(log_dir)
    assert len(entries) == 1

    e = entries[0]
    assert e["tool"] == "gui_bar"
    assert e["status"] == "error"
    assert e["output"] is None
    assert "something went wrong" in e["error"]
    assert isinstance(e["duration_ms"], float | int)


def test_large_input_truncated(monkeypatch, tmp_path):
    """Input fields larger than 8 KiB are truncated with the ellipsis marker."""
    mod = _reload_call_log(monkeypatch, tmp_path)

    big_value = "A" * 20_000  # well over 8 KiB when JSON-serialized

    def handler(arguments: dict) -> dict:
        return {"ok": True}

    wrapped = mod.wrap_handler("gui_big", handler)
    wrapped({"data": big_value})

    log_dir = tmp_path / "logs" / "mcp" / "measure"
    entries = _collect_jsonl_entries(log_dir)
    assert len(entries) == 1

    # The input field was serialized; the string containing 20000 A's would be
    # >8 KiB, so the JSONL entry's input field must be a truncated string.
    raw_line = next(
        (log_dir / f).read_text(encoding="utf-8")
        for f in sorted(os.listdir(log_dir))
        if f.endswith("-calls.jsonl")
    )
    assert "truncated" in raw_line


def test_large_output_truncated(monkeypatch, tmp_path):
    """Output fields larger than 8 KiB are truncated with the ellipsis marker."""
    mod = _reload_call_log(monkeypatch, tmp_path)

    big_output = "B" * 20_000

    def handler(arguments: dict) -> str:
        return big_output

    wrapped = mod.wrap_handler("gui_bigout", handler)
    wrapped({})

    log_dir = tmp_path / "logs" / "mcp" / "measure"
    raw_line = next(
        (log_dir / f).read_text(encoding="utf-8")
        for f in sorted(os.listdir(log_dir))
        if f.endswith("-calls.jsonl")
    )
    assert "truncated" in raw_line


def test_unserializable_input_does_not_crash(monkeypatch, tmp_path):
    """An input containing a non-JSON-serializable object falls back to repr."""
    mod = _reload_call_log(monkeypatch, tmp_path)

    class Weird:
        def __repr__(self):
            return "<Weird object>"

    def handler(arguments: dict) -> dict:
        return {"ok": True}

    wrapped = mod.wrap_handler("gui_weird", handler)
    # Should not raise even though Weird() is not normally JSON-serializable.
    result = wrapped({"obj": Weird()})
    assert result == {"ok": True}

    log_dir = tmp_path / "logs" / "mcp" / "measure"
    entries = _collect_jsonl_entries(log_dir)
    # Entry was written (possibly with repr fallback in the value)
    assert len(entries) == 1


def test_kill_switch_no_file_written(monkeypatch, tmp_path):
    """ZCU_MCP_CALL_LOG=0 disables all logging — no file is created."""
    mod = _reload_call_log(monkeypatch, tmp_path, enabled=False)

    def handler(arguments: dict) -> dict:
        return {"ok": True}

    wrapped = mod.wrap_handler("gui_noop", handler)
    result = wrapped({"x": 1})
    assert result == {"ok": True}

    log_dir = tmp_path / "logs" / "mcp" / "measure"
    # Directory should not exist (or contain no calls files) when disabled
    call_files = list(log_dir.glob("*-calls.jsonl")) if log_dir.is_dir() else []
    assert call_files == []


@pytest.mark.parametrize("env_val", ["0", "false", "no", "False", "NO", "False"])
def test_kill_switch_variants(monkeypatch, tmp_path, env_val):
    """All recognized kill-switch values disable logging."""
    monkeypatch.setenv("ZCU_MCP_CALL_LOG", env_val)
    sys.modules.pop("zcu_tools.mcp.core.call_log", None)
    import zcu_tools.mcp.core.call_log as mod

    monkeypatch.setattr(mod, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(mod, "_log_file", None)
    monkeypatch.setattr(mod, "_open_attempted", False)
    monkeypatch.setattr(mod, "_internal_error_reported", False)

    wrapped = mod.wrap_handler("gui_x", lambda args: {"ok": True})
    wrapped({})

    log_dir = tmp_path / "logs" / "mcp" / "measure"
    call_files = list(log_dir.glob("*-calls.jsonl")) if log_dir.is_dir() else []
    assert call_files == []


def test_purge_keeps_newest_n(monkeypatch, tmp_path):
    """Opening the log purges old *-calls.jsonl files, keeping only the newest N."""

    log_dir = tmp_path / "logs" / "mcp" / "measure"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Pre-create 12 stale call-log files with sortable timestamps
    for i in range(12):
        (log_dir / f"2026-01-{i + 1:02d}_000000-calls.jsonl").write_text(
            '{"old": true}\n', encoding="utf-8"
        )

    # Now reload and trigger the first write (which runs purge)
    monkeypatch.delenv("ZCU_MCP_CALL_LOG", raising=False)
    sys.modules.pop("zcu_tools.mcp.core.call_log", None)
    import zcu_tools.mcp.core.call_log as mod2  # fresh module

    monkeypatch.setattr(mod2, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(mod2, "_log_file", None)
    monkeypatch.setattr(mod2, "_open_attempted", False)
    monkeypatch.setattr(mod2, "_internal_error_reported", False)

    # Trigger lazy open (writes a new file as entry 13)
    wrapped = mod2.wrap_handler("gui_purge", lambda args: {"ok": True})
    wrapped({})

    remaining = sorted(log_dir.glob("*-calls.jsonl"))
    # purge runs before the new file is opened: 12 old files → purge deletes 2
    # (sorted old[:-10]) → 10 old remain → new session file is then opened → 11
    # total.  This mirrors logging_setup.py's behaviour where purge precedes open.
    assert len(remaining) == 11


def test_write_failure_does_not_affect_handler(monkeypatch, tmp_path):
    """A log write failure must not propagate — the handler result is still returned."""
    mod = _reload_call_log(monkeypatch, tmp_path)

    # Patch _write_entry to always raise
    def _bad_write(entry):
        raise OSError("disk full")

    monkeypatch.setattr(mod, "_write_entry", _bad_write)

    def handler(arguments: dict) -> dict:
        return {"ok": True}

    wrapped = mod.wrap_handler("gui_diskfull", handler)
    # Should not raise despite write failure
    result = wrapped({"x": 1})
    assert result == {"ok": True}


def test_wrap_handler_transparent(monkeypatch, tmp_path):
    """The wrapped handler returns the same value as the original handler."""
    mod = _reload_call_log(monkeypatch, tmp_path)

    expected = {"nested": [1, 2, 3], "flag": True}

    def handler(arguments: dict) -> dict:
        return {**expected, "echo": arguments.get("key")}

    wrapped = mod.wrap_handler("gui_echo", handler)
    result = wrapped({"key": "val"})

    assert result == {**expected, "echo": "val"}
