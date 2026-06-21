"""Per-session JSONL call-log for the MCP tool layer (Phase 166).

Each top-level MCP tool invocation is recorded as one JSON line:
  {ts, tool, input, output, status, duration_ms}      # success
  {ts, tool, input, output:null, status, error, duration_ms}  # error

Design decisions:
- Lazy file open: the log file is only created on the first write, so
  importing this module (and even calling wrap_handler) has zero I/O side
  effects during testing or in processes that never invoke a wrapped tool.
- Path convention mirrors logging_setup.session_log_path:
  <repo_root>/logs/mcp/measure/<timestamp>-calls.jsonl
  Keeps all measure MCP logs co-located for easy inspection.
- Env kill-switch: ZCU_MCP_CALL_LOG=0|false|no disables all writes so unit
  tests and CI can opt out with zero overhead.
- Never raises: every write is guarded by try/except so a log failure never
  breaks a tool call. Internal errors are written to stderr at most once.
- Flush per line: each JSON line is flushed immediately so a crash never
  loses the last entry.
- Field truncation at 8 KiB: large inputs/outputs (e.g. screenshots as
  base64) are truncated rather than written in full; they are only a hint.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TRUNCATE_BYTES = 8192
_DEFAULT_RETAIN = 10
_SESSION_STAMP = "%Y-%m-%d_%H%M%S"

# Resolved once at import time — same parent-chain logic as server.py comments:
# call_log.py -> core -> mcp -> zcu_tools -> lib -> repo root (parents[4]).
_REPO_ROOT: Path = Path(__file__).resolve().parents[4]

# ---------------------------------------------------------------------------
# Kill-switch: ZCU_MCP_CALL_LOG=0|false|no  →  entire module is no-op.
# ---------------------------------------------------------------------------


def _logging_enabled() -> bool:
    raw = os.environ.get("ZCU_MCP_CALL_LOG", "").strip().lower()
    return raw not in {"0", "false", "no"}


# ---------------------------------------------------------------------------
# Lazy file handle (module-level singleton for the lifetime of the process)
# ---------------------------------------------------------------------------

_log_file: TextIO | None = None
_open_attempted: bool = False
_internal_error_reported: bool = False


def _log_path() -> Path:
    """Build the per-session call-log path under <repo>/logs/mcp/measure/."""
    log_dir = _REPO_ROOT / "logs" / "mcp" / "measure"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime(_SESSION_STAMP)
    return log_dir / f"{stamp}-calls.jsonl"


def _purge_old_call_logs(log_dir: Path, retain: int) -> None:
    """Delete all but the newest ``retain`` ``*-calls.jsonl`` files in log_dir.

    Uses the same sortable-timestamp convention as purge_old_logs in
    logging_setup.py, but targets ``*-calls.jsonl`` instead of ``*.log``
    so the two purge functions remain independent.
    """
    if not log_dir.is_dir():
        return
    files = sorted(log_dir.glob("*-calls.jsonl"))
    for stale in files[:-retain]:
        try:
            stale.unlink(missing_ok=True)
        except OSError:
            pass  # Best-effort; skip locked files on Windows


def _get_log_file():  # type: ignore[return]
    """Return the open log file, opening it lazily on first call.

    Returns None if the kill-switch is active or if opening failed.
    """
    global _log_file, _open_attempted, _internal_error_reported

    if _open_attempted:
        return _log_file
    _open_attempted = True

    if not _logging_enabled():
        return None

    try:
        path = _log_path()
        _purge_old_call_logs(path.parent, _DEFAULT_RETAIN)
        _log_file = open(path, "w", encoding="utf-8")  # noqa: SIM115
    except Exception as exc:
        if not _internal_error_reported:
            _internal_error_reported = True
            print(
                f"[call_log] Failed to open call log: {exc}",
                file=sys.stderr,
            )
        _log_file = None

    return _log_file


# ---------------------------------------------------------------------------
# JSON serialization helpers
# ---------------------------------------------------------------------------


def _safe_dumps(obj: Any) -> str:
    """Serialize ``obj`` to a JSON string, falling back to repr on failure."""
    try:
        return json.dumps(obj, default=str)
    except Exception:
        try:
            return repr(obj)
        except Exception:
            return "<unrepresentable>"


def _truncate_if_needed(serialized: str) -> str:
    """Truncate ``serialized`` to _TRUNCATE_BYTES with an ellipsis marker.

    The marker includes the original byte count so readers know how much was
    dropped without needing the original value.
    """
    encoded = serialized.encode("utf-8")
    if len(encoded) <= _TRUNCATE_BYTES:
        return serialized
    n_bytes = len(encoded)
    truncated = encoded[:_TRUNCATE_BYTES].decode("utf-8", errors="replace")
    return truncated + f"…<truncated {n_bytes} bytes>"


def _serialize_field(obj: Any) -> Any:
    """Return a JSON-safe value for one log field, truncated if >8 KiB.

    The field is serialized to a string first (to measure bytes), then
    parsed back to a native type so the JSONL line stores structured JSON
    rather than a doubly-encoded string.  Falls back to the truncated string
    if round-trip parse fails.
    """
    raw = _safe_dumps(obj)
    truncated = _truncate_if_needed(raw)
    try:
        return json.loads(truncated)
    except Exception:
        # truncated string may not be valid JSON after mid-character cut —
        # store it as a plain string rather than crashing.
        return truncated


# ---------------------------------------------------------------------------
# Log write
# ---------------------------------------------------------------------------


def _write_entry(entry: dict[str, Any]) -> None:
    """Write one JSON line to the call log.  Never raises."""
    global _internal_error_reported
    try:
        fh = _get_log_file()
        if fh is None:
            return
        line = json.dumps(entry, default=str) + "\n"
        fh.write(line)
        fh.flush()
    except Exception as exc:
        if not _internal_error_reported:
            _internal_error_reported = True
            print(f"[call_log] Write failed: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

HandlerFn = Callable[[dict[str, Any]], Any]


def wrap_handler(name: str, handler: HandlerFn) -> HandlerFn:
    """Return a new handler that logs each invocation then delegates to ``handler``.

    The wrapper is transparent: it accepts the same arguments and returns (or
    re-raises) exactly what the original handler would.  The only side effect
    is writing one JSONL line per call.

    ``name`` is the MCP tool name used as the ``tool`` field in the log entry.
    """

    def _wrapped(arguments: dict[str, Any]) -> Any:
        ts = datetime.now().isoformat()
        t_start = datetime.now().timestamp()

        try:
            result = handler(arguments)
        except Exception as exc:
            duration_ms = round((datetime.now().timestamp() - t_start) * 1000, 3)
            error_text = str(exc)
            reason = getattr(exc, "reason", None)
            if reason:
                error_text = f"{error_text} [reason={reason}]"
            try:
                _write_entry(
                    {
                        "ts": ts,
                        "tool": name,
                        "input": _serialize_field(arguments),
                        "output": None,
                        "status": "error",
                        "error": _truncate_if_needed(error_text),
                        "duration_ms": duration_ms,
                    }
                )
            except Exception:
                pass  # log failure must never suppress the handler's exception
            raise  # re-raise original exception unchanged

        duration_ms = round((datetime.now().timestamp() - t_start) * 1000, 3)
        try:
            _write_entry(
                {
                    "ts": ts,
                    "tool": name,
                    "input": _serialize_field(arguments),
                    "output": _serialize_field(result),
                    "status": "success",
                    "duration_ms": duration_ms,
                }
            )
        except Exception:
            pass  # log failure must never suppress the handler's result
        return result

    return _wrapped


__all__ = ["wrap_handler"]
