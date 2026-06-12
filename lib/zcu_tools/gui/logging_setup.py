"""Shared GUI logging setup (Phase 157).

One place that decides *how* every GUI entry point (the four ``script/run_*_gui``
launchers) and the measure MCP server configure file logging, so a new sibling
namespace is never silently left out of the log again.

Design decisions baked in here:

- The file handler is attached at the whole ``zcu_tools.gui`` namespace (plus any
  ``extra_namespaces`` an entry point needs, e.g. measure adds
  ``zcu_tools.experiment.v2_gui``). Attaching at the package root — not each app
  sub-namespace — means cross-cutting subpackages (``event_bus``, ``plotting``,
  ``remote``, ``session``) always reach the file. The root cause of the event
  this phase addresses was an app sub-namespace handler that missed a sibling.
- Per-session timestamped files under ``<root>/logs/<group>/<app>/``: each launch
  writes its own file, so a previous session's evidence is never overwritten
  (the old ``mode="w"`` fixed-file scheme discarded it). Retention purges all but
  the newest ``retain`` files in that directory on startup.
- The file handler stays at DEBUG; the stderr handler stays at WARNING. High-
  frequency bookkeeping logs at DEBUG so it can be filtered later without code
  changes.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

LOG_FORMAT = "%(asctime)s.%(msecs)03d [%(levelname)-7s] %(name)s: %(message)s"
LOG_DATE = "%H:%M:%S"

# Sortable per-session filename stem (lexical order == chronological order), so a
# plain ``sorted()`` of the directory is also a recency ordering for retention.
_SESSION_STAMP = "%Y-%m-%d_%H%M%S"

_DEFAULT_RETAIN = 10


def session_log_path(log_root: Path, group: str, app_name: str) -> Path:
    """Build the per-session log path ``<log_root>/logs/<group>/<app_name>/<stamp>.log``.

    ``group`` is ``"gui"`` for the GUI launchers and ``"mcp"`` for MCP servers.
    The directory is created here (Fast Fail: a non-creatable path raises now,
    not mid-session).
    """
    log_dir = log_root / "logs" / group / app_name
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime(_SESSION_STAMP)
    return log_dir / f"{stamp}.log"


def purge_old_logs(log_dir: Path, retain: int) -> None:
    """Delete all but the newest ``retain`` ``*.log`` files in ``log_dir``.

    Recency is decided by the sortable timestamped filename (newest sorts last),
    so this needs no stat calls. ``retain`` must be >= 1 (Fast Fail).
    """
    if retain < 1:
        raise ValueError(f"retain must be >= 1, got {retain}")
    if not log_dir.is_dir():
        return
    log_files = sorted(log_dir.glob("*.log"))
    for stale in log_files[:-retain]:
        stale.unlink(missing_ok=True)


def setup_gui_logging(
    *,
    app_name: str,
    log_root: Path,
    to_file: bool = True,
    log_file: Path | None = None,
    extra_namespaces: tuple[str, ...] = (),
    group: str = "gui",
    retain: int = _DEFAULT_RETAIN,
) -> Path | None:
    """Configure logging for a GUI entry point: DEBUG to file, WARNING to stderr.

    ``app_name`` names the per-app log directory (``measure``/``fluxdep``/...).
    ``log_root`` is the repo root the per-session ``logs/`` tree hangs under.
    ``log_file`` (an explicit ``--log-file`` override) wins over the per-session
    scheme: when given it is used verbatim and no retention purge runs. When
    omitted, a timestamped per-session file is created and old files are purged
    down to ``retain``.

    The file handler is attached at ``zcu_tools.gui`` plus every name in
    ``extra_namespaces``. Returns the resolved log file path, or ``None`` when
    ``to_file`` is False.
    """
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE))
    root.addHandler(stderr_handler)

    if not to_file:
        return None

    if log_file is not None:
        target = log_file
        target.parent.mkdir(parents=True, exist_ok=True)
    else:
        target = session_log_path(log_root, group, app_name)
        purge_old_logs(target.parent, retain)

    file_handler = logging.FileHandler(target, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE))

    for name in ("zcu_tools.gui", *extra_namespaces):
        log = logging.getLogger(name)
        log.addHandler(file_handler)
        log.setLevel(logging.DEBUG)

    print(f"[{app_name}] Logging DEBUG output to: {target}", file=sys.stderr)
    return target


__all__ = [
    "LOG_FORMAT",
    "LOG_DATE",
    "session_log_path",
    "purge_old_logs",
    "setup_gui_logging",
]
