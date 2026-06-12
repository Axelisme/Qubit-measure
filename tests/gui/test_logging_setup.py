"""Tests for the shared GUI logging-setup helper (Phase 157).

The regression at the heart of this phase: a file handler attached at an app
sub-namespace silently missed a sibling namespace (event_bus). These tests pin
that the helper attaches at the whole ``zcu_tools.gui`` namespace so sibling
records propagate to the file, plus the per-session file creation and retention.
"""

from __future__ import annotations

import logging
from pathlib import Path

from zcu_tools.gui.logging_setup import (
    purge_old_logs,
    session_log_path,
    setup_gui_logging,
)


def _detach(handlers_before: dict[str, list[logging.Handler]]) -> None:
    """Remove handlers added during a test from the named loggers + root.

    The logging module is process-global; these tests must not leak handlers
    into other tests. ``handlers_before`` is a snapshot taken before setup.
    """
    for name, before in handlers_before.items():
        log = logging.getLogger(name) if name else logging.getLogger()
        for handler in list(log.handlers):
            if handler not in before:
                log.removeHandler(handler)
                handler.close()


def _snapshot(names: tuple[str, ...]) -> dict[str, list[logging.Handler]]:
    snap: dict[str, list[logging.Handler]] = {}
    for name in names:
        log = logging.getLogger(name) if name else logging.getLogger()
        snap[name] = list(log.handlers)
    return snap


_WATCHED = (
    "",  # root
    "zcu_tools.gui",
    "zcu_tools.experiment.v2_gui",
)


def test_session_log_path_is_under_logs_group_app(tmp_path: Path) -> None:
    path = session_log_path(tmp_path, "gui", "measure")
    assert path.parent == tmp_path / "logs" / "gui" / "measure"
    assert path.parent.is_dir()  # created eagerly (Fast Fail)
    assert path.suffix == ".log"


def test_setup_creates_per_session_file_and_attaches_at_gui(tmp_path: Path) -> None:
    snap = _snapshot(_WATCHED)
    try:
        target = setup_gui_logging(
            app_name="measure",
            log_root=tmp_path,
            extra_namespaces=("zcu_tools.experiment.v2_gui",),
        )
        assert target is not None
        assert target.exists()
        assert target.parent == tmp_path / "logs" / "gui" / "measure"

        gui_handlers = logging.getLogger("zcu_tools.gui").handlers
        assert any(isinstance(h, logging.FileHandler) for h in gui_handlers)
        extra_handlers = logging.getLogger("zcu_tools.experiment.v2_gui").handlers
        assert any(isinstance(h, logging.FileHandler) for h in extra_handlers)
    finally:
        _detach(snap)


def test_sibling_namespace_record_reaches_file(tmp_path: Path) -> None:
    """A record logged on a sibling of the app namespace (event_bus / plotting /
    remote / session) reaches the file handler via propagation. This is the
    direct regression test for the missed-sibling bug."""
    snap = _snapshot(_WATCHED)
    try:
        target = setup_gui_logging(app_name="measure", log_root=tmp_path)
        assert target is not None
        for sibling in (
            "zcu_tools.gui.event_bus",
            "zcu_tools.gui.plotting.host",
            "zcu_tools.gui.remote.control_service",
            "zcu_tools.gui.session.services.connection",
        ):
            logging.getLogger(sibling).error("marker-%s", sibling)
        for handler in logging.getLogger("zcu_tools.gui").handlers:
            handler.flush()
        contents = target.read_text(encoding="utf-8")
        assert "marker-zcu_tools.gui.event_bus" in contents
        assert "marker-zcu_tools.gui.plotting.host" in contents
        assert "marker-zcu_tools.gui.remote.control_service" in contents
        assert "marker-zcu_tools.gui.session.services.connection" in contents
    finally:
        _detach(snap)


def test_no_file_when_to_file_false(tmp_path: Path) -> None:
    snap = _snapshot(_WATCHED)
    try:
        target = setup_gui_logging(app_name="measure", log_root=tmp_path, to_file=False)
        assert target is None
        assert not (tmp_path / "logs").exists()
    finally:
        _detach(snap)


def test_explicit_log_file_override_wins_and_skips_purge(tmp_path: Path) -> None:
    snap = _snapshot(_WATCHED)
    explicit = tmp_path / "custom" / "my.log"
    try:
        target = setup_gui_logging(
            app_name="measure", log_root=tmp_path, log_file=explicit
        )
        assert target == explicit
        assert explicit.exists()
        # The per-session logs/ tree is not created when an explicit file is given.
        assert not (tmp_path / "logs").exists()
    finally:
        _detach(snap)


def test_purge_keeps_newest_n(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs" / "gui" / "measure"
    log_dir.mkdir(parents=True)
    # Sortable stamped names: lexical order == chronological order.
    names = [f"2026-06-12_1500{i:02d}.log" for i in range(15)]
    for name in names:
        (log_dir / name).write_text("x", encoding="utf-8")
    purge_old_logs(log_dir, retain=10)
    remaining = sorted(p.name for p in log_dir.glob("*.log"))
    assert remaining == names[-10:]


def test_setup_purges_old_session_files_to_retain(tmp_path: Path) -> None:
    snap = _snapshot(_WATCHED)
    log_dir = tmp_path / "logs" / "gui" / "measure"
    log_dir.mkdir(parents=True)
    for i in range(12):
        (log_dir / f"2026-06-12_1400{i:02d}.log").write_text("old", encoding="utf-8")
    try:
        setup_gui_logging(app_name="measure", log_root=tmp_path, retain=10)
        # Purge runs first on the 12 old files (down to retain=10), then the new
        # session file is created on top → 11 files.
        remaining = list(log_dir.glob("*.log"))
        assert len(remaining) == 11
    finally:
        _detach(snap)
