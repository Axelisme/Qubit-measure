"""Verify session log routing, warning and exception records, and file retention."""

from __future__ import annotations

import logging
import sys
import threading
import warnings
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.app.measure.utils.error_handler import install_global_exception_hook
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
    "zcu_tools.gui.runtime",
    "zcu_tools.experiment.v2_gui",
    "zcu_tools.program.v2",
    "py.warnings",
)


@pytest.fixture(autouse=True, name="_restore_logger_levels")
def fixture_restore_logger_levels() -> Iterator[None]:
    """Restore process-global levels as well as the handlers detached by each test."""
    levels_before = {name: logging.getLogger(name).level for name in _WATCHED}
    yield
    for name, level in levels_before.items():
        logging.getLogger(name).setLevel(level)


@pytest.fixture(autouse=True, name="_restore_warning_capture")
def fixture_restore_warning_capture() -> Iterator[None]:
    with warnings.catch_warnings():
        original_showwarning = warnings.showwarning
        # Detect prior capture even if pytest replaced its warning hook.
        marker = MagicMock()
        warnings.showwarning = marker
        logging.captureWarnings(False)
        previous_capture = warnings.showwarning
        warnings.showwarning = original_showwarning
        try:
            yield
        finally:
            logging.captureWarnings(False)
            if previous_capture is not marker:
                warnings.showwarning = previous_capture
                logging.captureWarnings(True)


def test_session_log_path_is_under_logs_group_app(tmp_path: Path) -> None:
    path = session_log_path(tmp_path, "gui", "measure")
    assert path.parent == tmp_path / "logs" / "gui" / "measure"
    assert path.parent.is_dir()  # created eagerly (Fast Fail)
    assert path.suffix == ".log"


def test_setup_creates_per_session_file(tmp_path: Path) -> None:
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
    finally:
        _detach(snap)


def test_file_logging_uses_namespace_thresholds(tmp_path: Path) -> None:
    snap = _snapshot(_WATCHED)
    try:
        target = setup_gui_logging(
            app_name="measure",
            log_root=tmp_path,
            extra_namespaces=("zcu_tools.program.v2", "zcu_tools.gui.runtime"),
        )
        assert target is not None

        external = logging.getLogger("zcu_tools.simulators.example")
        external.warning("external warning marker")
        external.debug("external debug marker")
        logging.getLogger("zcu_tools.gui.runtime").debug("gui debug marker")
        logging.getLogger("zcu_tools.program.v2").debug("extra debug marker")
        logging.getLogger("zcu_tools.program.v20").debug("lookalike debug marker")

        contents = target.read_text(encoding="utf-8")
        assert contents.count("external warning marker") == 1
        assert "external debug marker" not in contents
        assert contents.count("gui debug marker") == 1
        assert contents.count("extra debug marker") == 1
        assert "lookalike debug marker" not in contents
    finally:
        _detach(snap)


def test_python_warning_reaches_file_and_stderr_once(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    snap = _snapshot(_WATCHED)
    try:
        target = setup_gui_logging(app_name="measure", log_root=tmp_path)
        assert target is not None
        with warnings.catch_warnings():
            warnings.simplefilter("always", UserWarning)
            warnings.warn("python warning marker", UserWarning, stacklevel=1)

        marker = "UserWarning: python warning marker"
        assert target.read_text(encoding="utf-8").count(marker) == 1
        assert capsys.readouterr().err.count(marker) == 1
    finally:
        _detach(snap)


def test_main_thread_exception_reaches_log_and_presenter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snap = _snapshot(_WATCHED)
    try:
        target = setup_gui_logging(app_name="measure", log_root=tmp_path)
        assert target is not None
        presenter = MagicMock()
        previous_hook = MagicMock()
        monkeypatch.setattr(sys, "excepthook", previous_hook)
        monkeypatch.setattr(threading, "excepthook", threading.excepthook)
        install_global_exception_hook(presenter)

        try:
            raise ValueError("unhandled main-thread marker")
        except ValueError as error:
            args = (type(error), error, error.__traceback__)
        sys.excepthook(*args)

        presenter.assert_called_once_with(*args)
        previous_hook.assert_called_once_with(*args)
        contents = target.read_text(encoding="utf-8")
        assert "[ERROR" in contents
        assert "Traceback (most recent call last):" in contents
        assert "ValueError: unhandled main-thread marker" in contents
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
        contents = target.read_text(encoding="utf-8")
        assert "marker-zcu_tools.gui.event_bus" in contents
        assert "marker-zcu_tools.gui.plotting.host" in contents
        assert "marker-zcu_tools.gui.remote.control_service" in contents
        assert "marker-zcu_tools.gui.session.services.connection" in contents
    finally:
        _detach(snap)


def test_extra_namespace_record_reaches_file(tmp_path: Path) -> None:
    snap = _snapshot(_WATCHED)
    try:
        target = setup_gui_logging(
            app_name="autofluxdep",
            log_root=tmp_path,
            extra_namespaces=("zcu_tools.program.v2",),
        )
        assert target is not None

        logging.getLogger("zcu_tools.program.v2.sim.engine").error("sim-marker")
        contents = target.read_text(encoding="utf-8")
        assert "sim-marker" in contents
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


def test_setup_replaces_managed_handlers_instead_of_stacking(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    snap = _snapshot(_WATCHED)
    first = tmp_path / "first.log"
    second = tmp_path / "second.log"
    try:
        setup_gui_logging(
            app_name="measure",
            log_root=tmp_path,
            log_file=first,
            extra_namespaces=("zcu_tools.program.v2",),
        )
        setup_gui_logging(app_name="measure", log_root=tmp_path, log_file=second)

        logging.getLogger("zcu_tools.gui.runtime").error("single-marker")
        logging.getLogger("zcu_tools.program.v2").debug("expired debug marker")

        assert first.read_text(encoding="utf-8") == ""
        contents = second.read_text(encoding="utf-8")
        assert contents.count("single-marker") == 1
        assert "expired debug marker" not in contents
        assert capsys.readouterr().err.count("single-marker") == 1
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


def test_purge_tolerates_unlink_oserror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # On Windows a stale log still held open by another running GUI/MCP-server
    # instance cannot be deleted (PermissionError / WinError 32). purge must skip
    # it and continue, NOT crash — else logging setup (and the whole process)
    # fails to start when several instances run concurrently.
    log_dir = tmp_path / "logs" / "mcp" / "measure"
    log_dir.mkdir(parents=True)
    names = [f"2026-06-12_1500{i:02d}.log" for i in range(13)]
    for name in names:
        (log_dir / name).write_text("x", encoding="utf-8")
    locked = log_dir / names[0]  # oldest stale file: pretend it is held open

    real_unlink = Path.unlink

    def fake_unlink(self: Path, missing_ok: bool = False) -> None:
        if self == locked:
            raise PermissionError(32, "in use by another process")
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", fake_unlink)

    purge_old_logs(log_dir, retain=10)  # must not raise

    remaining = sorted(p.name for p in log_dir.glob("*.log"))
    assert names[0] in remaining  # locked file skipped, survives
    assert names[1] not in remaining and names[2] not in remaining  # others purged
    assert all(n in remaining for n in names[-10:])  # newest 10 kept


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
