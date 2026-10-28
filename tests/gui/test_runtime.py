from __future__ import annotations

import logging
import sys
from functools import partial
from pathlib import Path
from threading import Event, Thread
from typing import ClassVar, TextIO

import pytest
from PyQt6.QtCore import QObject, pyqtSignal
from zcu_tools.gui import plotting, runtime
from zcu_tools.gui.remote.rpc_endpoint import ControlOptions
from zcu_tools.gui.runtime import (
    GuiAssembly,
    GuiExitDiagnostics,
    GuiLaunchOptions,
    GuiRuntimeBehavior,
    GuiRuntimeSpec,
    build_control_options,
    run_gui_runtime,
)


@pytest.fixture(autouse=True)
def restore_plotting_shutdown_state():
    """Restore the shared qapp after exercising standalone process shutdown."""
    # Production must keep shutdown=True through interpreter finalization.
    # These tests reuse qapp instead of exiting the process (ADR-0064).
    before = plotting.is_shutting_down()
    try:
        yield
    finally:
        plotting.set_shutting_down(value=before)


def test_build_control_options_disabled() -> None:
    spec = GuiRuntimeSpec(
        app_name="fluxdep",
        app_slug="fluxdep",
        default_control_port=8766,
    )
    options = GuiLaunchOptions(log_root=Path("."), no_control=True)

    assert build_control_options(spec, options) is None


def test_build_control_options_omitted_port_uses_default_with_fallback() -> None:
    spec = GuiRuntimeSpec(
        app_name="fluxdep",
        app_slug="fluxdep",
        default_control_port=8766,
    )
    options = GuiLaunchOptions(log_root=Path("."), control_token="token")

    control = build_control_options(spec, options)

    assert control is not None
    assert control.port == 8766
    assert control.token == "token"
    assert control.allow_port_fallback is True
    assert control.app_slug == "fluxdep"


def test_build_control_options_explicit_port_is_pinned() -> None:
    spec = GuiRuntimeSpec(
        app_name="dispersive",
        app_slug="dispersive",
        default_control_port=8767,
    )
    options = GuiLaunchOptions(log_root=Path("."), control_port=9000)

    control = build_control_options(spec, options)

    assert control is not None
    assert control.port == 9000
    assert control.allow_port_fallback is False
    assert control.app_slug == "dispersive"


@pytest.mark.parametrize(
    "extra_namespaces", [(), ("user_experiments",)], ids=["default", "injected"]
)
def test_launch_configures_logging_before_constructing_behavior(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, extra_namespaces: tuple[str, ...]
) -> None:
    events: list[str] = []
    namespaces: list[tuple[str, ...]] = []
    fixed_namespaces = ("runtime_capability",)

    def configure_logging(
        *,
        app_name: str,
        log_root: Path,
        to_file: bool,
        log_file: Path | None,
        extra_namespaces: tuple[str, ...],
        group: str,
    ) -> Path | None:
        del app_name, log_root, to_file, log_file, group
        namespaces.append(extra_namespaces)
        events.append("logging")
        return None

    class LaunchBehavior(_Behavior):
        spec: ClassVar[GuiRuntimeSpec] = GuiRuntimeSpec(
            app_name="launch",
            app_slug="launch",
            default_control_port=9999,
            logging_extra_namespaces=fixed_namespaces,
        )

        def __init__(self, received_events: list[str]) -> None:
            assert namespaces == [fixed_namespaces + extra_namespaces]
            events.append("construct")
            super().__init__(received_events)

    def run_behavior(
        behavior: GuiRuntimeBehavior,
        control: ControlOptions | None,
        *,
        exit_diagnostics: GuiExitDiagnostics | None = None,
    ) -> int:
        assert isinstance(behavior, LaunchBehavior)
        assert control is None
        events.append("run")
        return 17

    monkeypatch.setattr(runtime, "setup_gui_logging", configure_logging)
    monkeypatch.setattr(runtime, "run_gui_runtime", run_behavior)

    result = runtime.launch_gui_runtime(
        LaunchBehavior.spec,
        GuiLaunchOptions(log_root=tmp_path, no_control=True),
        behavior_factory=partial(LaunchBehavior, events),
        logging_namespaces=extra_namespaces,
    )

    assert result == 17
    assert events == ["logging", "construct", "run"]
    assert LaunchBehavior.spec.logging_extra_namespaces == fixed_namespaces


def test_teardown_summary_identifies_live_non_daemon_thread(
    qapp, caplog: pytest.LogCaptureFixture
) -> None:
    from qtpy.QtCore import QTimer

    release = Event()
    worker = Thread(target=release.wait, name="exit-diagnostic-worker", daemon=False)

    class ClosingBehavior(_Behavior):
        def after_show(self, assembly: GuiAssembly) -> None:
            QTimer.singleShot(0, qapp.closeAllWindows)

        def assemble(self, control: ControlOptions | None) -> GuiAssembly:
            from qtpy.QtWidgets import QWidget

            return GuiAssembly(controller=object(), window=QWidget())

        def close(self, assembly: GuiAssembly | None) -> None:
            logging.getLogger("zcu_tools.gui.runtime").info("app resources closed")

    worker.start()
    try:
        with caplog.at_level(logging.INFO, logger="zcu_tools.gui.runtime"):
            assert run_gui_runtime(ClosingBehavior([]), None) == 0
        summary = next(
            record.message
            for record in caplog.records
            if "teardown complete" in record.message
        )
        assert "name='exit-diagnostic-worker'" in summary
        assert "daemon=False" in summary
        assert f"ident={worker.ident}" in summary
        assert caplog.messages.index("app resources closed") < caplog.messages.index(
            summary
        )
    finally:
        release.set()
        # release guarantees the worker can finish; join is only a deadlock guard.
        worker.join(timeout=30)
        assert not worker.is_alive()


@pytest.mark.parametrize("failure", [None, "startup", "close"])
def test_runtime_releases_only_its_quit_subscriptions(
    qapp, failure: str | None
) -> None:
    from qtpy.QtCore import QTimer
    from zcu_tools.gui.lifetime import LifetimeScope

    events: list[str] = []

    class ScopedBehavior(_Behavior):
        def before_show(self, assembly: GuiAssembly) -> None:
            if failure == "startup":
                raise RuntimeError("startup failed")

        def after_show(self, assembly: GuiAssembly) -> None:
            QTimer.singleShot(0, qapp.quit)

        def close(self, assembly: GuiAssembly | None) -> None:
            if failure == "close":
                raise RuntimeError("close failed")

    external = LifetimeScope()
    external.connect(qapp.aboutToQuit, lambda: events.append("external"))
    try:
        before = qapp.receivers(qapp.aboutToQuit)
        behavior = ScopedBehavior(events)
        control = ControlOptions(port=9999, app_slug="fake")
        if failure is None:
            assert run_gui_runtime(behavior, control) == 0
        else:
            with pytest.raises(RuntimeError, match=f"{failure} failed"):
                run_gui_runtime(behavior, control)
        assert qapp.receivers(qapp.aboutToQuit) == before
        # Production still suppresses canvas removal through process exit.
        assert plotting.is_shutting_down() is (failure != "startup")
        events.clear()
        plotting.set_shutting_down(value=False)
        qapp.aboutToQuit.emit()
        assert events == ["external"]
        assert not plotting.is_shutting_down()
    finally:
        external.close()


class _RecordingWatchdog:
    def __init__(self) -> None:
        self.requests: list[tuple[float, bool, TextIO]] = []

    def dump_traceback_later(
        self, timeout: float, *, file: TextIO, exit: bool = False
    ) -> None:
        self.requests.append((timeout, exit, file))


@pytest.mark.parametrize("to_file", [True, False], ids=["log-file", "stderr"])
def test_runtime_arms_exit_watchdog_with_session_log_or_stderr(
    qapp, tmp_path: Path, to_file: bool, capsys: pytest.CaptureFixture[str]
) -> None:
    from qtpy.QtCore import QTimer

    watchdog = _RecordingWatchdog()
    log_file = tmp_path / "session.log" if to_file else None
    diagnostics = GuiExitDiagnostics(log_file, timeout=9, watchdog=watchdog)
    try:
        QTimer.singleShot(0, qapp.quit)
        assert run_gui_runtime(_Behavior([]), None, exit_diagnostics=diagnostics) == 0
        assert len(watchdog.requests) == 1
        timeout, exits_process, output = watchdog.requests[0]
        assert timeout == 9
        assert exits_process is False
        assert not output.closed
        if log_file is not None:
            assert Path(output.name) == log_file
            assert output.fileno() >= 0
            header = log_file.read_text()
        else:
            assert output is sys.stderr
            header = capsys.readouterr().err
        assert "GUI exit watchdog armed" in header
        assert "9 seconds" in header
    finally:
        # The recorder owns the injected output, just as faulthandler does.
        for _, _, output in watchdog.requests:
            if to_file:
                output.close()


def test_exit_diagnostics_appends_stacks_without_replacing_session_log(
    tmp_path: Path,
) -> None:
    log_file = tmp_path / "session.log"
    log_file.write_text("normal GUI teardown\n")
    watchdog = _RecordingWatchdog()
    GuiExitDiagnostics(log_file, watchdog=watchdog).arm()
    timeout, exits_process, output = watchdog.requests[0]
    try:
        assert timeout == 5
        assert exits_process is False
        assert log_file.read_text().startswith("normal GUI teardown\n")
        # Even if repositioned, the append fd must not overwrite session events.
        output.seek(0)
        output.write("diagnostic stack output\n")
        output.flush()
        content = log_file.read_text()
        assert content.startswith("normal GUI teardown\n")
        assert "GUI exit watchdog armed" in content
        assert content.endswith("diagnostic stack output\n")
    finally:
        output.close()


class _App(QObject):
    aboutToQuit = pyqtSignal()

    def __init__(self, events: list[str], code: int = 0) -> None:
        super().__init__()
        self._events = events
        self._code = code

    def exec(self) -> int:
        self._events.append("exec")
        self.aboutToQuit.emit()
        return self._code


class _Window:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    def show(self) -> None:
        self._events.append("show")


class _Adapter:
    def __init__(self, events: list[str], *, fail: bool = False) -> None:
        self._events = events
        self._fail = fail

    def start(self) -> int:
        self._events.append("start")
        if self._fail:
            raise RuntimeError("bind failed")
        return 12345

    def stop(self) -> None:
        self._events.append("stop")


class _Behavior(GuiRuntimeBehavior):
    spec: ClassVar[GuiRuntimeSpec] = GuiRuntimeSpec(
        app_name="fake",
        app_slug="fake",
        default_control_port=9999,
    )

    def __init__(self, events: list[str], *, fail_start: bool = False) -> None:
        self._events = events
        self._fail_start = fail_start

    def assemble(self, control: ControlOptions | None) -> GuiAssembly:
        self._events.append("assemble")
        return GuiAssembly(
            controller=object(),
            window=_Window(self._events),
            control_adapter=(
                _Adapter(self._events, fail=self._fail_start)
                if control is not None
                else None
            ),
        )

    def before_show(self, assembly: GuiAssembly) -> None:
        del assembly
        self._events.append("before")

    def after_show(self, assembly: GuiAssembly) -> None:
        del assembly
        self._events.append("after")


@pytest.fixture
def rendering_events(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    events: list[str] = []
    monkeypatch.setattr(plotting, "ensure_host", lambda: events.append("host"))
    monkeypatch.setattr(
        plotting, "install_mathtext_lock", lambda: events.append("mathtext-lock")
    )
    monkeypatch.setattr(
        plotting, "prewarm_mathtext", lambda: events.append("mathtext-prewarm")
    )
    monkeypatch.setattr(
        plotting, "set_shutting_down", lambda value: events.append(f"shutdown:{value}")
    )
    return events


def test_run_gui_runtime_orders_lifecycle(
    monkeypatch: pytest.MonkeyPatch, rendering_events: list[str]
) -> None:
    events = rendering_events
    app = _App(events, code=17)
    monkeypatch.setattr(runtime, "_get_or_create_qapplication", lambda: app)

    code = run_gui_runtime(
        _Behavior(events),
        ControlOptions(port=9999, app_slug="fake"),
    )

    assert code == 17
    assert events == [
        "host",
        "mathtext-lock",
        "mathtext-prewarm",
        "assemble",
        "before",
        "show",
        "start",
        "after",
        "exec",
        "shutdown:True",
        "stop",
    ]


def test_run_gui_runtime_reports_control_start_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    rendering_events: list[str],
) -> None:
    events = rendering_events
    app = _App(events, code=17)
    monkeypatch.setattr(runtime, "_get_or_create_qapplication", lambda: app)

    code = run_gui_runtime(
        _Behavior(events, fail_start=True),
        ControlOptions(port=9999, app_slug="fake", allow_port_fallback=False),
    )

    assert code == 1
    assert events == [
        "host",
        "mathtext-lock",
        "mathtext-prewarm",
        "assemble",
        "before",
        "show",
        "start",
    ]
    assert "cannot open control socket" in capsys.readouterr().err


def test_run_gui_runtime_rejects_control_adapter_when_control_disabled(
    monkeypatch: pytest.MonkeyPatch, rendering_events: list[str]
) -> None:
    events = rendering_events
    app = _App(events)
    monkeypatch.setattr(runtime, "_get_or_create_qapplication", lambda: app)

    class BadBehavior(_Behavior):
        def assemble(self, control: ControlOptions | None) -> GuiAssembly:
            del control
            return GuiAssembly(
                controller=object(),
                window=_Window(events),
                control_adapter=_Adapter(events),
            )

    with pytest.raises(RuntimeError, match="control adapter"):
        run_gui_runtime(BadBehavior(events), None)
