from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pytest
from zcu_tools.gui import runtime
from zcu_tools.gui.remote.rpc_endpoint import ControlOptions
from zcu_tools.gui.runtime import (
    GuiAssembly,
    GuiLaunchOptions,
    GuiRuntimeBehavior,
    GuiRuntimeSpec,
    PlotPolicy,
    build_control_options,
    run_gui_runtime,
)


def test_build_control_options_disabled() -> None:
    spec = GuiRuntimeSpec(
        app_name="fluxdep",
        app_slug="fluxdep",
        plot_policy=PlotPolicy.EMBEDDED_BACKEND,
        default_control_port=8766,
    )
    options = GuiLaunchOptions(log_root=Path("."), no_control=True)

    assert build_control_options(spec, options) is None


def test_build_control_options_omitted_port_uses_default_with_fallback() -> None:
    spec = GuiRuntimeSpec(
        app_name="fluxdep",
        app_slug="fluxdep",
        plot_policy=PlotPolicy.EMBEDDED_BACKEND,
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
        plot_policy=PlotPolicy.EMBEDDED_BACKEND,
        default_control_port=8767,
    )
    options = GuiLaunchOptions(log_root=Path("."), control_port=9000)

    control = build_control_options(spec, options)

    assert control is not None
    assert control.port == 9000
    assert control.allow_port_fallback is False
    assert control.app_slug == "dispersive"


class _Signal:
    def __init__(self, events: list[str]) -> None:
        self._events = events
        self.callbacks: list[object] = []

    def connect(self, callback: object) -> None:
        self._events.append("connect")
        self.callbacks.append(callback)


class _App:
    def __init__(self, events: list[str], code: int = 0) -> None:
        self._events = events
        self.aboutToQuit = _Signal(events)
        self._code = code

    def exec(self) -> int:
        self._events.append("exec")
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
        plot_policy=PlotPolicy.NONE,
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


def test_run_gui_runtime_orders_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    app = _App(events, code=17)
    monkeypatch.setattr(runtime, "_get_or_create_qapplication", lambda: app)
    monkeypatch.setattr(
        runtime,
        "_configure_post_qt_plot_policy",
        lambda policy, app: events.append(f"post:{policy.value}"),
    )

    code = run_gui_runtime(
        _Behavior(events),
        ControlOptions(port=9999, app_slug="fake"),
    )

    assert code == 17
    assert events == [
        "post:none",
        "assemble",
        "before",
        "show",
        "start",
        "connect",
        "after",
        "exec",
    ]


def test_run_gui_runtime_reports_control_start_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    events: list[str] = []
    app = _App(events, code=17)
    monkeypatch.setattr(runtime, "_get_or_create_qapplication", lambda: app)
    monkeypatch.setattr(
        runtime,
        "_configure_post_qt_plot_policy",
        lambda _policy, _app: events.append("post"),
    )

    code = run_gui_runtime(
        _Behavior(events, fail_start=True),
        ControlOptions(port=9999, app_slug="fake", allow_port_fallback=False),
    )

    assert code == 1
    assert events == ["post", "assemble", "before", "show", "start"]
    assert "cannot open control socket" in capsys.readouterr().err


def test_run_gui_runtime_rejects_control_adapter_when_control_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    app = _App(events)
    monkeypatch.setattr(runtime, "_get_or_create_qapplication", lambda: app)
    monkeypatch.setattr(runtime, "_configure_post_qt_plot_policy", lambda *_: None)

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
