"""Live GUI-side RPC contracts for a service-owned interactive analysis."""

from __future__ import annotations

import base64
import socket
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal
from uuid import uuid4

import numpy as np
import pytest
from matplotlib.backend_bases import MouseButton, MouseEvent
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from qtpy.QtWidgets import QPushButton
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.onetone.flux_dep import FluxDepResult
from zcu_tools.experiment.v2.twotone.fluxdep import FreqFluxResult
from zcu_tools.experiment.v2_gui.measure.adapters._support import FluxPickParams
from zcu_tools.experiment.v2_gui.measure.adapters._support.flux_pick_frontend import (
    FluxPickFrontend,
)
from zcu_tools.experiment.v2_gui.measure.adapters._support.flux_pick_plugin import (
    make_flux_pick_plugin,
    render_flux_pick,
)
from zcu_tools.gui.app.measure.adapter import AnalyzeRequest
from zcu_tools.gui.app.measure.services.guard import AnalyzePermit
from zcu_tools.gui.app.measure.ui.main_window import MainWindow
from zcu_tools.gui.session.adapters.qt_owner_scheduler import QtOwnerScheduler
from zcu_tools.plotting.plots import NonPresentingHost, Plots
from zcu_tools.resources.context import MetaDict, ModuleLibrary

from ._helpers import Fixture, mcp_client, open_client, recv_response, send


class DeferredBackground:
    def __init__(self) -> None:
        self.pending = []

    def run_background(self, compute, on_done, on_error) -> None:
        self.pending.append((compute, on_done, on_error))


class InteractiveFixture(Fixture):
    def __init__(self) -> None:
        super().__init__(active_label="interactive-test")
        self.widgets: list[FluxPickFrontend] = []


@pytest.fixture
def fx(qapp):
    fixture = InteractiveFixture()
    fixture.start()
    yield fixture
    fixture.ctrl._background_svc.quiesce()  # pyright: ignore[reportPrivateUsage] - queued owner deliveries precede widget GC
    for widget in fixture.widgets:
        widget.teardown()
        widget.deleteLater()
    fixture.stop()
    qapp.processEvents()


@pytest.fixture
def mounted_fx(qapp):
    fixture = InteractiveFixture()
    fixture.state.set_context(
        replace(
            fixture.state.session_env,
            md=MetaDict(),
            ml=ModuleLibrary(),
            soc=None,
            soccfg=None,
        )
    )
    window = MainWindow(fixture.ctrl)
    fixture.ctrl.add_view(window)
    fixture.service.render_view = window
    fixture.start()
    window.show()
    qapp.processEvents()
    yield fixture, window
    fixture.ctrl._background_svc.quiesce()  # pyright: ignore[reportPrivateUsage] - owner deliveries before Qt teardown
    fixture.stop()
    window.deleteLater()
    qapp.processEvents()


def _start_mounted(
    fx: InteractiveFixture,
    window: MainWindow,
    adapter: str,
    *,
    analyze_params: FluxPickParams | None = None,
    pane: Literal["run", "data"] | None = None,
):
    tab_id = fx.ctrl.new_tab(adapter)
    devs = np.linspace(-5.0, 5.0, 60)
    freqs = np.linspace(4.0, 5.0, 30)
    signals = np.exp(-(devs[:, None] ** 2)) * np.exp(
        1j * devs[:, None] * freqs[None, :] / 10
    )
    # Synthetic run input is fixture setup; starting analysis, mounting, RPC,
    # result publication and writeback must all use their production seams.
    run_result = (
        RunRecord(cfg=None, result=FluxDepResult(devs, freqs, signals))
        if adapter == "onetone/flux_dep"
        else RunRecord(cfg=None, result=FreqFluxResult(devs, freqs, signals))
    )
    fx.state.get_tab(tab_id).run.result = run_result
    tab_widget = window._tab_widgets[tab_id]  # pyright: ignore[reportPrivateUsage] - test fixture locates the mounted presentation
    if pane is not None:
        tab_widget.select_pane(pane)
    token = fx.ctrl.run_analyze_control.analyze(
        tab_id, FluxPickParams() if analyze_params is None else analyze_params
    )
    widget = tab_widget.interactive_frontend()
    assert isinstance(widget, FluxPickFrontend)
    return tab_id, token, widget


def _start(fx, *, background=None):
    tab_id = fx.ctrl.new_tab("fake")
    devs = np.linspace(-5.0, 5.0, 60)
    freqs = np.linspace(4.0, 5.0, 30)
    signals = np.exp(-(devs[:, None] ** 2)) * np.ones((1, 30))
    plots = Plots(NonPresentingHost())
    analyze_params = FluxPickParams()
    plugin = make_flux_pick_plugin(
        AnalyzeRequest(
            run_result=SimpleNamespace(signals=signals, values=devs, freqs=freqs),
            analyze_params=analyze_params,
            md=MetaDict(),
            ml=ModuleLibrary(),
            predictor=None,
        ),
        force_magnitude=True,
        plots=plots,
        result_builder=render_flux_pick,
    )
    plugin.bind_background(background or fx.ctrl.run_background)
    # Set up a real AnalyzeService operation; RPC interactions below always go
    # through the shipped socket and RunAnalyzeControlFacet, not a handler stub.
    token = fx.ctrl._analyze_svc.start_plugin(
        AnalyzePermit(tab_id=tab_id),
        plugin,
        QtOwnerScheduler(),
        analyze_params_instance=analyze_params,
        plots=plots,
    )
    fx.view.interactive_presentation.return_value = None
    return tab_id, token, plugin


def _rpc(sock: socket.socket, method: str, params: dict[str, Any]) -> dict[str, Any]:
    rid = uuid4().hex
    send(sock, {"id": rid, "method": method, "params": params})
    return recv_response(sock, rid)


def _interact(
    sock: socket.socket, tab_id: str, payload=None, **params
) -> dict[str, Any]:
    args = {"tab_id": tab_id, **params}
    if payload is not None:
        args["payload"] = payload
    return _rpc(sock, "tab.interact", args)


def _button(widget: FluxPickFrontend, name: str) -> QPushButton:
    return next(
        button for button in widget.findChildren(QPushButton) if button.text() == name
    )


def _pointer(canvas: FigureCanvasQTAgg, name: str, x: float, y: float = 4.5) -> None:
    px, py = canvas.figure.axes[0].transData.transform((x, y))
    event = MouseEvent(name, canvas, int(px), int(py), button=MouseButton.LEFT)
    canvas.callbacks.process(name, event)


def test_socket_discovery_commands_done_and_headless_figure(fx) -> None:
    tab_id, token, _plugin = _start(fx)
    fx.service.render_view = None
    with open_client(fx.service.port) as sock:
        initial = _interact(sock, tab_id)
        assert initial["ok"] is True
        result = initial["result"]
        assert result["plugin"] == "flux_pick"
        assert result["operation_id"] == token
        assert result["figure"] is None
        assert result["preview_active"] is False
        names = [item["name"] for item in result["commands"]]
        assert names == [
            "move_line",
            "set_conjugate",
            "swap_lines",
            "auto_align",
            "done",
        ]
        assert result["commands"][0]["params"]["required"] == ["role", "position"]
        start = result["state"]
        moved = _interact(
            sock,
            tab_id,
            {
                "command": "move_line",
                "args": {"role": "half", "position": start["flux_half"] + 0.6},
            },
        )
        assert moved["result"]["state"]["flux_half"] == pytest.approx(
            start["flux_half"] + 0.6
        )
        assert (
            _interact(
                sock, tab_id, {"command": "set_conjugate", "args": {"enabled": True}}
            )["result"]["state"]["conjugate"]
            is True
        )
        committed = _interact(sock, tab_id)["result"]["state"]
        done = _interact(sock, tab_id, {"command": "done"})
        assert done["result"]["state"] == committed
        assert done["result"]["operation_id"] == token
        assert fx.ctrl.get_tab_snapshot(tab_id).analysis.source_operation_id == token
        assert base64.b64decode(done["result"]["figure"]["png_b64"]).startswith(
            b"\x89PNG"
        )
        assert fx.ctrl.get_tab_analyze_result(tab_id).flx_half == pytest.approx(
            committed["flux_half"]
        )
        settled = _rpc(sock, "operation.await", {"operation_id": token, "timeout": 0.1})
        assert settled["result"]["status"] == "finished"
        assert _interact(sock, tab_id)["error"]["code"] == "precondition_failed"


def test_socket_done_receipt_skips_png_and_settles_original_operation(
    fx, monkeypatch
) -> None:
    tab_id, token, _plugin = _start(fx)

    def unavailable_renderer(_figure):
        raise RuntimeError("PNG renderer unavailable")

    monkeypatch.setattr(
        "zcu_tools.gui.app.measure.remote.handlers.interactive.render_figure_png",
        unavailable_renderer,
    )
    with open_client(fx.service.port) as sock:
        initial = _interact(sock, tab_id, include_figure=False)["result"]
        rejected = _interact(
            sock,
            tab_id,
            {"command": "done", "args": {"unexpected": 1}},
            include_figure=False,
        )
        assert rejected["error"]["code"] == "invalid_params"
        assert _interact(sock, tab_id, include_figure=False)["result"] == initial

        done = _interact(sock, tab_id, {"command": "done"}, include_figure=False)
        assert done["ok"] is True
        result = done["result"]
        assert result == {**initial, "figure": None, "preview_active": False}
        assert result["operation_id"] == token
        settled = _rpc(sock, "operation.await", {"operation_id": token, "timeout": 0.1})
        assert settled["result"]["status"] == "finished"
        summary = _rpc(
            sock, "tab.get_analyze_result", {"tab_id": tab_id, "operation_id": token}
        )
        assert summary["ok"] is True
        assert _interact(sock, tab_id)["error"]["code"] == "precondition_failed"


def test_commands_after_gui_and_context_changes_need_no_seen_baseline(fx) -> None:
    tab_id, token, plugin = _start(fx)
    active = fx.ctrl.run_analyze_control.get_interactive(tab_id)
    assert active is not None
    plugin.execute_command(active.session, "set_conjugate", {"enabled": True})
    fx.state.set_context(
        replace(fx.state.session_env, md=MetaDict(), ml=ModuleLibrary())
    )
    with open_client(fx.service.port) as sock:
        # First request is a mutation, without any hidden snapshot reads.
        changed = _interact(
            sock, tab_id, {"command": "set_conjugate", "args": {"enabled": False}}
        )
        assert changed["ok"] is True
        assert changed["result"]["state"]["conjugate"] is False
        committed = plugin.project_state(active.session.snapshot())
        assert isinstance(committed, dict)
        assert committed["conjugate"] is False
        assert fx.ctrl.run_analyze_control.get_interactive(tab_id) is active
        assert _interact(sock, tab_id, {"command": "done"})["ok"] is True
        settled = _rpc(sock, "operation.await", {"operation_id": token, "timeout": 0.1})
        assert settled["result"]["status"] == "finished"


def test_invalid_payload_preserves_active_session_until_done(fx) -> None:
    tab_id, _token, _plugin = _start(fx)
    with open_client(fx.service.port) as sock:
        baseline = _interact(sock, tab_id)["result"]["state"]
        bad = [
            {"command": "not_registered"},
            {"command": "move_line", "args": {"role": "bad", "position": 1}},
            {"command": "move_line", "args": {"role": "half"}},
            {"command": "move_line", "args": {"role": "half", "position": True}},
            {"command": "swap_lines", "args": {"unexpected": 1}},
            {"command": "done", "args": {"unexpected": 1}},
            {"args": {}},
            {"command": "swap_lines", "args": []},
        ]
        for payload in bad:
            reply = _interact(sock, tab_id, payload)
            assert reply["ok"] is False
            assert reply["error"]["code"] == "invalid_params"
            assert _interact(sock, tab_id)["result"]["state"] == baseline
        assert _interact(sock, "missing")["error"]["code"] == "invalid_params"
        assert _interact(sock, tab_id, {"command": "done"})["ok"] is True
        assert (
            _interact(sock, tab_id, {"command": "swap_lines"})["error"]["code"]
            == "precondition_failed"
        )


def test_equal_line_command_rejects_without_changing_session_or_operation(fx) -> None:
    tab_id, token, _plugin = _start(fx)
    with open_client(fx.service.port) as sock:
        original = _interact(sock, tab_id)["result"]["state"]
        reply = _interact(
            sock,
            tab_id,
            {
                "command": "move_line",
                "args": {"role": "half", "position": original["flux_int"]},
            },
        )
        assert reply["error"]["code"] == "invalid_params"
        assert _interact(sock, tab_id)["result"]["state"] == original
        assert fx.ctrl.run_analyze_control.get_interactive(tab_id) is not None
        done = _interact(sock, tab_id, {"command": "done"})
        assert done["result"]["state"] == original
        assert (
            _rpc(sock, "operation.await", {"operation_id": token, "timeout": 0.1})[
                "result"
            ]["status"]
            == "finished"
        )


def test_live_frontend_preview_remote_commit_and_done_share_result(fx, qapp) -> None:
    tab_id, token, plugin = _start(fx)
    active = fx.ctrl.run_analyze_control.get_interactive(tab_id)
    assert active is not None
    session = active.session
    widget = FluxPickFrontend(
        plugin,
        session,
        fx.ctrl,
        lambda: fx.ctrl.run_analyze_control.finish_interactive(tab_id),
        lambda: fx.ctrl.run_analyze_control.cancel_analyze(tab_id),
    )
    fx.widgets.append(widget)
    widget.show()
    qapp.processEvents()
    canvas = widget.findChild(FigureCanvasQTAgg)
    assert canvas is not None
    canvas.draw()
    fx.view.interactive_presentation.side_effect = lambda _id: (
        widget.figure,
        widget.preview_active,
    )
    fx.view.discard_interactive_preview.side_effect = lambda _id: (
        widget.cancel_preview()
    )
    fx.view.unmount_interactive_analysis.side_effect = lambda _id, **_kw: (
        widget.teardown()
    )

    with open_client(fx.service.port) as sock:
        initial = _interact(sock, tab_id)["result"]["state"]
        _pointer(canvas, "button_press_event", initial["flux_half"])
        _pointer(canvas, "motion_notify_event", initial["flux_half"] + 0.5)
        preview = _interact(sock, tab_id)["result"]
        assert preview["state"] == initial
        assert preview["preview_active"] is True
        assert base64.b64decode(preview["figure"]["png_b64"]).startswith(b"\x89PNG")

        moved = _interact(
            sock,
            tab_id,
            {
                "command": "move_line",
                "args": {"role": "half", "position": initial["flux_half"] + 0.8},
            },
        )["result"]["state"]
        assert widget.preview_active is False
        assert moved["flux_half"] == pytest.approx(initial["flux_half"] + 0.8)
        assert session.snapshot().flux_half == pytest.approx(moved["flux_half"])
        # GUI action uses the same session; the next remote read sees its commit.
        _button(widget, "Swap Lines").click()
        after_gui = _interact(sock, tab_id)["result"]["state"]
        assert after_gui["flux_half"] == moved["flux_int"]
        _pointer(canvas, "button_press_event", after_gui["flux_half"])
        _pointer(canvas, "motion_notify_event", after_gui["flux_half"] + 0.2)
        assert widget.preview_active is True
        done = _interact(sock, tab_id, {"command": "done"})["result"]
        assert done["state"] == after_gui
        assert done["preview_active"] is False
        assert widget.preview_active is False
        assert base64.b64decode(done["figure"]["png_b64"]).startswith(b"\x89PNG")
        result = fx.ctrl.get_tab_analyze_result(tab_id)
        assert result.flx_half == pytest.approx(after_gui["flux_half"])
        assert fx.state.get_tab(tab_id).analysis.plots["pick"] is not widget.figure
        assert (
            _rpc(sock, "operation.await", {"operation_id": token, "timeout": 0.1})[
                "result"
            ]["status"]
            == "finished"
        )


def test_auto_align_busy_failure_and_terminal_delivery_use_same_command(
    fx, qapp
) -> None:
    deferred = DeferredBackground()
    tab_id, token, plugin = _start(fx, background=deferred.run_background)
    active = fx.ctrl.run_analyze_control.get_interactive(tab_id)
    assert active is not None
    session = active.session
    widget = FluxPickFrontend(
        plugin,
        session,
        fx.ctrl,
        lambda: fx.ctrl.run_analyze_control.finish_interactive(tab_id),
        lambda: fx.ctrl.run_analyze_control.cancel_analyze(tab_id),
    )
    fx.widgets.append(widget)
    widget.show()
    qapp.processEvents()
    fx.view.interactive_presentation.side_effect = lambda _id: (
        widget.figure,
        widget.preview_active,
    )
    fx.view.discard_interactive_preview.side_effect = lambda _id: (
        widget.cancel_preview()
    )
    fx.view.unmount_interactive_analysis.side_effect = lambda _id, **_kw: (
        widget.teardown()
    )
    with open_client(fx.service.port) as sock:
        original = _interact(sock, tab_id)["result"]["state"]
        pending = _interact(sock, tab_id, {"command": "auto_align"})
        assert pending["result"]["info"]["alignment_busy"] is True
        assert len(deferred.pending) == 1
        assert not _button(widget, "Auto Align").isEnabled()
        repeated = _interact(sock, tab_id, {"command": "auto_align"})
        assert repeated["error"]["code"] == "precondition_failed"
        _, _, on_error = deferred.pending[0]
        on_error(RuntimeError("alignment failed"))
        assert _interact(sock, tab_id)["result"]["state"] == original
        assert (
            _interact(sock, tab_id)["result"]["info"]["alignment_error"]
            == "alignment failed"
        )
        assert _button(widget, "Auto Align").isEnabled()
        _button(widget, "Auto Align").click()
        assert len(deferred.pending) == 2
        assert (
            _interact(sock, tab_id, {"command": "done"})["result"]["state"] == original
        )
        compute, on_done, _ = deferred.pending[1]
        on_done(compute())
        assert fx.ctrl.get_tab_analyze_result(tab_id).flx_half == original["flux_half"]
        assert (
            _rpc(sock, "operation.await", {"operation_id": token, "timeout": 0.1})[
                "result"
            ]["status"]
            == "finished"
        )


def test_cancel_during_remote_alignment_ignores_late_delivery(fx) -> None:
    deferred = DeferredBackground()
    tab_id, token, _plugin = _start(fx, background=deferred.run_background)
    with open_client(fx.service.port) as sock:
        assert _interact(sock, tab_id)["ok"] is True
        assert _interact(sock, tab_id, {"command": "auto_align"})["ok"] is True
        cancelled = _rpc(sock, "analyze.cancel", {"tab_id": tab_id})
        assert cancelled["result"]["cancelled"] is True
        compute, on_done, _ = deferred.pending[0]
        on_done(compute())
        assert fx.ctrl.get_tab_analyze_result(tab_id) is None
        assert _interact(sock, tab_id)["error"]["code"] == "precondition_failed"
        assert (
            _rpc(sock, "operation.await", {"operation_id": token, "timeout": 0.1})[
                "result"
            ]["status"]
            == "cancelled"
        )


@pytest.mark.parametrize("pane", ["run", "data"])
@pytest.mark.parametrize("terminal", ["done", "cancel"])
def test_controller_interactive_mount_is_visible_from_run_or_data(
    mounted_fx, qapp, pane: Literal["run", "data"], terminal: str
) -> None:
    fx, window = mounted_fx
    tab_id, token, widget = _start_mounted(fx, window, "onetone/flux_dep", pane=pane)
    qapp.processEvents()
    assert widget.isVisibleTo(window)
    with open_client(fx.service.port) as sock:
        if terminal == "done":
            assert _interact(sock, tab_id, {"command": "done"})["ok"] is True
        else:
            assert (
                _rpc(sock, "analyze.cancel", {"tab_id": tab_id})["result"]["cancelled"]
                is True
            )
        assert _rpc(sock, "operation.await", {"operation_id": token, "timeout": 0.1})[
            "result"
        ]["status"] == ("finished" if terminal == "done" else "cancelled")
    assert window.interactive_presentation(tab_id) is None


@pytest.mark.parametrize("terminal", ["done", "cancel"])
def test_mcp_interactive_uses_mounted_plugin_and_original_operation(
    mounted_fx, tmp_path, terminal, request
) -> None:
    fx, window = mounted_fx
    tab_id, token, widget = _start_mounted(fx, window, "onetone/flux_dep")
    active = fx.ctrl.run_analyze_control.get_interactive(tab_id)
    assert active is not None
    bridge, call = mcp_client(fx.service.port, tmp_path, request=request)
    try:
        call("connect", {"port": fx.service.port})
        read = call("tab_interact", {"tab": tab_id})
        assert read["state"] == active.plugin.project_state(active.session.snapshot())
        assert Path(read["figure"]).read_bytes().startswith(b"\x89PNG")
        assert read["preview_active"] is False
        assert "done" in {command["name"] for command in read["commands"]}
        changed = call(
            "tab_interact",
            {
                "tab": tab_id,
                "payload": {"command": "set_conjugate", "args": {"enabled": True}},
            },
        )
        assert changed["state"]["conjugate"] is True
        assert changed["state"] == active.plugin.project_state(
            active.session.snapshot()
        )
        running = call("status", {})["running"]
        assert len(running) == 1
        op = running[0]["op"]
        assert read["handle"] == op
        assert changed["handle"] == op
        if terminal == "done":
            result = call(
                "tab_interact", {"tab": tab_id, "payload": {"command": "done"}}
            )
            assert result["interaction"]["state"] == changed["state"]
            assert result["status"] == "finished", result
            assert Path(result["figure"]).read_bytes().startswith(b"\x89PNG")
            committed = fx.state.get_tab(tab_id).analysis.plots
            assert committed is not None
            assert committed["pick"] is not widget.figure
        else:
            call("cancel", {"op": op})
        status = "finished" if terminal == "done" else "cancelled"
        assert call("wait", {"op": op, "timeout": 0.1})["status"] == status
        with open_client(fx.service.port) as sock:
            assert (
                _rpc(sock, "operation.await", {"operation_id": token, "timeout": 0.1})[
                    "result"
                ]["status"]
                == status
            )
        assert fx.ctrl.run_analyze_control.get_interactive(tab_id) is None
        assert window.interactive_presentation(tab_id) is None
    finally:
        bridge.disconnect()


@dataclass
class TaggedFluxPickParams(FluxPickParams):
    label: str


@pytest.mark.parametrize("terminal", ["done", "cancel"])
def test_interactive_submitted_params_replace_previous_pane_only_on_done(
    mounted_fx, terminal: str
) -> None:
    fx, window = mounted_fx
    previous_params = TaggedFluxPickParams(label="previous request")
    tab_id, previous_token, _ = _start_mounted(
        fx, window, "onetone/flux_dep", analyze_params=previous_params
    )
    with open_client(fx.service.port) as sock:
        assert _interact(sock, tab_id, {"command": "done"})["ok"] is True
        assert (
            _rpc(
                sock,
                "operation.await",
                {"operation_id": previous_token, "timeout": 0.1},
            )["result"]["status"]
            == "finished"
        )
        previous = fx.state.get_tab(tab_id).analysis
        old_params = previous.params
        old_result = previous.result
        old_plots = previous.plots
        assert old_params is previous_params
        assert old_result is not None
        assert old_plots is not None

        submitted = TaggedFluxPickParams(label="new request")
        token = fx.ctrl.run_analyze_control.analyze(tab_id, submitted)
        pending = fx.state.get_tab(tab_id).analysis
        assert pending.params is old_params
        assert pending.result is old_result
        assert pending.plots is old_plots

        changed = _interact(
            sock, tab_id, {"command": "set_conjugate", "args": {"enabled": True}}
        )["result"]["state"]
        assert changed["conjugate"] is True
        assert fx.state.get_tab(tab_id).analysis.params is old_params

        if terminal == "done":
            assert _interact(sock, tab_id, {"command": "done"})["ok"] is True
            committed = fx.state.get_tab(tab_id).analysis
            assert committed.params is submitted
            assert committed.result is not old_result
            assert committed.result.flx_half == pytest.approx(changed["flux_half"])
            assert committed.result.flx_int == pytest.approx(changed["flux_int"])
            assert fx.ctrl.get_tab_analyze_result(tab_id) is committed.result
            assert committed.plots is not old_plots
        else:
            assert (
                _rpc(sock, "analyze.cancel", {"tab_id": tab_id})["result"]["cancelled"]
                is True
            )
            cancelled = fx.state.get_tab(tab_id).analysis
            assert cancelled.params is old_params
            assert cancelled.result is old_result
            assert cancelled.plots is old_plots
        assert _rpc(sock, "operation.await", {"operation_id": token, "timeout": 0.1})[
            "result"
        ]["status"] == ("finished" if terminal == "done" else "cancelled")


def test_mcp_done_writeback_save_and_close_share_the_gui_result(
    mounted_fx, tmp_path, request
) -> None:
    fx, window = mounted_fx
    tab_id, _, widget = _start_mounted(fx, window, "onetone/flux_dep")
    bridge, call = mcp_client(fx.service.port, tmp_path, request=request)
    try:
        call("connect", {"port": fx.service.port})
        done = call("tab_interact", {"tab": tab_id, "payload": {"command": "done"}})
        committed = fx.state.get_tab(tab_id).analysis.plots
        assert committed is not None
        assert committed["pick"] is not widget.figure
        call("tab_get", {"tab": tab_id, "include": ["summary"]})
        preview = call("writeback", {"tab": tab_id})
        expected = {
            "flx_half": done["interaction"]["state"]["flux_half"],
            "flx_int": done["interaction"]["state"]["flux_int"],
            "flx_period": 2
            * abs(
                done["interaction"]["state"]["flux_int"]
                - done["interaction"]["state"]["flux_half"]
            ),
        }
        assert {
            item["target"]: item["proposed"] for item in preview["items"]
        } == expected
        call("rpc_call", {"method": "context.snapshot"})
        written = call(
            "writeback",
            {"tab": tab_id, "write": [{"id": item["id"]} for item in preview["items"]]},
        )
        assert {
            item["target"]: item["after"] for item in written["written"]
        } == expected
        assert call("rpc_call", {"method": "context.snapshot"})["md"] == expected
        call("tab_get", {"tab": tab_id, "include": ["summary", "artifacts"]})
        image = tmp_path / "interactive-result.png"
        saved = call(
            "tab_save",
            {
                "tab": tab_id,
                "artifacts": ["analysis:pick"],
                "paths": {"analysis:pick": str(image)},
            },
        )
        if "op" in saved:
            assert (
                call("wait", {"op": saved["op"], "timeout": 5})["status"] == "finished"
            )
        else:
            assert saved["saved"] == {"analysis:pick": str(image)}
        assert image.read_bytes().startswith(b"\x89PNG")
        artifacts = call("tab_get", {"tab": tab_id, "include": ["artifacts"]})[
            "artifacts"
        ]
        analysis = next(item for item in artifacts if item["key"] == "analysis:pick")
        assert analysis["status"] == "saved"
        assert analysis["last_saved_path"] == str(image)
        assert call("tab_close", {"tab": tab_id, "discard_unsaved": True}) == {
            "closed": tab_id
        }
        assert not fx.ctrl.run_analyze_control.has_tab(tab_id)
    finally:
        bridge.disconnect()


def test_interactive_commands_follow_before_commit_but_reads_do_not(
    mounted_fx, monkeypatch
) -> None:
    fx, window = mounted_fx
    tab_id, token, _ = _start_mounted(fx, window, "onetone/flux_dep")
    active = fx.ctrl.run_analyze_control.get_interactive(tab_id)
    assert active is not None
    follow = window.select_tab_pane
    observed = []

    def select(tab, pane):
        observed.append(
            (tab, pane, active.plugin.project_state(active.session.snapshot()))
        )
        follow(tab, pane)

    monkeypatch.setattr(window, "select_tab_pane", select)
    with open_client(fx.service.port) as sock:
        original = _interact(sock, tab_id)["result"]["state"]
        invalid = _interact(sock, tab_id, {"command": "unknown"})
        assert invalid["ok"] is False
        assert observed == []
        changed = _interact(
            sock,
            tab_id,
            {
                "command": "set_conjugate",
                "args": {"enabled": not original["conjugate"]},
            },
        )
        assert changed["ok"] is True
        assert observed == [(tab_id, "analysis", original)]
        assert changed["result"]["state"]["conjugate"] is not original["conjugate"]
        done = _interact(sock, tab_id, {"command": "done"})
        assert done["ok"] is True
        assert observed[-1] == (tab_id, "analysis", changed["result"]["state"])
        assert len(observed) == 2
        assert (
            _rpc(sock, "operation.await", {"operation_id": token, "timeout": 0.1})[
                "result"
            ]["status"]
            == "finished"
        )


@pytest.mark.parametrize("adapter", ["onetone/flux_dep", "twotone/flux_dep"])
def test_production_flux_adapter_mounts_remote_preview_and_original_writeback(
    mounted_fx, adapter: str, qapp
) -> None:
    fx, window = mounted_fx
    tab_id, token, widget = _start_mounted(fx, window, adapter)
    canvas = widget.findChild(FigureCanvasQTAgg)
    assert canvas is not None
    canvas.draw()
    with open_client(fx.service.port) as sock:
        original = _interact(sock, tab_id)["result"]["state"]
        assert window.interactive_presentation(tab_id) == (widget.figure, False)
        _pointer(canvas, "button_press_event", original["flux_half"])
        _pointer(canvas, "motion_notify_event", original["flux_half"] + 0.4)
        preview = _interact(sock, tab_id)["result"]
        assert preview["state"] == original
        assert preview["preview_active"] is True
        assert base64.b64decode(preview["figure"]["png_b64"]).startswith(b"\x89PNG")
        done = _interact(sock, tab_id, {"command": "done"})
        assert done["ok"] is True
        assert done["result"]["state"] == original
        assert done["result"]["preview_active"] is False
        committed = fx.state.get_tab(tab_id).analysis.plots
        assert committed is not None
        assert committed["pick"] is not widget.figure
        assert window.interactive_presentation(tab_id) is None
        qapp.processEvents()
        tab_widget = window._tab_widgets[tab_id]  # pyright: ignore[reportPrivateUsage] - locate the tested view
        assert (
            tab_widget.get_current_figure_for_pane("analysis")
            is fx.state.get_tab(tab_id).analysis.plots["pick"]
        )
        assert (
            _rpc(sock, "operation.await", {"operation_id": token, "timeout": 0.1})[
                "result"
            ]["status"]
            == "finished"
        )
        draft = _rpc(
            sock, "tab.writeback_preview", {"tab_id": tab_id, "subtab_id": "analysis"}
        )["result"]
        assert draft["has_draft"] is True
        assert {
            item["target_name"]: item["proposed_value"] for item in draft["items"]
        } == {
            "flx_half": pytest.approx(original["flux_half"]),
            "flx_int": pytest.approx(original["flux_int"]),
            "flx_period": pytest.approx(
                2 * abs(original["flux_int"] - original["flux_half"])
            ),
        }
    qapp.processEvents()


def test_mounted_equal_seed_done_failure_preserves_editor_then_recovers(
    mounted_fx,
) -> None:
    fx, window = mounted_fx
    fx.state.session_env.md.flx_half = 0.0
    fx.state.session_env.md.flx_int = 0.0
    tab_id, token, widget = _start_mounted(fx, window, "onetone/flux_dep")
    with open_client(fx.service.port) as sock:
        rejected = _interact(sock, tab_id, {"command": "done"})
        assert rejected["error"]["code"] == "precondition_failed"
        assert window.interactive_presentation(tab_id)[0] is widget.figure
        assert fx.ctrl.get_tab_analyze_result(tab_id) is None
        moved = _interact(
            sock,
            tab_id,
            {"command": "move_line", "args": {"role": "half", "position": 1.0}},
        )
        assert moved["ok"] is True
        assert _interact(sock, tab_id, {"command": "done"})["ok"] is True
        assert (
            _rpc(sock, "operation.await", {"operation_id": token, "timeout": 0.1})[
                "result"
            ]["status"]
            == "finished"
        )


def test_two_mounted_tabs_unmount_one_without_retiring_other(mounted_fx, qapp) -> None:
    fx, window = mounted_fx
    first_id, first_token, first_widget = _start_mounted(fx, window, "onetone/flux_dep")
    second_id, second_token, second_widget = _start_mounted(
        fx, window, "twotone/flux_dep"
    )
    assert window.interactive_presentation(first_id)[0] is first_widget.figure
    assert window.interactive_presentation(second_id)[0] is second_widget.figure
    with open_client(fx.service.port) as sock:
        before = _interact(sock, second_id)["result"]["state"]
        assert _rpc(sock, "analyze.cancel", {"tab_id": first_id})["result"]["cancelled"]
        assert window.interactive_presentation(first_id) is None
        qapp.processEvents()
        assert window.interactive_presentation(second_id)[0] is second_widget.figure
        _button(second_widget, "Swap Lines").click()
        after = _interact(sock, second_id)["result"]["state"]
        assert after["flux_half"] == before["flux_int"]
        moved = _interact(
            sock,
            second_id,
            {
                "command": "move_line",
                "args": {"role": "half", "position": before["flux_int"] + 0.6},
            },
        )["result"]["state"]
        canvas = second_widget.findChild(FigureCanvasQTAgg)
        assert canvas is not None
        assert np.asarray(canvas.figure.axes[0].lines[0].get_xdata(), dtype=float).item(
            0
        ) == pytest.approx(moved["flux_half"])
        assert _interact(sock, second_id, {"command": "done"})["ok"] is True
        for token, status in ((first_token, "cancelled"), (second_token, "finished")):
            assert (
                _rpc(sock, "operation.await", {"operation_id": token, "timeout": 0.1})[
                    "result"
                ]["status"]
                == status
            )
    qapp.processEvents()


def test_two_tabs_keep_independent_remote_sessions(fx) -> None:
    first_id, first_token, _ = _start(fx)
    second_id, second_token, _ = _start(fx)
    with open_client(fx.service.port) as sock:
        first = _interact(sock, first_id)["result"]["state"]
        second = _interact(sock, second_id)["result"]["state"]
        assert (
            _interact(sock, first_id, {"command": "swap_lines"})["result"]["state"][
                "flux_half"
            ]
            == first["flux_int"]
        )
        assert _interact(sock, second_id)["result"]["state"] == second
        assert (
            _rpc(sock, "analyze.cancel", {"tab_id": first_id})["result"]["cancelled"]
            is True
        )
        assert _interact(sock, first_id)["error"]["code"] == "precondition_failed"
        assert (
            _interact(sock, second_id, {"command": "done"})["result"]["state"] == second
        )
        assert (
            _rpc(
                sock, "operation.await", {"operation_id": first_token, "timeout": 0.1}
            )["result"]["status"]
            == "cancelled"
        )
        assert (
            _rpc(
                sock, "operation.await", {"operation_id": second_token, "timeout": 0.1}
            )["result"]["status"]
            == "finished"
        )
