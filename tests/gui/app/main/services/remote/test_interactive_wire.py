"""Live GUI-side RPC contracts for a service-owned interactive analysis."""

from __future__ import annotations

import base64
import socket
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import numpy as np
import pytest
from matplotlib.backend_bases import MouseButton, MouseEvent
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from qtpy.QtWidgets import QPushButton
from zcu_tools.experiment.v2_gui.adapters._support.flux_pick_frontend import (
    FluxPickFrontend,
)
from zcu_tools.experiment.v2_gui.adapters._support.flux_pick_plugin import (
    make_flux_pick_plugin,
)
from zcu_tools.gui.app.main.adapter import AnalyzeRequest
from zcu_tools.gui.app.main.services.guard import AnalyzePermit
from zcu_tools.gui.session.adapters.qt_owner_scheduler import QtOwnerScheduler
from zcu_tools.meta_tool import MetaDict, ModuleLibrary

from ._helpers import Fixture, open_client, recv_response, send


class DeferredBackground:
    def __init__(self) -> None:
        self.pending = []

    def run_background(self, compute, on_done, on_error) -> None:
        self.pending.append((compute, on_done, on_error))


class InteractiveFixture(Fixture):
    def __init__(self) -> None:
        super().__init__()
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


def _start(fx, *, background=None):
    tab_id = fx.ctrl.new_tab("fake")
    devs = np.linspace(-5.0, 5.0, 60)
    freqs = np.linspace(4.0, 5.0, 30)
    signals = np.exp(-(devs[:, None] ** 2)) * np.ones((1, 30))
    plugin = make_flux_pick_plugin(
        AnalyzeRequest(
            run_result=SimpleNamespace(signals=signals, values=devs, freqs=freqs),
            analyze_params=object(),
            md=MetaDict(),
            ml=ModuleLibrary(),
            predictor=None,
        ),
        force_magnitude=True,
    )
    plugin.bind_background(background or fx.ctrl.run_background)
    # Set up a real AnalyzeService operation; RPC interactions below always go
    # through the shipped socket and RunAnalyzeControlFacet, not a handler stub.
    token = fx.ctrl._analyze_svc.start_plugin(
        AnalyzePermit(tab_id=tab_id), plugin, QtOwnerScheduler()
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
        assert done["result"]["figure"] is None
        assert fx.ctrl.get_tab_analyze_result(tab_id).flx_half == pytest.approx(
            committed["flux_half"]
        )
        settled = _rpc(sock, "operation.await", {"operation_id": token, "timeout": 0.1})
        assert settled["result"]["status"] == "finished"
        assert _interact(sock, tab_id)["error"]["code"] == "precondition_failed"


def test_invalid_payload_and_stale_guard_leave_one_active_session(fx) -> None:
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
        stale = _interact(
            sock, tab_id, {"command": "swap_lines"}, expected_versions={"context": -1}
        )
        assert stale["error"]["code"] == "precondition_failed"
        assert stale["error"]["reason"] == "stale_version"
        assert _interact(sock, tab_id)["result"]["state"] == baseline
        assert _interact(sock, "missing")["error"]["code"] == "invalid_params"
        assert _interact(sock, tab_id, {"command": "done"})["ok"] is True
        assert (
            _interact(sock, tab_id, {"command": "swap_lines"})["error"]["code"]
            == "precondition_failed"
        )


def test_live_frontend_preview_remote_commit_and_done_share_result(fx, qapp) -> None:
    tab_id, token, plugin = _start(fx)
    session = fx.ctrl.run_analyze_control.get_interactive(tab_id).session
    widget = FluxPickFrontend(
        plugin,
        session,
        fx.ctrl,
        lambda figure: fx.ctrl.run_analyze_control.finish_interactive(tab_id, figure),
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
        assert result.figure is widget.figure
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
    session = fx.ctrl.run_analyze_control.get_interactive(tab_id).session
    widget = FluxPickFrontend(
        plugin,
        session,
        fx.ctrl,
        lambda figure: fx.ctrl.run_analyze_control.finish_interactive(tab_id, figure),
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
