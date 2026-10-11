"""Live GUI-side RPC contracts for a service-owned interactive analysis."""

from __future__ import annotations

import base64
from dataclasses import replace

import pytest

from zcu_lab.v2._support.measure.flux_pick_frontend import FluxPickFrontend

from ._helpers import make_ctx, open_client
from ._interactive_support import (
    DeferredBackground,
    button,
    interact,
    rpc,
    start,
    start_mounted,
)


def test_socket_discovery_commands_done_and_headless_figure(fx) -> None:
    tab_id, token, _plugin = start(fx)
    fx.service.render_view = None
    with open_client(fx.service.port) as sock:
        initial = interact(sock, tab_id)
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
        moved = interact(
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
            interact(
                sock, tab_id, {"command": "set_conjugate", "args": {"enabled": True}}
            )["result"]["state"]["conjugate"]
            is True
        )
        committed = interact(sock, tab_id)["result"]["state"]
        done = interact(sock, tab_id, {"command": "done"})
        assert done["result"]["state"] == committed
        assert done["result"]["operation_id"] == token
        assert fx.ctrl.get_tab_snapshot(tab_id).analysis.source_operation_id == token
        assert base64.b64decode(done["result"]["figure"]["png_b64"]).startswith(
            b"\x89PNG"
        )
        assert fx.ctrl.get_tab_analyze_result(tab_id).flx_half == pytest.approx(
            committed["flux_half"]
        )
        settled = rpc(sock, "operation.await", {"operation_id": token, "timeout": 0.1})
        assert settled["result"]["status"] == "finished"
        assert interact(sock, tab_id)["error"]["code"] == "precondition_failed"


def test_socket_done_receipt_skips_png_and_settles_original_operation(
    fx, monkeypatch
) -> None:
    tab_id, token, _plugin = start(fx)

    def unavailable_renderer(_figure):
        raise RuntimeError("PNG renderer unavailable")

    monkeypatch.setattr(
        "zcu_tools.gui.app.measure.remote.handlers.interactive.render_figure_png",
        unavailable_renderer,
    )
    with open_client(fx.service.port) as sock:
        initial = interact(sock, tab_id, include_figure=False)["result"]
        rejected = interact(
            sock,
            tab_id,
            {"command": "done", "args": {"unexpected": 1}},
            include_figure=False,
        )
        assert rejected["error"]["code"] == "invalid_params"
        assert interact(sock, tab_id, include_figure=False)["result"] == initial

        done = interact(sock, tab_id, {"command": "done"}, include_figure=False)
        assert done["ok"] is True
        result = done["result"]
        assert result == {**initial, "figure": None, "preview_active": False}
        assert result["operation_id"] == token
        settled = rpc(sock, "operation.await", {"operation_id": token, "timeout": 0.1})
        assert settled["result"]["status"] == "finished"
        summary = rpc(
            sock, "tab.get_analyze_result", {"tab_id": tab_id, "operation_id": token}
        )
        assert summary["ok"] is True
        assert interact(sock, tab_id)["error"]["code"] == "precondition_failed"


def test_commands_after_gui_and_context_changes_need_no_seen_baseline(fx) -> None:
    tab_id, token, plugin = start(fx)
    active = fx.ctrl.run_analyze_control.get_interactive(tab_id)
    assert active is not None
    plugin.execute_command(active.session, "set_conjugate", {"enabled": True})
    fx.state.set_context(
        replace(fx.state.session_env, parameters=make_ctx().parameters)
    )
    with open_client(fx.service.port) as sock:
        # First request is a mutation, without any hidden snapshot reads.
        changed = interact(
            sock, tab_id, {"command": "set_conjugate", "args": {"enabled": False}}
        )
        assert changed["ok"] is True
        assert changed["result"]["state"]["conjugate"] is False
        committed = plugin.project_state(active.session.snapshot())
        assert isinstance(committed, dict)
        assert committed["conjugate"] is False
        assert fx.ctrl.run_analyze_control.get_interactive(tab_id) is active
        assert interact(sock, tab_id, {"command": "done"})["ok"] is True
        settled = rpc(sock, "operation.await", {"operation_id": token, "timeout": 0.1})
        assert settled["result"]["status"] == "finished"


def test_invalid_payload_preserves_active_session_until_done(fx) -> None:
    tab_id, _token, _plugin = start(fx)
    with open_client(fx.service.port) as sock:
        baseline = interact(sock, tab_id)["result"]["state"]
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
            reply = interact(sock, tab_id, payload)
            assert reply["ok"] is False
            assert reply["error"]["code"] == "invalid_params"
            assert interact(sock, tab_id)["result"]["state"] == baseline
        assert interact(sock, "missing")["error"]["code"] == "invalid_params"
        assert interact(sock, tab_id, {"command": "done"})["ok"] is True
        assert (
            interact(sock, tab_id, {"command": "swap_lines"})["error"]["code"]
            == "precondition_failed"
        )


def test_equal_line_command_rejects_without_changing_session_or_operation(fx) -> None:
    tab_id, token, _plugin = start(fx)
    with open_client(fx.service.port) as sock:
        original = interact(sock, tab_id)["result"]["state"]
        reply = interact(
            sock,
            tab_id,
            {
                "command": "move_line",
                "args": {"role": "half", "position": original["flux_int"]},
            },
        )
        assert reply["error"]["code"] == "invalid_params"
        assert interact(sock, tab_id)["result"]["state"] == original
        assert fx.ctrl.run_analyze_control.get_interactive(tab_id) is not None
        done = interact(sock, tab_id, {"command": "done"})
        assert done["result"]["state"] == original
        assert (
            rpc(sock, "operation.await", {"operation_id": token, "timeout": 0.1})[
                "result"
            ]["status"]
            == "finished"
        )


def test_auto_align_busy_failure_and_terminal_delivery_use_same_command(
    fx, qapp
) -> None:
    deferred = DeferredBackground()
    tab_id, token, plugin = start(fx, background=deferred.run_background)
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
        original = interact(sock, tab_id)["result"]["state"]
        pending = interact(sock, tab_id, {"command": "auto_align"})
        assert pending["result"]["info"]["alignment_busy"] is True
        assert len(deferred.pending) == 1
        assert not button(widget, "Auto Align").isEnabled()
        repeated = interact(sock, tab_id, {"command": "auto_align"})
        assert repeated["error"]["code"] == "precondition_failed"
        _, _, on_error = deferred.pending[0]
        on_error(RuntimeError("alignment failed"))
        assert interact(sock, tab_id)["result"]["state"] == original
        assert (
            interact(sock, tab_id)["result"]["info"]["alignment_error"]
            == "alignment failed"
        )
        assert button(widget, "Auto Align").isEnabled()
        button(widget, "Auto Align").click()
        assert len(deferred.pending) == 2
        assert (
            interact(sock, tab_id, {"command": "done"})["result"]["state"] == original
        )
        compute, on_done, _ = deferred.pending[1]
        on_done(compute())
        assert fx.ctrl.get_tab_analyze_result(tab_id).flx_half == original["flux_half"]
        assert (
            rpc(sock, "operation.await", {"operation_id": token, "timeout": 0.1})[
                "result"
            ]["status"]
            == "finished"
        )


def test_cancel_during_remote_alignment_ignores_late_delivery(fx) -> None:
    deferred = DeferredBackground()
    tab_id, token, _plugin = start(fx, background=deferred.run_background)
    with open_client(fx.service.port) as sock:
        assert interact(sock, tab_id)["ok"] is True
        assert interact(sock, tab_id, {"command": "auto_align"})["ok"] is True
        cancelled = rpc(sock, "analyze.cancel", {"tab_id": tab_id})
        assert cancelled["result"]["cancelled"] is True
        compute, on_done, _ = deferred.pending[0]
        on_done(compute())
        assert fx.ctrl.get_tab_analyze_result(tab_id) is None
        assert interact(sock, tab_id)["error"]["code"] == "precondition_failed"
        assert (
            rpc(sock, "operation.await", {"operation_id": token, "timeout": 0.1})[
                "result"
            ]["status"]
            == "cancelled"
        )


def test_interactive_commands_follow_before_commit_but_reads_do_not(
    mounted_fx, monkeypatch
) -> None:
    fx, window = mounted_fx
    tab_id, token, _ = start_mounted(fx, window, "onetone/flux_dep")
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
        original = interact(sock, tab_id)["result"]["state"]
        invalid = interact(sock, tab_id, {"command": "unknown"})
        assert invalid["ok"] is False
        assert observed == []
        changed = interact(
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
        done = interact(sock, tab_id, {"command": "done"})
        assert done["ok"] is True
        assert observed[-1] == (tab_id, "analysis", changed["result"]["state"])
        assert len(observed) == 2
        assert (
            rpc(sock, "operation.await", {"operation_id": token, "timeout": 0.1})[
                "result"
            ]["status"]
            == "finished"
        )


def test_two_tabs_keep_independent_remote_sessions(fx) -> None:
    first_id, first_token, _ = start(fx)
    second_id, second_token, _ = start(fx)
    with open_client(fx.service.port) as sock:
        first = interact(sock, first_id)["result"]["state"]
        second = interact(sock, second_id)["result"]["state"]
        assert (
            interact(sock, first_id, {"command": "swap_lines"})["result"]["state"][
                "flux_half"
            ]
            == first["flux_int"]
        )
        assert interact(sock, second_id)["result"]["state"] == second
        assert (
            rpc(sock, "analyze.cancel", {"tab_id": first_id})["result"]["cancelled"]
            is True
        )
        assert interact(sock, first_id)["error"]["code"] == "precondition_failed"
        assert (
            interact(sock, second_id, {"command": "done"})["result"]["state"] == second
        )
        assert (
            rpc(sock, "operation.await", {"operation_id": first_token, "timeout": 0.1})[
                "result"
            ]["status"]
            == "cancelled"
        )
        assert (
            rpc(
                sock, "operation.await", {"operation_id": second_token, "timeout": 0.1}
            )["result"]["status"]
            == "finished"
        )
