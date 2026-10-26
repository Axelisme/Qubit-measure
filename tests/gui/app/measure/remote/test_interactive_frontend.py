"""Mounted interactive presentation, result writeback and multi-tab lifecycle."""

from __future__ import annotations

import base64
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pytest
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from zcu_tools.gui.app.measure.flux_pick_frontend import FluxPickFrontend
from zcu_tools.gui.app.measure.interactive_flux_pick import FluxPickParams
from zcu_tools.gui.app.measure.remote.save_types import SaveSnapshotWire
from zcu_tools.gui.app.measure.ui.exp_tab_widget import ExpTabWidget
from zcu_tools.gui.app.measure.ui.main_window import MainWindow
from zcu_tools.gui.plotting.figure_export import render_figure_png
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_types import CommandOrigin, PointCreateRequest
from zcu_tools.resources.entry import component_registry

from tests.gui.app.measure._components import register_all as register_components
from tests.gui.app.measure._record_support import require_record
from tests.resources.entry.fakes import registry_state, restore_registry

from ._helpers import mcp_client, open_client
from ._interactive_support import (
    InteractiveFixture,
    button,
    interact,
    pointer,
    rpc,
    start_interactive,
    start_mounted,
)


@pytest.fixture
def mounted_fx(qapp, tmp_path):
    before = registry_state()
    try:
        register_components(component_registry)
        fixture = InteractiveFixture(project_root=str(tmp_path))
        fixture.ctrl.context_control.add_component(
            "R1", "resonator", {"freq": 4500.0, "kappa": 1.0}
        )
        fixture.ctrl.context_control.add_component(
            "Q1", "qubit/fluxonium", {"resonator": "R1"}
        )
        fixture.ctrl.context_control.set_alias_rows(
            (AliasRow("qubit", "Q1"), AliasRow("resonator", "$qubit.resonator"))
        )
        fixture.state.set_context(
            replace(fixture.state.session_env, soc=None, soccfg=None)
        )
        window = MainWindow(fixture.ctrl, templates=lambda: {})
        fixture.ctrl.add_view(window)
        fixture.service.render_view = window
        fixture.start()
        window.show()
        qapp.processEvents()
        yield fixture, window
        fixture.ctrl.close()
        fixture.stop()
        window.deleteLater()
        qapp.processEvents()
    finally:
        restore_registry(before)


def test_live_frontend_preview_remote_commit_and_done_share_result(fx, qapp) -> None:
    tab_id, token, plugin = start_interactive(fx)
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
        initial = interact(sock, tab_id)["result"]["state"]
        pointer(canvas, "button_press_event", initial["flux_half"])
        pointer(canvas, "motion_notify_event", initial["flux_half"] + 0.5)
        preview = interact(sock, tab_id)["result"]
        assert preview["state"] == initial
        assert preview["preview_active"] is True
        assert base64.b64decode(preview["figure"]["png_b64"]).startswith(b"\x89PNG")

        moved = interact(
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
        button(widget, "Swap Lines").click()
        after_gui = interact(sock, tab_id)["result"]["state"]
        assert after_gui["flux_half"] == moved["flux_int"]
        pointer(canvas, "button_press_event", after_gui["flux_half"])
        pointer(canvas, "motion_notify_event", after_gui["flux_half"] + 0.2)
        assert widget.preview_active is True
        done = interact(sock, tab_id, {"command": "done"})["result"]
        assert done["state"] == after_gui
        assert done["preview_active"] is False
        assert widget.preview_active is False
        assert base64.b64decode(done["figure"]["png_b64"]).startswith(b"\x89PNG")
        result = fx.ctrl.run_analyze_control.get_tab_analyze_result(tab_id)
        assert result.flx_half == pytest.approx(after_gui["flux_half"])
        assert fx.state.get_tab(tab_id).analysis.plots["pick"] is not widget.figure
        assert (
            rpc(sock, "operation.await", {"operation_id": token, "timeout": 0.1})[
                "result"
            ]["status"]
            == "finished"
        )


@pytest.mark.parametrize("pane", ["run", "data"])
@pytest.mark.parametrize("terminal", ["done", "cancel"])
def test_controller_interactive_mount_is_visible_from_run_or_data(
    mounted_fx, qapp, pane: Literal["run", "data"], terminal: str
) -> None:
    fx, window = mounted_fx
    tab_id, token, widget = start_mounted(fx, window, "synthetic/magnitude", pane=pane)
    qapp.processEvents()
    assert widget.isVisibleTo(window)
    with open_client(fx.service.port) as sock:
        if terminal == "done":
            assert interact(sock, tab_id, {"command": "done"})["ok"] is True
        else:
            assert rpc(sock, "operation.cancel", {"operation_id": token})["result"][
                "status"
            ] in ("cancelling", "cancelled")
        assert rpc(sock, "operation.await", {"operation_id": token, "timeout": 0.1})[
            "result"
        ]["status"] == ("finished" if terminal == "done" else "cancelled")
    assert window.interactive_presentation(tab_id) is None


def _assert_mcp_interactive_discovery(
    call: Callable[[str, dict[str, Any]], dict[str, Any]],
    tab_id: str,
) -> dict[str, Any]:
    read = call("tab_interact", {"tab": tab_id})
    # The caller keeps structured data; MCP stdio contracts own image delivery.
    # Read the GUI PNG through its public RPC instead of a session file.
    figure = call("rpc_call", {"method": "tab.interact", "params": {"tab_id": tab_id}})[
        "figure"
    ]
    assert base64.b64decode(figure["png_b64"], validate=True).startswith(b"\x89PNG")
    assert read["preview_active"] is False
    assert "done" in {command["name"] for command in read["commands"]}
    return read


def _wait_for_mcp_interactive_completion(
    call: Callable[[str, dict[str, Any]], dict[str, Any]],
    receipt: dict[str, Any],
    committed_state: dict[str, Any],
) -> dict[str, Any]:
    assert receipt["interaction"]["state"] == committed_state
    # Done's bounded first wait may end before downstream capture/save.
    # Join that same completion owner, not merely its GUI operation.
    completion = call("wait", {"execution": receipt["execution"]})
    assert completion["execution"] == receipt["execution"]
    assert completion["interaction"] == receipt["interaction"]
    return completion


@pytest.mark.parametrize("terminal", ["done", "cancel"])
def test_mcp_interactive_uses_mounted_plugin_and_original_operation(
    mounted_fx, tmp_path, terminal, request
) -> None:
    fx, window = mounted_fx
    tab_id, token, widget = start_mounted(
        fx,
        window,
        "synthetic/magnitude",
        native_source=tmp_path / "source.h5" if terminal == "done" else None,
    )
    source = require_record(fx.state.get_tab(tab_id).run).metadata
    captured = fx.state.get_tab(tab_id).run.parameters
    active = fx.ctrl.run_analyze_control.get_interactive(tab_id)
    assert active is not None
    bridge, call = mcp_client(fx.service.port, tmp_path, request=request)
    try:
        call("connect", {"port": fx.service.port})
        read = _assert_mcp_interactive_discovery(call, tab_id)
        assert read["state"] == active.plugin.project_state(active.session.snapshot())
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
            assert source is not None and source.snapshot.point is not None
            assert captured is not None
            storage = captured.capture_storage()
            fx.ctrl.context_control.new_point(
                PointCreateRequest(
                    "later", "", CommandOrigin("gui"), clone_from=source.snapshot.point
                )
            )
            result = call(
                "tab_interact", {"tab": tab_id, "payload": {"command": "done"}}
            )
            result = _wait_for_mcp_interactive_completion(
                call, result, changed["state"]
            )
            assert result["status"] == "finished", result
            assert result["run_id"] == source.run_id
            assert result["op"] == op
            analysis = SaveSnapshotWire.model_validate(
                {"run_id": result["run_id"], "artifacts": result["artifacts"]}
            ).artifacts["analysis"]["results"]
            assert analysis.status == "Saved"
            png_outputs = analysis.members["figure:pick"]
            assert len(png_outputs) == 2
            assert {Path(output.path).parent for output in png_outputs} == {
                storage.database_path
                / "runs"
                / f"synthetic.magnitude_Q1-R1_{source.run_id}"
                / "figures",
                storage.result_path / "points" / source.snapshot.point / "figures",
            }
            for output in png_outputs:
                assert output.status == "saved" and output.error is None
                assert Path(output.path).read_bytes().startswith(b"\x89PNG")
            committed = fx.state.get_tab(tab_id).analysis.plots
            assert committed is not None
            assert committed["pick"] is not widget.figure
        else:
            call("cancel", {"op": op})
        status = "finished" if terminal == "done" else "cancelled"
        assert call("wait", {"op": op, "timeout": 0.1})["status"] == status
        with open_client(fx.service.port) as sock:
            assert (
                rpc(sock, "operation.await", {"operation_id": token, "timeout": 0.1})[
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
    tab_id, previous_token, _ = start_mounted(
        fx, window, "synthetic/magnitude", analyze_params=previous_params
    )
    with open_client(fx.service.port) as sock:
        assert interact(sock, tab_id, {"command": "done"})["ok"] is True
        assert (
            rpc(
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

        changed = interact(
            sock, tab_id, {"command": "set_conjugate", "args": {"enabled": True}}
        )["result"]["state"]
        assert changed["conjugate"] is True
        assert fx.state.get_tab(tab_id).analysis.params is old_params

        if terminal == "done":
            assert interact(sock, tab_id, {"command": "done"})["ok"] is True
            committed = fx.state.get_tab(tab_id).analysis
            assert committed.params is submitted
            assert committed.result is not old_result
            assert committed.result.flx_half == pytest.approx(changed["flux_half"])
            assert committed.result.flx_int == pytest.approx(changed["flux_int"])
            assert (
                fx.ctrl.run_analyze_control.get_tab_analyze_result(tab_id)
                is committed.result
            )
            assert committed.plots is not old_plots
        else:
            assert rpc(sock, "operation.cancel", {"operation_id": token})["result"][
                "status"
            ] in ("cancelling", "cancelled")
            cancelled = fx.state.get_tab(tab_id).analysis
            assert cancelled.params is old_params
            assert cancelled.result is old_result
            assert cancelled.plots is old_plots
        assert rpc(sock, "operation.await", {"operation_id": token, "timeout": 0.1})[
            "result"
        ]["status"] == ("finished" if terminal == "done" else "cancelled")


def test_mcp_done_writeback_save_and_close_share_the_gui_result(
    mounted_fx, tmp_path, request
) -> None:
    fx, window = mounted_fx
    tab_id, _, widget = start_mounted(
        fx, window, "synthetic/magnitude", native_source=tmp_path / "source.h5"
    )
    source = require_record(fx.state.get_tab(tab_id).run).metadata
    captured = fx.state.get_tab(tab_id).run.parameters
    assert source is not None and source.snapshot.point is not None
    assert captured is not None
    storage = captured.capture_storage()
    bridge, call = mcp_client(fx.service.port, tmp_path, request=request)
    try:
        call("connect", {"port": fx.service.port})
        done = call("tab_interact", {"tab": tab_id, "payload": {"command": "done"}})
        committed = fx.state.get_tab(tab_id).analysis.plots
        assert committed is not None
        assert committed["pick"] is not widget.figure
        call("rpc_call", {"method": "tab.snapshot", "params": {"tab_id": tab_id}})[
            "tabs"
        ][0]
        preview = call(
            "rpc_call",
            {
                "method": "tab.writeback_preview",
                "params": {"tab_id": tab_id, "subtab_id": "analysis"},
            },
        )
        expected = {
            "$qubit.flux_half": done["interaction"]["state"]["flux_half"],
            "$qubit.flux_int": done["interaction"]["state"]["flux_int"],
            "$qubit.flux_period": 2
            * abs(
                done["interaction"]["state"]["flux_int"]
                - done["interaction"]["state"]["flux_half"]
            ),
        }
        assert {
            item["target_name"]: item["proposed"] for item in preview["items"]
        } == expected
        call("rpc_call", {"method": "context.snapshot"})
        written = call(
            "rpc_call",
            {
                "method": "tab.writeback_write",
                "params": {
                    "write": [{"id": item["id"]} for item in preview["items"]],
                    "tab_id": tab_id,
                    "subtab_id": "analysis",
                },
            },
        )
        actual_expected = {
            "Q1.flux_half": expected["$qubit.flux_half"],
            "Q1.flux_int": expected["$qubit.flux_int"],
            "Q1.flux_period": expected["$qubit.flux_period"],
        }
        assert {
            item["target"]: item["after"] for item in written["written"]
        } == actual_expected
        component = call("rpc_call", {"method": "context.snapshot"})["point"][
            "components"
        ]["Q1"]
        assert {
            "$qubit.flux_half": component["flux_half"],
            "$qubit.flux_int": component["flux_int"],
            "$qubit.flux_period": component["flux_period"],
        } == expected
        call("rpc_call", {"method": "tab.snapshot", "params": {"tab_id": tab_id}})[
            "tabs"
        ][0]
        fx.ctrl.context_control.new_point(
            PointCreateRequest(
                "later", "", CommandOrigin("gui"), clone_from=source.snapshot.point
            )
        )
        saved = SaveSnapshotWire.model_validate(
            call(
                "rpc_call",
                {
                    "method": "tab.save_artifacts",
                    "params": {
                        "artifacts": [{"section": "analysis", "name": "results"}],
                        "tab_id": tab_id,
                    },
                },
            )
        )
        assert saved.run_id == source.run_id
        assert set(saved.artifacts) == {"analysis"}
        assert set(saved.artifacts["analysis"]) == {"results"}
        analysis = saved.artifacts["analysis"]["results"]
        assert analysis.status == "Saved"
        png_outputs = analysis.members["figure:pick"]
        assert len(png_outputs) == 2
        assert {Path(output.path).parent for output in png_outputs} == {
            storage.database_path
            / "runs"
            / f"synthetic.magnitude_Q1-R1_{source.run_id}"
            / "figures",
            storage.result_path / "points" / source.snapshot.point / "figures",
        }
        for output in png_outputs:
            assert output.format == "png"
            assert output.status == "saved" and output.error is None
            assert Path(output.path).read_bytes().startswith(b"\x89PNG")
        json_outputs = [
            output
            for outputs in analysis.members.values()
            for output in outputs
            if output.format == "json"
        ]
        assert len(json_outputs) == 1
        assert json_outputs[0].status == "saved" and json_outputs[0].error is None
        assert Path(json_outputs[0].path).is_file()
        artifacts = call(
            "rpc_call", {"method": "tab.snapshot", "params": {"tab_id": tab_id}}
        )["tabs"][0]["artifacts"]
        assert artifacts["analysis"]["results"] == analysis.model_dump(mode="json")
        assert call("tab_close", {"tab": tab_id, "discard_unsaved": True}) == {
            "closed": tab_id
        }
        assert not fx.ctrl.run_analyze_control.has_tab(tab_id)
    finally:
        bridge.disconnect()


@pytest.mark.parametrize("adapter", ["synthetic/magnitude", "synthetic/phase"])
def test_registered_flux_adapter_mounts_remote_preview_and_original_writeback(
    mounted_fx, adapter: str, qapp
) -> None:
    fx, window = mounted_fx
    tab_id, token, widget = start_mounted(fx, window, adapter)
    canvas = widget.findChild(FigureCanvasQTAgg)
    assert canvas is not None
    canvas.draw()
    with open_client(fx.service.port) as sock:
        original = interact(sock, tab_id)["result"]["state"]
        assert window.interactive_presentation(tab_id) == (widget.figure, False)
        assert rpc(sock, "control.release", {})["ok"] is True
        pointer(canvas, "button_press_event", original["flux_half"])
        pointer(canvas, "motion_notify_event", original["flux_half"] + 0.4)
        presentation = window.interactive_presentation(tab_id)
        assert presentation is not None
        figure, preview_active = presentation
        assert figure is widget.figure
        assert preview_active is True
        assert render_figure_png(figure).startswith(b"\x89PNG")
        assert rpc(sock, "control.acquire", {})["ok"] is True
        preview = interact(sock, tab_id)["result"]
        assert preview["state"] == original
        assert preview["preview_active"] is False
        done = interact(sock, tab_id, {"command": "done"})
        assert done["ok"] is True
        assert done["result"]["state"] == original
        assert done["result"]["preview_active"] is False
        committed = fx.state.get_tab(tab_id).analysis.plots
        assert committed is not None
        assert committed["pick"] is not widget.figure
        assert window.interactive_presentation(tab_id) is None
        qapp.processEvents()
        tab_widget = next(
            tab for tab in window.findChildren(ExpTabWidget) if tab.tab_id == tab_id
        )
        assert (
            tab_widget.get_current_figure_for_pane("analysis")
            is fx.state.get_tab(tab_id).analysis.plots["pick"]
        )
        assert (
            rpc(sock, "operation.await", {"operation_id": token, "timeout": 0.1})[
                "result"
            ]["status"]
            == "finished"
        )
        draft = rpc(
            sock, "tab.writeback_preview", {"tab_id": tab_id, "subtab_id": "analysis"}
        )["result"]
        assert draft["has_draft"] is True
        assert {
            item["target_name"]: item["proposed_value"] for item in draft["items"]
        } == {
            "$qubit.flux_half": pytest.approx(original["flux_half"]),
            "$qubit.flux_int": pytest.approx(original["flux_int"]),
            "$qubit.flux_period": pytest.approx(
                2 * abs(original["flux_int"] - original["flux_half"])
            ),
        }
        assert {item["actual_path"]: item["proposed"] for item in draft["items"]} == {
            "Q1.flux_half": pytest.approx(original["flux_half"]),
            "Q1.flux_int": pytest.approx(original["flux_int"]),
            "Q1.flux_period": pytest.approx(
                2 * abs(original["flux_int"] - original["flux_half"])
            ),
        }
    qapp.processEvents()


def test_mounted_equal_seed_done_failure_preserves_editor_then_recovers(
    mounted_fx,
) -> None:
    fx, window = mounted_fx
    parameters = fx.ctrl.context_control.read_context()
    fx.ctrl.context_control.set_parameter(parameters.target, "Q1.flux_half", 0.0)
    fx.ctrl.context_control.set_parameter(parameters.target, "Q1.flux_int", 0.0)
    tab_id, token, widget = start_mounted(fx, window, "synthetic/magnitude")
    with open_client(fx.service.port) as sock:
        rejected = interact(sock, tab_id, {"command": "done"})
        assert rejected["error"]["code"] == "precondition_failed"
        assert window.interactive_presentation(tab_id)[0] is widget.figure
        assert fx.ctrl.run_analyze_control.get_tab_analyze_result(tab_id) is None
        moved = interact(
            sock,
            tab_id,
            {"command": "move_line", "args": {"role": "half", "position": 1.0}},
        )
        assert moved["ok"] is True
        assert interact(sock, tab_id, {"command": "done"})["ok"] is True
        assert (
            rpc(sock, "operation.await", {"operation_id": token, "timeout": 0.1})[
                "result"
            ]["status"]
            == "finished"
        )


def test_two_mounted_tabs_unmount_one_without_retiring_other(mounted_fx, qapp) -> None:
    fx, window = mounted_fx
    first_id, first_token, first_widget = start_mounted(
        fx, window, "synthetic/magnitude"
    )
    second_id, second_token, second_widget = start_mounted(
        fx, window, "synthetic/phase"
    )
    assert window.interactive_presentation(first_id)[0] is first_widget.figure
    assert window.interactive_presentation(second_id)[0] is second_widget.figure
    with open_client(fx.service.port) as sock:
        before = interact(sock, second_id)["result"]["state"]
        assert rpc(sock, "operation.cancel", {"operation_id": first_token})["result"][
            "status"
        ] in ("cancelling", "cancelled")
        assert window.interactive_presentation(first_id) is None
        qapp.processEvents()
        assert window.interactive_presentation(second_id)[0] is second_widget.figure
        assert rpc(sock, "control.release", {})["ok"] is True
        button(second_widget, "Swap Lines").click()
        assert rpc(sock, "control.acquire", {})["ok"] is True
        after = interact(sock, second_id)["result"]["state"]
        assert after["flux_half"] == before["flux_int"]
        moved = interact(
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
        assert interact(sock, second_id, {"command": "done"})["ok"] is True
        for token, status in ((first_token, "cancelled"), (second_token, "finished")):
            assert (
                rpc(sock, "operation.await", {"operation_id": token, "timeout": 0.1})[
                    "result"
                ]["status"]
                == status
            )
    qapp.processEvents()


def test_interactive_commands_follow_before_commit_but_reads_do_not(
    mounted_fx, monkeypatch
) -> None:
    fx, window = mounted_fx
    tab_id, token, _ = start_mounted(fx, window, "synthetic/magnitude")
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
