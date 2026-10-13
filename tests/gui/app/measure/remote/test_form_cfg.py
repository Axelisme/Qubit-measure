"""Form RPC observations and atomic writes on an in-process app connection."""

import json
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from pydantic import BaseModel, Field
from zcu_tools.cfg_model import Range
from zcu_tools.gui.app.measure.adapter import Form
from zcu_tools.gui.app.measure.remote import ControlOptions, RemoteControlAdapter
from zcu_tools.gui.remote.rpc_endpoint import ClientLink
from zcu_tools.gui.remote.wire import Request

from tests.gui.app.measure._synthetic import SampleAdapter, SampleCfg, SampleResult

from ._helpers import Fixture


class TupleCfg(SampleCfg):
    samples: tuple[complex, ...] = (1 + 2j, 3 - 4j)


class TupleAdapter(SampleAdapter):
    form = Form(TupleCfg, {"sweep": Range.between(1.0, 2.0, expts=3)})


class ExtraInputs(BaseModel):
    scale: float = Field(default=1.0, ge=0)


def retain_cfg(extras: ExtraInputs, cfg: SampleCfg) -> SampleCfg:
    return cfg


class ExtrasAdapter(SampleAdapter):
    form = Form(
        SampleCfg,
        {"sweep": Range.between(1.0, 2.0, expts=3)},
        extras=ExtraInputs,
        lower=retain_cfg,
    )


class ImmediateScheduler:
    """Dispatch on the test's current owner thread."""

    def is_owner_thread(self) -> bool:
        return True

    def post(self, callback: Callable[[], None]) -> None:
        callback()

    def call[T](self, callback: Callable[[], T]) -> T:
        return callback()


class FormRpc:
    """Record real dispatch replies without a socket or a rendered window."""

    def __init__(self, remote: RemoteControlAdapter) -> None:
        self.remote = remote
        self.link = ClientLink("form-test", token_required=False)
        remote.on_client_open(self.link)
        self.call("control.connect", {})

    def call(self, method: str, params: dict[str, object]) -> dict[str, Any]:
        self.remote.route(self.link, Request("form-request", method, params))
        return json.loads(self.link.outbound.get(timeout=2.0))


@pytest.fixture
def form_rpc(qapp) -> Iterator[tuple[Fixture, str, FormRpc]]:
    fixture = Fixture()
    tab_id = fixture.ctrl.new_tab("fake")
    remote = RemoteControlAdapter(
        fixture.ctrl,
        ControlOptions(port=0),
        ImmediateScheduler(),
        render_view=fixture.view,
    )
    rpc = FormRpc(remote)
    try:
        yield fixture, tab_id, rpc
    finally:
        fixture.ctrl.quiesce()
        fixture.ctrl.tab_control.close_tab(tab_id)
        remote.on_client_close(rpc.link, on_owner_thread=True)


def test_get_cfg_returns_authored_input_and_ordinary_leaf_values(form_rpc) -> None:
    fixture, tab, rpc = form_rpc
    owner = fixture.ctrl.cfg_resources.lookup(tab)
    owner.write({("gain",): {"expr": "2 / 5"}})
    reply = rpc.call("tab.get_cfg", {"tab_id": tab})
    assert reply["ok"], reply
    observed = reply["result"]
    assert observed["input"]["gain"] == {"expr": "2 / 5"}
    assert {tuple(item["path"]): item["value"] for item in observed["values"]}[
        ("gain",)
    ] == 0.4
    assert observed["schema"]["properties"]["gain"]
    assert observed["errors"] == []


def test_edit_cfg_commits_leaf_batch_and_returns_invalid_input_errors(form_rpc) -> None:
    fixture, tab, rpc = form_rpc
    reply = rpc.call(
        "tab.edit_cfg",
        {
            "tab_id": tab,
            "edits": [
                {"path": ["gain"], "value": {"expr": "2 / 5"}},
                {"path": ["sweep", "start"], "value": 8.0},
            ],
        },
    )
    assert reply["ok"], reply
    assert fixture.ctrl.cfg_resources.lookup(tab).input.data["gain"] == {
        "expr": "2 / 5"
    }
    invalid = rpc.call(
        "tab.edit_cfg",
        {
            "tab_id": tab,
            "edits": [
                {"path": ["gain"], "value": {"expr": "1 / 0"}},
            ],
        },
    )
    assert invalid["ok"], invalid
    assert invalid["result"]["input"]["gain"] == {"expr": "1 / 0"}
    assert invalid["result"]["errors"][0]["path"] == ["gain"]
    assert rpc.call("tab.get_cfg", {"tab_id": tab})["result"] == invalid["result"]


def test_reset_returns_fresh_form_defaults_and_keeps_results(form_rpc) -> None:
    fixture, tab, rpc = form_rpc
    owner = fixture.ctrl.cfg_resources.lookup(tab)
    initial = owner.input.data
    owner.write({("gain",): 0.75})
    reply = rpc.call("tab.reset_cfg", {"tab_id": tab})
    assert reply["ok"], reply
    assert reply["result"]["input"] == initial
    assert reply["result"]["errors"] == []


def test_edit_structural_error_rejects_the_entire_batch(form_rpc) -> None:
    fixture, tab, rpc = form_rpc
    owner = fixture.ctrl.cfg_resources.lookup(tab)
    initial = owner.input.data
    reply = rpc.call(
        "tab.edit_cfg",
        {
            "tab_id": tab,
            "edits": [
                {"path": ["gain"], "value": 0.75},
                {"path": ["missing"], "value": 2},
            ],
        },
    )
    assert reply["ok"] is False
    assert reply["error"]["code"] == "invalid_params", reply
    assert owner.input.data == initial


def test_reset_aliases_fixes_parameter_reads_until_next_reset(form_rpc) -> None:
    from tests.resources.entry.fakes import (
        register_fakes,
        registry_state,
        restore_registry,
    )

    fixture, tab, rpc = form_rpc
    before = registry_state()
    try:
        register_fakes()
        context = fixture.ctrl.context_control
        context.add_component("Q1", "fake/drive/a", {"rate": 10.0})
        context.add_component("Q2", "fake/drive/a", {"rate": 20.0})
        rows = context.get_alias_rows()
        reset = rpc.call("tab.reset_cfg", {"tab_id": tab, "aliases": {"qubit": "Q2"}})
        assert reset["ok"], reset
        context.set_parameter(context.read_context().target, "Q2.rate", 25.0)
        query = {"tab_id": tab, "paths": ["$qubit.rate"], "module_refs": []}
        frozen = rpc.call("tab.parameter_read", query)
        assert frozen["ok"], frozen
        assert frozen["result"] == {"values": {"$qubit.rate": 20.0}, "modules": {}}
        assert rpc.call("tab.get_cfg", {"tab_id": tab})["result"] == reset["result"]
        assert context.get_alias_rows() == rows
        reset = rpc.call("tab.reset_cfg", {"tab_id": tab, "aliases": {"qubit": "Q1"}})
        assert reset["ok"], reset
        assert rpc.call("tab.parameter_read", query)["result"]["values"] == {
            "$qubit.rate": 10.0
        }
    finally:
        restore_registry(before)


def test_form_tuple_complex_values_have_lossless_json_projection(form_rpc) -> None:
    fixture, _, rpc = form_rpc
    fixture.registry.register("tuple", TupleAdapter)
    tab = fixture.ctrl.new_tab("tuple")
    try:
        reply = rpc.call("tab.get_cfg", {"tab_id": tab})
        assert reply["ok"], reply
        observed = reply["result"]
        leaves = {tuple(item["path"]): item["value"] for item in observed["values"]}
        assert leaves[("samples",)] == [
            {"__complex__": [1.0, 2.0]},
            {"__complex__": [3.0, -4.0]},
        ]
        assert observed["input"]["samples"] == leaves[("samples",)]
        assert observed["errors"] == []
    finally:
        fixture.ctrl.tab_control.close_tab(tab)


def test_extras_errors_keep_authored_input_and_other_preview_values(form_rpc) -> None:
    fixture, _, rpc = form_rpc
    fixture.registry.register("extras", ExtrasAdapter)
    tab = fixture.ctrl.new_tab("extras")
    try:
        reply = rpc.call(
            "tab.edit_cfg",
            {"tab_id": tab, "edits": [{"path": ["extras", "scale"], "value": -1.0}]},
        )
        assert reply["ok"], reply
        assert reply["result"]["input"]["extras"]["scale"] == -1.0
        assert reply["result"]["errors"][0]["path"] == ["extras", "scale"]
        leaves = {
            tuple(item["path"]): item["value"] for item in reply["result"]["values"]
        }
        assert leaves[("gain",)] == 0.1
    finally:
        fixture.ctrl.tab_control.close_tab(tab)


def test_recipe_with_real_form_and_adapter_preserves_acquisition_result(
    form_rpc, tmp_path
) -> None:
    from contextlib import closing

    from zcu_tools.experiment.records import RunRecord
    from zcu_tools.mcp.measure.execution_reply import SummaryParameter
    from zcu_tools.mcp.measure.recipe import RecipeSession
    from zcu_tools.mcp.measure.session import GuiRpcError

    from tests.mcp.measure._support import make_client

    fixture, tab, rpc = form_rpc

    def respond(method: str, params: dict[str, object]) -> dict[str, object]:
        reply = rpc.call(method, params)
        if not reply["ok"]:
            error = reply["error"]
            raise GuiRpcError(error["message"], reason=error.get("reason"))
        return reply["result"]

    client = make_client(tmp_path, respond)
    with closing(client.context.session):
        session = RecipeSession(
            client.context, summary_parameters=(SummaryParameter("gain", "gain"),)
        )
        prepared = session.open_tab("fake", reuse=tab)
        prepared.set("gain", 0.75)
        operation = prepared.run()
        assert fixture.ctrl.quiesce()
        completed, status = operation.complete_in_current_worker()
        assert status == "completed"
        if completed is None:
            raise TypeError("Expected completed recipe Run")
        assert completed.snapshot().actual["fields"]["gain"]["value"] == 0.75
        snapshot = fixture.ctrl.tab_control.get_tab_snapshot(tab)
        if snapshot.run is None or not isinstance(snapshot.run.result, RunRecord):
            raise TypeError("Expected a completed RunRecord")
        record = snapshot.run.result
        assert SampleCfg.model_validate(record.cfg).gain == 0.75
        if not isinstance(record.result, SampleResult):
            raise TypeError("Expected synthetic acquisition data")
        assert record.result.data.tolist() == [0.1, 0.2, 0.3]
        effects = [
            (method, params)
            for method, params in client.transport.sent
            if method in ("tab.reset_cfg", "tab.edit_cfg", "tab.run_start")
        ]
        assert effects == [
            ("tab.reset_cfg", {"tab_id": tab}),
            (
                "tab.edit_cfg",
                {"tab_id": tab, "edits": [{"path": ["gain"], "value": 0.75}]},
            ),
            ("tab.run_start", {"tab_id": tab}),
        ]


def test_run_aliases_rejected_before_view_or_operation_effects(form_rpc) -> None:
    fixture, tab, rpc = form_rpc
    selected = []
    fixture.view.select_tab_pane.side_effect = lambda *args: selected.append(args)
    reply = rpc.call("tab.run_start", {"tab_id": tab, "aliases": {"qubit": "Q2"}})
    assert reply["ok"] is False
    assert reply["error"]["reason"] == "run_aliases_not_supported", reply
    assert selected == []
    assert fixture.ctrl.tab_control.get_running_tab_id() is None
