"""Form RPC observations and atomic writes on an in-process app connection."""

import typing
from typing import ClassVar

import pytest
from pydantic import BaseModel, Field
from zcu_tools.cfg_model import Range
from zcu_tools.gui.app.measure.services.form_contracts import (
    CfgPreconditionError,
    CfgPreconditionReason,
)
from zcu_tools.gui.form import FormModel
from zcu_tools.gui.form_adapter import Form, FormContext

from tests.gui.app.measure._synthetic import SampleAdapter, SampleCfg, SampleResult

from ._form_rpc import FormRpc, ModuleRpcAdapter, ModuleRpcCfg
from ._helpers import Fixture


class TupleCfg(SampleCfg):
    samples: tuple[complex, ...] = (1 + 2j, 3 - 4j)


class TupleAdapter(SampleAdapter):
    @staticmethod
    def make_form(ctx: FormContext) -> Form[TupleCfg]:
        del ctx
        return Form(TupleCfg, {"sweep": Range.between(1.0, 2.0, expts=3)})


class ExtraInputs(BaseModel):
    scale: float = Field(default=1.0, ge=0)


def retain_cfg(extras: ExtraInputs, cfg: SampleCfg) -> SampleCfg:
    return cfg


class ExtrasAdapter(SampleAdapter):
    @staticmethod
    def make_form(ctx: FormContext) -> Form[SampleCfg]:
        del ctx
        return Form(
            SampleCfg,
            {"sweep": Range.between(1.0, 2.0, expts=3)},
            extras=ExtraInputs,
            lower=retain_cfg,
        )


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
    assert set(observed) == {"input", "values", "errors"}
    assert observed["errors"] == []


def test_describe_cfg_reads_current_local_input_and_errors_as_observer(
    form_rpc,
) -> None:
    fixture, tab, rpc = form_rpc
    owner = fixture.ctrl.cfg_resources.lookup(tab)
    owner.write({("gain",): {"expr": "1 / 0"}})
    before = owner.input.data
    fixture.ctrl.control_identity.reclaim_local()
    fixture.state.set_tab_running(tab, running=True)
    try:
        reply = rpc.call("tab.describe_cfg", {"tab_id": tab, "path": ["gain"]})
        assert reply["ok"], reply
        local = reply["result"]
        assert local["description"]["target"]["expression"]
        assert local["description"]["path"] == ["gain"]
        assert local["description"]["children"] == []
        assert local["input_present"] is True
        assert local["input"] == {"expr": "1 / 0"}
        assert local["value"] is None
        assert local["errors"][0]["path"] == ["gain"]
        assert owner.input.data == before
    finally:
        fixture.state.set_tab_running(tab, running=False)


@pytest.mark.parametrize("path", [["missing"], ["gain", "child"], [1], "gain"])
def test_describe_cfg_rejects_unknown_or_malformed_paths(form_rpc, path) -> None:
    _, tab, rpc = form_rpc
    reply = rpc.call("tab.describe_cfg", {"tab_id": tab, "path": path})
    assert not reply["ok"]
    assert reply["error"]["code"] == "invalid_params", reply


class FailingOptionsExtras(FormModel):
    amount: float = 1.0
    broken: ClassVar[bool] = False

    @classmethod
    def options(cls, _env):
        if cls.broken:
            raise ValueError("Synthetic declaration failed")
        return {}


def retain_failing_cfg(extras: FailingOptionsExtras, cfg: SampleCfg) -> SampleCfg:
    return cfg


class FailingOptionsAdapter(SampleAdapter):
    @staticmethod
    def make_form(ctx: FormContext) -> Form[SampleCfg]:
        del ctx
        return Form(
            SampleCfg,
            {"sweep": Range.between(1.0, 2.0, expts=3)},
            extras=FailingOptionsExtras,
            lower=retain_failing_cfg,
        )


def test_describe_cfg_does_not_classify_author_valueerror_as_bad_path(form_rpc) -> None:
    fixture, _, rpc = form_rpc
    fixture.registry.register("failing-options", FailingOptionsAdapter)
    tab = fixture.ctrl.new_tab("failing-options")
    try:
        FailingOptionsExtras.broken = True
        reply = rpc.call("tab.describe_cfg", {"tab_id": tab, "path": ["extras"]})
        assert not reply["ok"]
        assert reply["error"]["code"] == "controller_error", reply
    finally:
        FailingOptionsExtras.broken = False
        fixture.ctrl.tab_control.close_tab(tab)


class MissingValuesCfg(SampleCfg):
    required_value: float
    nullable_value: float | None = None


class MissingValuesAdapter(SampleAdapter):
    @staticmethod
    def make_form(ctx: FormContext) -> Form[MissingValuesCfg]:
        del ctx
        return Form(MissingValuesCfg, {"sweep": Range.between(1.0, 2.0, expts=3)})


def test_describe_cfg_distinguishes_missing_from_null(form_rpc) -> None:
    fixture, _, rpc = form_rpc
    fixture.registry.register("missing-values", MissingValuesAdapter)
    tab = fixture.ctrl.new_tab("missing-values")
    try:
        fixture.ctrl.cfg_resources.lookup(tab).write({("nullable_value",): None})
        missing = rpc.call(
            "tab.describe_cfg", {"tab_id": tab, "path": ["required_value"]}
        )
        assert missing["ok"], missing
        null = rpc.call("tab.describe_cfg", {"tab_id": tab, "path": ["nullable_value"]})
        assert null["ok"], null
        assert missing["result"]["input_present"] is False
        assert null["result"]["input_present"] is True
        assert missing["result"]["input"] is null["result"]["input"] is None
        assert missing["result"]["errors"][0]["path"] == ["required_value"]
        assert null["result"]["errors"] == []
    finally:
        fixture.ctrl.tab_control.close_tab(tab)


def test_describe_cfg_path_errors_keep_request_path(form_rpc) -> None:
    _, tab, rpc = form_rpc
    path = ["gain", "child"]
    reply = rpc.call("tab.describe_cfg", {"tab_id": tab, "path": path})
    assert not reply["ok"]
    assert reply["error"]["code"] == "invalid_params"
    assert reply["error"]["reason"] == "form_path"
    assert reply["error"]["data"] == {"form_path": path}


def test_describe_cfg_root_is_single_layer_and_readable_through_mcp(
    form_rpc, tmp_path
) -> None:
    from contextlib import closing

    from tests.mcp.measure._support import make_client

    _, tab, rpc = form_rpc

    def respond(method, params):
        reply = rpc.call(method, params)
        assert reply["ok"], reply
        return reply["result"]

    client = make_client(tmp_path, respond)
    with closing(client.context.session):
        assert "tab.describe_cfg" in client.context.gui.catalog
        reply = client.call(
            "rpc_call",
            {"method": "tab.describe_cfg", "params": {"tab_id": tab, "path": []}},
        )
        observed = reply
        assert observed["description"]["target"] is None
        assert "gain" in [
            child["name"] for child in observed["description"]["children"]
        ]
        assert observed["input_present"] is True
        fields = {child["name"]: child for child in observed["description"]["children"]}
        assert fields["gain"]["can_expand"] is False
        assert fields["gain"]["selection_fixed"] is False
        assert fields["sweep"]["can_expand"] is True


@pytest.mark.parametrize("author_failure", [False, True])
def test_describe_cfg_mcp_preserves_path_and_internal_error_boundary(
    form_rpc, tmp_path, author_failure: bool
) -> None:
    from contextlib import closing

    from zcu_tools.gui.remote.errors import RemoteError
    from zcu_tools.mcp.measure.session import GuiRpcError

    from tests.mcp.measure._support import make_client

    fixture, _, rpc = form_rpc
    fixture.registry.register("failing-options", FailingOptionsAdapter)
    tab = fixture.ctrl.new_tab("failing-options")
    path = ["extras"] if author_failure else ["gain", "child"]

    def respond(method, params):
        reply = rpc.call(method, params)
        if not reply["ok"]:
            error = reply["error"]
            raise RemoteError(
                error["code"],
                error["message"],
                reason=error.get("reason"),
                data=error.get("data"),
            )
        return reply["result"]

    try:
        FailingOptionsExtras.broken = author_failure
        client = make_client(tmp_path, respond)
        with closing(client.context.session):
            with pytest.raises(GuiRpcError) as caught:
                client.call(
                    "rpc_call",
                    {
                        "method": "tab.describe_cfg",
                        "params": {"tab_id": tab, "path": path},
                    },
                )
            assert caught.value.code == (
                "controller_error" if author_failure else "invalid_params"
            )
            if not author_failure:
                assert caught.value.reason == "form_path"
                assert caught.value.data == {"form_path": path}
                assert client.call(
                    "rpc_call",
                    {
                        "method": "tab.describe_cfg",
                        "params": {"tab_id": tab, "path": ["gain"]},
                    },
                )["description"]["path"] == ["gain"]
    finally:
        FailingOptionsExtras.broken = False
        fixture.ctrl.tab_control.close_tab(tab)


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
            client.context, summary_parameters=(SummaryParameter("gain", ("gain",)),)
        )
        prepared = session.open_tab("fake", reuse=tab)
        prepared.set(("gain",), 0.75)
        operation = prepared.run()
        fixture.ctrl.close()
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
            if method in ("tab.reset_cfg", "tab.edit_recipe_fields", "tab.run_start")
        ]
        assert effects == [
            ("tab.reset_cfg", {"tab_id": tab}),
            (
                "tab.edit_recipe_fields",
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


@pytest.mark.parametrize("draft", ["partial", "invalid-expression"])
def test_existing_inline_draft_is_a_guarded_noop(
    form_rpc: tuple[Fixture, str, FormRpc], draft: str
) -> None:
    fixture, _, rpc = form_rpc
    fixture.registry.register("module-rpc", ModuleRpcAdapter)
    tab = fixture.ctrl.new_tab("module-rpc")
    try:
        owner = fixture.ctrl.cfg_resources.lookup(tab)
        owner.select(("pulse",), {"kind": "inline", "module": {"type": "pulse"}})
        if draft == "invalid-expression":
            owner.write(
                {
                    ("pulse", "module", "ch"): 0,
                    ("pulse", "module", "nqz"): 1,
                    ("pulse", "module", "freq"): {"expr": "1 / 0"},
                    ("pulse", "module", "gain"): 0.2,
                    ("pulse", "module", "waveform"): {
                        "kind": "inline",
                        "module": {"style": "const", "length": 0.1},
                    },
                }
            )
        before = owner.input.data
        version_key = f"tab:{tab}:cfg"
        version = fixture.state.version.get(version_key)
        notifications: list[str] = []
        owner.subscribe(lambda: notifications.append("changed"))
        reply = rpc.call("tab.use_inline", {"tab_id": tab, "field": ["pulse"]})
        assert reply["ok"], reply
        assert reply["result"]["input"] == before
        assert reply["result"]["errors"]
        assert owner.input.data == before
        assert fixture.state.version.get(version_key) == version
        assert notifications == []
    finally:
        fixture.ctrl.tab_control.close_tab(tab)


class TaggedExtraInputs(BaseModel):
    kind: typing.Literal["inline"] = "inline"


def retain_tagged_cfg(extras: TaggedExtraInputs, cfg: ModuleRpcCfg) -> ModuleRpcCfg:
    return cfg


class TaggedExtrasAdapter(ModuleRpcAdapter):
    @staticmethod
    def make_form(ctx: FormContext) -> Form[ModuleRpcCfg]:
        del ctx
        return Form(
            ModuleRpcCfg,
            {"sweep": Range.between(1.0, 2.0, expts=3)},
            extras=TaggedExtraInputs,
            lower=retain_tagged_cfg,
        )


def test_use_inline_rejects_tagged_nonmodule_input(
    form_rpc: tuple[Fixture, str, FormRpc],
) -> None:
    fixture, _, rpc = form_rpc
    fixture.registry.register("tagged-extras", TaggedExtrasAdapter)
    tab = fixture.ctrl.new_tab("tagged-extras")
    try:
        owner = fixture.ctrl.cfg_resources.lookup(tab)
        before = owner.input.data
        assert before["extras"] == {"kind": "inline"}
        version_key = f"tab:{tab}:cfg"
        version = fixture.state.version.get(version_key)
        notifications: list[str] = []
        owner.subscribe(lambda: notifications.append("changed"))
        reply = rpc.call("tab.use_inline", {"tab_id": tab, "field": ["extras"]})
        assert not reply["ok"], reply
        assert reply["error"]["code"] == "invalid_params"
        assert reply["error"]["reason"] == "form_errors"
        assert "extras" in reply["error"]["message"]
        assert "Inline" in reply["error"]["message"]
        assert owner.input.data == before
        assert fixture.state.version.get(version_key) == version
        assert notifications == []
    finally:
        fixture.ctrl.tab_control.close_tab(tab)


def test_module_rpc_conversion_values_errors_and_readonly_admission(form_rpc) -> None:
    fixture, _, rpc = form_rpc
    fixture.registry.register("module-rpc", ModuleRpcAdapter)
    tab = fixture.ctrl.new_tab("module-rpc")
    try:
        owner = fixture.ctrl.cfg_resources.lookup(tab)
        notifications: list[str] = []
        owner.subscribe(lambda: notifications.append("changed"))
        # Selecting Inline uses ordinary type defaults, not hidden fallback.
        owner.select(("pulse",), {"kind": "inline", "module": {"type": "pulse"}})
        assert owner.input.data["pulse"]["kind"] == "inline"
        assert notifications == ["changed"]
        owner.write(
            {
                ("pulse", "module", "ch"): 0,
                ("pulse", "module", "nqz"): 1,
                ("pulse", "module", "freq"): {"expr": "50 * 2"},
                ("pulse", "module", "gain"): 0.2,
                ("pulse", "module", "waveform"): {
                    "kind": "inline",
                    "module": {"style": "const", "length": 0.1},
                },
                ("gain",): {"expr": "1 / 0"},
            }
        )
        before = owner.input.data
        read = rpc.call("tab.read_module", {"tab_id": tab, "field": ["pulse"]})
        assert read["ok"], read
        assert read["result"]["value"]["freq"] == 100.0
        assert read["result"]["mode"] == "inline"
        assert read["result"]["source"] == "pulse"
        assert owner.input.data == before
        changed = len(notifications)
        assert rpc.call("tab.use_inline", {"tab_id": tab, "field": ["pulse"]})["ok"]
        assert len(notifications) == changed
        fixture.ctrl.control_identity.reclaim_local()
        assert rpc.call("tab.read_module", {"tab_id": tab, "field": ["pulse"]})["ok"]
        denied = rpc.call("tab.use_inline", {"tab_id": tab, "field": ["pulse"]})
        assert not denied["ok"]
        assert denied["error"]["reason"] == "not_holder"
        owner.write({("pulse", "module", "freq"): {"expr": "1 / 0"}})
        invalid = rpc.call("tab.read_module", {"tab_id": tab, "field": ["pulse"]})
        assert not invalid["ok"]
        assert invalid["error"]["reason"] == "not_valid"
        assert len(notifications) == changed + 1
    finally:
        fixture.ctrl.tab_control.close_tab(tab)


@pytest.mark.parametrize(
    "field", [["gain"], ["missing"], ["pulse", "module", "freq"], ["pulse", "", "freq"]]
)
def test_module_rpc_rejects_nonmodule_paths_without_mutation(
    form_rpc, field: list[str]
) -> None:
    fixture, tab, rpc = form_rpc
    owner = fixture.ctrl.cfg_resources.lookup(tab)
    before = owner.input.data
    for method in ("tab.read_module", "tab.use_inline"):
        rejected = rpc.call(method, {"tab_id": tab, "field": field})
        assert not rejected["ok"], rejected
        assert rejected["error"]["code"] == "invalid_params", rejected
    assert owner.input.data == before


def test_use_inline_rejects_disabled_module_without_fallback(form_rpc) -> None:
    fixture, _, rpc = form_rpc
    fixture.registry.register("module-rpc", ModuleRpcAdapter)
    tab = fixture.ctrl.new_tab("module-rpc")
    try:
        owner = fixture.ctrl.cfg_resources.lookup(tab)
        before = owner.input.data
        notifications: list[str] = []
        owner.subscribe(lambda: notifications.append("changed"))
        reply = rpc.call("tab.use_inline", {"tab_id": tab, "field": ["pulse"]})
        assert not reply["ok"], "Disabled module must not silently become blank inline"
        assert reply["error"]["reason"] == "form_errors"
        assert reply["error"]["code"] == "invalid_params"
        assert "pulse" in reply["error"]["message"]
        assert "Inline" in reply["error"]["message"]
        assert owner.input.data == before
        assert notifications == []
    finally:
        fixture.ctrl.tab_control.close_tab(tab)


def test_inline_noop_takes_busy_reentry_and_lifetime_guards(
    form_rpc: tuple[Fixture, str, FormRpc],
) -> None:
    fixture, _, rpc = form_rpc
    fixture.registry.register("module-rpc", ModuleRpcAdapter)
    tab = fixture.ctrl.new_tab("module-rpc")
    owner = fixture.ctrl.cfg_resources.lookup(tab)
    try:
        owner.write(
            {
                ("pulse",): {
                    "kind": "inline",
                    "module": {
                        "type": "pulse",
                        "ch": 0,
                        "nqz": 1,
                        "freq": 100.0,
                        "gain": 0.2,
                        "waveform": {
                            "kind": "inline",
                            "module": {"style": "const", "length": 0.1},
                        },
                    },
                }
            }
        )
        before = owner.input.data
        version_key = f"tab:{tab}:cfg"
        version = fixture.state.version.get(version_key)
        notifications: list[str] = []
        owner.subscribe(lambda: notifications.append("changed"))
        successful = rpc.call("tab.use_inline", {"tab_id": tab, "field": ["pulse"]})
        assert successful["ok"], successful
        assert owner.input.data == before
        assert fixture.state.version.get(version_key) == version
        assert notifications == []

        fixture.state.set_tab_running(tab, running=True)
        try:
            busy = rpc.call("tab.use_inline", {"tab_id": tab, "field": ["pulse"]})
            assert not busy["ok"]
            assert busy["error"]["reason"] == "mutation_blocked"
        finally:
            fixture.state.set_tab_running(tab, running=False)
        reentrant: list[str] = []

        def during_notification() -> None:
            rejected = rpc.call("tab.use_inline", {"tab_id": tab, "field": ["pulse"]})
            assert not rejected["ok"]
            reentrant.append(rejected["error"]["reason"])

        owner.subscribe(during_notification)
        owner.write({("gain",): 0.3})
        assert reentrant == ["reentrant_mutation"]
        assert notifications == ["changed"]
        assert fixture.state.version.get(version_key) == version + 1
        fixture.ctrl.tab_control.close_tab(tab)
        with pytest.raises(CfgPreconditionError) as gone:
            owner.require_mutation_allowed()
        assert gone.value.reason is CfgPreconditionReason.RESOURCE_GONE
    finally:
        if tab in fixture.ctrl.list_tab_ids():
            fixture.ctrl.tab_control.close_tab(tab)


def test_headless_sampling_commands_project_actual_values_and_reset(form_rpc):
    fixture, tab, rpc = form_rpc
    owner = fixture.ctrl.cfg_resources.lookup(tab)
    owner.write({("sweep", "start"): 0, ("sweep", "stop"): 1, ("sweep", "n"): 5})
    described = rpc.call("tab.describe_cfg", {"tab_id": tab, "path": ["sweep"]})
    commands = described["result"]["description"]["target"].get("commands", [])
    command = next(item for item in commands if item["name"] == "to_step")
    assert command["inputs"] == [
        ["sweep", "start"],
        ["sweep", "stop"],
        ["sweep", "n"],
    ]
    reply = rpc.call(
        "tab.run_cfg_command",
        {"tab_id": tab, "path": ["sweep"], "name": command["name"]},
    )
    assert reply["ok"], reply
    owner.write({("sweep", "step"): 0.3})
    before = owner.input.to_json()
    observed = rpc.call("tab.describe_cfg", {"tab_id": tab, "path": ["sweep", "stop"]})
    assert observed["ok"], observed
    assert observed["result"]["input"] == 1
    assert observed["result"]["value"] == pytest.approx(0.9)
    assert owner.input.to_json() == before
    inactive = rpc.call("tab.describe_cfg", {"tab_id": tab, "path": ["sweep", "n"]})
    assert not inactive["ok"]
    assert inactive["error"]["reason"] == "form_path"
    reset = rpc.call("tab.reset_cfg", {"tab_id": tab})
    assert reset["ok"], reset
    assert "n" in reset["result"]["input"]["sweep"]
    assert "step" not in reset["result"]["input"]["sweep"]


def test_headless_coordinate_command_observation_and_active_paths(form_rpc):
    fixture, tab, rpc = form_rpc
    owner = fixture.ctrl.cfg_resources.lookup(tab)
    owner.write({("sweep", "start"): 10, ("sweep", "stop"): 4, ("sweep", "n"): 4})
    described = rpc.call("tab.describe_cfg", {"tab_id": tab, "path": ["sweep"]})
    assert described["ok"], described
    commands = described["result"]["description"]["target"].get("commands", [])
    assert commands, "Sweep row must expose coordinate commands"
    command = commands[0]
    assert command["inputs"] == [["sweep", "start"], ["sweep", "stop"]]
    assert command["target"] == ["sweep"]
    changed = rpc.call(
        "tab.run_cfg_command",
        {"tab_id": tab, "path": ["sweep"], "name": command["name"]},
    )
    assert changed["ok"], changed
    assert changed["result"]["input"]["sweep"] == {
        "kind": "center",
        "center": 7.0,
        "span": -6.0,
        "n": 4,
    }
    assert owner.accept().values["sweep"] == {
        "start": 10.0,
        "stop": 4.0,
        "expts": 4,
        "step": -2.0,
    }
    inactive = rpc.call(
        "tab.run_cfg_command",
        {"tab_id": tab, "path": ["sweep", "start"], "name": command["name"]},
    )
    assert not inactive["ok"]
    assert inactive["error"]["reason"] == "form_path"
    before = owner.input.to_json()
    unknown = rpc.call(
        "tab.run_cfg_command",
        {"tab_id": tab, "path": ["sweep"], "name": "not-a-command"},
    )
    assert not unknown["ok"]
    assert owner.input.to_json() == before
    owner.write({("sweep", "center"): {"expr": "1 / 0"}})
    desc = rpc.call("tab.describe_cfg", {"tab_id": tab, "path": ["sweep"]})
    back = desc["result"]["description"]["target"]["commands"][0]["name"]
    before = owner.input.to_json()
    rejected = rpc.call(
        "tab.run_cfg_command",
        {"tab_id": tab, "path": ["sweep"], "name": back},
    )
    assert not rejected["ok"]
    assert owner.input.to_json() == before
    with pytest.raises(CfgPreconditionError) as invalid:
        owner.accept()
    assert invalid.value.reason is CfgPreconditionReason.NOT_VALID


@pytest.mark.parametrize("guard", ["busy", "holder"])
def test_coordinate_rpc_retains_host_guards(form_rpc, guard):
    fixture, tab, rpc = form_rpc
    owner = fixture.ctrl.cfg_resources.lookup(tab)
    before = owner.input.to_json()
    if guard == "busy":
        fixture.state.set_tab_running(tab, running=True)
    else:
        fixture.ctrl.control_identity.reclaim_local()
    try:
        reply = rpc.call(
            "tab.run_cfg_command",
            {"tab_id": tab, "path": ["sweep"], "name": "to_center"},
        )
        assert not reply["ok"]
        assert reply["error"].get("reason") == (
            "mutation_blocked" if guard == "busy" else "not_holder"
        ), reply
        assert owner.input.to_json() == before
    finally:
        fixture.state.set_tab_running(tab, running=False)
