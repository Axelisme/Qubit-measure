"""Indexed module and fixed-element aggregate contracts through Form RPC."""

from typing import Annotated

import pytest
from zcu_tools.cfg_model import ConfigBase, Expr, FieldMeta, Range
from zcu_tools.gui.form_adapter import FixedElements, Form, FormContext

from tests.gui.app.measure._synthetic import SampleAdapter, SampleCfg


class DiagnosticTone(ConfigBase):
    name: str
    frequency: float
    gain: float


class DiagnosticCfg(SampleCfg):
    tones: Annotated[list[DiagnosticTone], FieldMeta(element_label="name")]
    participants: list[int]


class DiagnosticAdapter(SampleAdapter):
    @staticmethod
    def make_form(ctx: FormContext) -> Form[DiagnosticCfg]:
        del ctx
        return Form(
            DiagnosticCfg,
            {
                "sweep": Range.between(1.0, 2.0, expts=3),
                "tones": [
                    {
                        "name": f"Q{i + 1}",
                        "frequency": Expr(f"Q{i + 1}.freq"),
                        "gain": 0.1,
                    }
                    for i in range(6)
                ],
                "participants": [0],
            },
            fixed_elements=(FixedElements(("tones",), "name", ("participants",)),),
        )


@pytest.mark.parametrize("method", ["tab.get_cfg", "tab.describe_cfg"])
def test_cfg_diagnostics_include_every_original_tone_and_seed_without_mutation(
    form_rpc, method: str
) -> None:
    fixture, _, rpc = form_rpc
    fixture.registry.register("diagnostics", DiagnosticAdapter)
    tab = fixture.ctrl.new_tab("diagnostics")
    owner = fixture.ctrl.cfg_resources.lookup(tab)
    before = owner.input.to_json()
    try:
        params: dict[str, object] = {"tab_id": tab}
        if method == "tab.describe_cfg":
            params["path"] = ["tones", 0, "frequency"]
        reply = rpc.call(method, params)
        assert reply["ok"], reply
        assert "diagnostics" in reply["result"], (
            "Cfg reads must include all-tone diagnostics"
        )
        diagnostics = {
            tuple(row["path"]): row for row in reply["result"]["diagnostics"]
        }
        for i in range(6):
            row = diagnostics[("tones", i, "frequency")]
            assert row == {
                "path": ["tones", i, "frequency"],
                "message": "Field required",
                "name": f"Q{i + 1}",
                "index": i,
                "seed_source": f"Q{i + 1}.freq",
                "seed_status": "missing",
                "uncalibrated": None,
            }
        assert diagnostics[("tones", 5, "gain")]["seed_status"] == "literal"
        assert diagnostics[("tones", 5, "gain")]["uncalibrated"] is None
        assert owner.input.to_json() == before
        owner.write({("tones", 5, "frequency"): 6000.0})
        updated = rpc.call(method, params)["result"]["diagnostics"]
        last = next(row for row in updated if row["path"] == ["tones", 5, "frequency"])
        assert last["message"] is None
        assert last["seed_source"] == "Q6.freq"
        assert last["seed_status"] == "missing", (
            "Editing does not rewrite seed provenance"
        )
    finally:
        fixture.ctrl.tab_control.close_tab(tab)


def test_seed_metadata_keeps_position_sources_and_only_declared_calibration(
    form_rpc,
) -> None:
    fixture, existing, rpc = form_rpc
    initial = rpc.call("tab.get_cfg", {"tab_id": existing})
    assert "diagnostics" in initial["result"], "Cfg reads must expose seed diagnostics"

    class SeedMetadataAdapter(SampleAdapter):
        @staticmethod
        def make_form(ctx: FormContext) -> Form[DiagnosticCfg]:
            del ctx
            return Form(
                DiagnosticCfg,
                {
                    "sweep": Range.between(1.0, 2.0, expts=3),
                    "tones": [
                        {"name": "A", "frequency": Expr("6000 + 1"), "gain": 0.1},
                        {"name": "B", "frequency": 6100.0, "gain": 0.2},
                        {"name": "C", "frequency": Expr("missing.freq"), "gain": 0.3},
                    ],
                    "participants": [1],
                },
                fixed_elements=(FixedElements(("tones",), "name", ("participants",)),),
            ).with_seed_metadata(
                {
                    ("tones", 0, "gain"): FieldMeta(
                        description="Synthetic amplitude candidate",
                        calibration="uncalibrated",
                    ),
                    ("tones", 1, "gain"): FieldMeta(calibration="measured"),
                }
            )

    fixture.registry.register("seed-metadata", SeedMetadataAdapter)
    tab = fixture.ctrl.new_tab("seed-metadata")
    try:
        observed = rpc.call("tab.get_cfg", {"tab_id": tab})["result"]
        rows = {tuple(row["path"]): row for row in observed["diagnostics"]}
        assert rows[("tones", 0, "frequency")]["seed_source"] == "6000 + 1"
        assert rows[("tones", 0, "frequency")]["seed_status"] == "available"
        assert rows[("tones", 0, "gain")]["uncalibrated"] is True
        assert (
            rows[("tones", 0, "gain")]["seed_source"] == "Synthetic amplitude candidate"
        )
        assert rows[("tones", 1, "gain")]["uncalibrated"] is False
        assert rows[("tones", 2, "frequency")]["seed_status"] == "missing"
        assert rows[("tones", 2, "frequency")]["uncalibrated"] is None
        assert rows[("tones", 2, "gain")]["uncalibrated"] is None
        owner = fixture.ctrl.cfg_resources.lookup(tab)
        owner.refresh_sources()
        refreshed = rpc.call("tab.get_cfg", {"tab_id": tab})["result"]
        assert refreshed == observed
        local = rpc.call("tab.describe_cfg", {"tab_id": tab, "path": ["tones", 1]})
        assert local["result"]["diagnostics"] == observed["diagnostics"]
    finally:
        fixture.ctrl.tab_control.close_tab(tab)


def test_single_read_elements_keeps_sibling_drafts(form_rpc):
    from tests.gui.form_adapter._indexed_models import indexed_env

    from ._form_rpc import IndexedRpcAdapter

    fixture, _, rpc = form_rpc
    context = fixture.ctrl.context_control
    library = indexed_env().library
    assert library is not None
    context.set_module(
        context.read_context().target,
        "Q.pulses.drive",
        library.module_spec("Q.pulses.drive"),
    )
    fixture.registry.register("indexed-rpc", IndexedRpcAdapter)
    tab = fixture.ctrl.new_tab("indexed-rpc")
    try:
        owner = fixture.ctrl.cfg_resources.lookup(tab)
        owner.write({("tones", 0, "phase"): {"expr": "1 / 0"}, ("participants",): [1]})
        sibling = owner.input.data["tones"][0]
        owner.use_inline(("tones", 1, "drive"))
        assert owner.input.data["tones"][0] == sibling
        edited = rpc.call(
            "tab.edit_recipe_fields",
            {
                "tab_id": tab,
                "edits": [
                    {
                        "path": ["tones", 1, "drive", "gain"],
                        "value": {"expr": "3 / 10"},
                    },
                ],
            },
        )
        assert edited["ok"], edited
        before = owner.input
        fixture.ctrl.control_identity.reclaim_local()
        observed = rpc.call("tab.read_elements", {"tab_id": tab, "path": ["tones"]})
        assert observed["ok"], observed
        rows = observed["result"]["elements"]
        assert [(row["index"], row["name"]) for row in rows] == [(0, "A"), (1, "B")]
        assert rows[0]["input"]["phase"] == {"expr": "1 / 0"}
        assert any(
            error["path"] == ["tones", 0, "phase"] for error in rows[0]["errors"]
        )
        assert {"path": ["tones", 1, "drive", "module", "gain"], "value": 0.3} in rows[
            1
        ]["values"]
        assert owner.input.same_as(before)
        fixture.state.set_tab_running(tab, running=True)
        assert rpc.call("tab.read_elements", {"tab_id": tab, "path": ["tones"]})["ok"]
        assert owner.input.same_as(before)
        fixture.state.set_tab_running(tab, running=False)
    finally:
        fixture.state.set_tab_running(tab, running=False)
        fixture.ctrl.tab_control.close_tab(tab)


def test_aggregate_relative_fields_use_real_rpc_and_keep_owner_errors(form_rpc):
    from tests.gui.form_adapter._indexed_models import indexed_env

    from ._form_rpc import IndexedRpcAdapter

    fixture, _, rpc = form_rpc
    context = fixture.ctrl.context_control
    library = indexed_env().library
    assert library is not None
    context.set_module(
        context.read_context().target,
        "Q.pulses.drive",
        library.module_spec("Q.pulses.drive"),
    )
    fixture.registry.register("indexed-fields", IndexedRpcAdapter)
    tab = fixture.ctrl.new_tab("indexed-fields")
    try:
        owner = fixture.ctrl.cfg_resources.lookup(tab)
        owner.write({("tones", 0, "phase"): {"expr": "1 / 0"}})
        before = owner.input
        fixture.ctrl.control_identity.reclaim_local()
        fixture.state.set_tab_running(tab, running=True)
        response = rpc.call(
            "tab.read_elements",
            {
                "tab_id": tab,
                "path": ["tones"],
                "fields": [["drive", "gain"], ["phase"]],
            },
        )
        assert response["ok"], response
        rows = response["result"]["elements"]
        assert [row["index"] for row in rows] == [0, 1]
        assert [row["fields"][0]["value"] for row in rows] == [0.2, 0.2]
        assert [row["fields"][0].get("element") for row in rows] == [
            {"index": 0, "name": "A"},
            {"index": 1, "name": "B"},
        ]
        assert rows[0]["fields"][1]["value_status"] != "available"
        assert rows[0]["fields"][1]["errors"]
        assert rows[1]["fields"][1]["value"] == 0
        rows[1]["input"]["name"] = "detached"
        assert owner.input.same_as(before)
        for fields in ("bad", [[]], [["absent"]], [["drive", -1]]):
            refused = rpc.call(
                "tab.read_elements",
                {
                    "tab_id": tab,
                    "path": ["tones"],
                    "fields": fields,
                },
            )
            assert not refused["ok"], refused
        assert owner.input.same_as(before)
    finally:
        fixture.state.set_tab_running(tab, running=False)
        fixture.ctrl.tab_control.close_tab(tab)


def test_fixed_element_rpc_describes_names_and_edits_typed_positions(form_rpc) -> None:
    from pydantic import create_model

    from tests.gui.form_adapter._fixed_models import named_form, named_input

    model = named_form()
    core = create_model(
        "FixedRpcCfg",
        __base__=SampleCfg,
        tones=(
            model.model_fields["tones"].rebuild_annotation(),
            named_input(model).data["tones"],
        ),
        participants=(model.model_fields["participants"].rebuild_annotation(), [1]),
    )

    class FixedAdapter(SampleAdapter):
        @staticmethod
        def make_form(ctx: FormContext):
            del ctx
            return Form(
                core,
                {"sweep": Range.between(1.0, 2.0, expts=3)},
                fixed_elements=model.fixed_elements,
            )

    fixture, _, rpc = form_rpc
    fixture.registry.register("fixed-elements", FixedAdapter)
    tab = fixture.ctrl.new_tab("fixed-elements")
    try:
        reply = rpc.call("tab.describe_cfg", {"tab_id": tab, "path": ["tones"]})
        assert reply["ok"], reply
        local = reply["result"]
        target = local["description"]["target"]
        assert "elements" in target, (
            "Fixed-list description must expose original positions and names"
        )
        assert target["elements"] == [
            {"index": 0, "name": "A", "path": ["tones", 0]},
            {"index": 1, "name": "B", "path": ["tones", 1]},
        ]
        assert local["value"] == [{"label": "A", "gain": 1}, {"label": "B", "gain": 2}]
        reply = rpc.call(
            "tab.edit_cfg",
            {
                "tab_id": tab,
                "edits": [{"path": ["tones", 1, "gain"], "value": {"expr": "3 * 2"}}],
            },
        )
        assert reply["ok"], reply
        local = rpc.call(
            "tab.describe_cfg", {"tab_id": tab, "path": ["tones", 1, "gain"]}
        )
        assert local["ok"], local
        assert local["result"]["description"]["path"] == ["tones", 1, "gain"]
        assert local["result"]["input_present"] is True
        assert local["result"]["input"] == {"expr": "3 * 2"}
        assert local["result"]["value"] == 6
        refused = rpc.call(
            "tab.edit_cfg",
            {"tab_id": tab, "edits": [{"path": ["tones", 0, "label"], "value": "C"}]},
        )
        assert not refused["ok"], refused
        assert refused["error"]["reason"] == "form_input_rejected"
        assert refused["error"]["data"]["form_path"] == ["tones", 0, "label"]
        unchanged = rpc.call(
            "tab.describe_cfg", {"tab_id": tab, "path": ["tones", 0, "label"]}
        )
        assert unchanged["ok"], unchanged
        assert unchanged["result"]["input"] == "A"
        for path in [
            ["tones", "1", "gain"],
            ["tones", -1],
            ["tones", 2],
            ["tones", True],
        ]:
            invalid = rpc.call("tab.describe_cfg", {"tab_id": tab, "path": path})
            assert not invalid["ok"], invalid
            assert invalid["error"]["code"] == "invalid_params"
            assert invalid["error"]["reason"] == "form_path"
            assert invalid["error"]["data"]["form_path"] == path
    finally:
        fixture.ctrl.tab_control.close_tab(tab)
