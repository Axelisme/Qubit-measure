"""Indexed module and fixed-element aggregate contracts through Form RPC."""

from zcu_tools.cfg_model import Range
from zcu_tools.gui.form_adapter import Form, FormContext

from tests.gui.app.measure._synthetic import SampleAdapter, SampleCfg


def test_indexed_module_rpc_and_single_read_elements_keep_sibling_drafts(form_rpc):
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
        owner.write(
            {("tones", 0, "phase"): {"expr": "1 / 0"}, ("participants",): ["B"]}
        )
        sibling = owner.input.data["tones"][0]
        read = rpc.call(
            "tab.read_module", {"tab_id": tab, "field": ["tones", 1, "drive"]}
        )
        assert read["ok"], read
        assert read["result"]["value"]["gain"] == 0.2
        copied = rpc.call(
            "tab.use_inline", {"tab_id": tab, "field": ["tones", 1, "drive"]}
        )
        assert copied["ok"], copied
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
        for bad_path in (
            ["tones", 5, "drive"],
            ["tones", "1", "drive"],
            ["tones", -1, "drive"],
        ):
            refused = rpc.call("tab.read_module", {"tab_id": tab, "field": bad_path})
            assert not refused["ok"], refused
            assert refused["error"]["data"]["form_path"] == bad_path
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
        participants=(model.model_fields["participants"].rebuild_annotation(), ["B"]),
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
