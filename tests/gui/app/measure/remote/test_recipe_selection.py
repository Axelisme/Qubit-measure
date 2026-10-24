"""Module selection replies and correction paths use the active Form contract."""

from dataclasses import replace

import pytest
from zcu_tools.cfg_model import ConfigBase, MuxReadoutCfg, MuxReadoutSpec, Range
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
)
from zcu_tools.gui.form_adapter import Form, FormContext

from tests.gui.app.measure._synthetic import SampleAdapter, SampleCfg


class MuxModules(ConfigBase):
    readout: MuxReadoutCfg


class MuxCfg(SampleCfg):
    modules: MuxModules


def MuxAdapter_make_form(ctx: FormContext) -> Form[MuxCfg]:
    del ctx
    return Form(MuxCfg, {"sweep": Range.between(1, 2, expts=3)})


MuxAdapter = Adapter(
    spec=replace(SampleAdapter.spec, cfg_type=MuxCfg),
    make_form=MuxAdapter_make_form,
    run=SampleAdapter.run,
    guide=None,
    analysis=SampleAdapter.analysis,
    post_analysis=SampleAdapter.post_analysis,
)


@pytest.fixture
def mux_rpc(form_rpc):
    fixture, _, rpc = form_rpc
    context = fixture.ctrl.context_control
    context.set_module(
        context.read_context().target,
        "MG.readout.mux",
        MuxReadoutSpec.model_validate(
            {
                "gen_ch": 0,
                "nqz": 1,
                "mixer_freq": 6000,
                "length": 1,
                "ro_length": 1,
                "tones": [
                    {"freq": 6010, "gain": 0.1, "ro_ch": 0, "label": "A"},
                    {"freq": 6020, "gain": 0.2, "ro_ch": 1, "label": "B"},
                ],
            }
        ),
    )
    fixture.registry.register("mux-selection", MuxAdapter)
    tab = fixture.ctrl.new_tab("mux-selection")
    try:
        yield fixture, tab, rpc
    finally:
        fixture.ctrl.tab_control.close_tab(tab)


def test_mux_selection_returns_complete_indexed_capture_in_one_reply(mux_rpc):
    fixture, tab, rpc = mux_rpc
    selected = rpc.call(
        "tab.select_recipe_module",
        {
            "tab_id": tab,
            "path": ["modules", "readout"],
            "action": "use_library",
            "key": "MG.readout.mux",
        },
    )
    assert selected["ok"], selected
    field = selected["result"]["field"]
    assert selected["result"]["changed"] is True
    assert field["value"]["tones"] == [
        {"freq": 6010, "gain": 0.1, "phase": 0, "ro_ch": 0, "label": "A"},
        {"freq": 6020, "gain": 0.2, "phase": 0, "ro_ch": 1, "label": "B"},
    ]
    tones = rpc.call(
        "tab.read_recipe_field",
        {"tab_id": tab, "path": ["modules", "readout", "tones"]},
    )
    assert tones["ok"], tones
    assert [child["path"] for child in tones["result"]["description"]["children"]] == [
        ["modules", "readout", "tones", 0],
        ["modules", "readout", "tones", 1],
    ]
    before = fixture.ctrl.cfg_resources.lookup(tab).input
    rejected = rpc.call(
        "tab.select_recipe_module",
        {
            "tab_id": tab,
            "path": ["modules", "readout"],
            "action": "use_library",
            "key": "MG.readout.missing",
        },
    )
    assert not rejected["ok"]
    assert fixture.ctrl.cfg_resources.lookup(tab).input.same_as(before)


def test_unknown_logical_selector_lists_real_root_fields(mux_rpc):
    _, tab, rpc = mux_rpc
    rejected = rpc.call(
        "tab.select_recipe_module",
        {"tab_id": tab, "path": ["readout"], "action": "use_inline"},
    )
    assert not rejected["ok"]
    assert rejected["error"]["reason"] == "form_path"
    assert "modules" in rejected["error"]["message"]


@pytest.mark.parametrize(
    "path,value",
    [
        (["modules", "readout"], "MG.readout.mux"),
        (["modules", "readout", "module", "gen_ch"], 1),
    ],
)
def test_cfg_edit_rejection_lists_active_paths_and_selector_command(
    mux_rpc, path, value
):
    fixture, tab, rpc = mux_rpc
    owner = fixture.ctrl.cfg_resources.lookup(tab)
    owner.select_recipe_module(
        ("modules", "readout"), "use_library", key="MG.readout.mux"
    )
    before = owner.input
    rejected = rpc.call(
        "tab.edit_cfg", {"tab_id": tab, "edits": [{"path": path, "value": value}]}
    )
    assert not rejected["ok"]
    assert "data" in rejected["error"], rejected
    assert ["gain"] in rejected["error"]["data"]["editable_paths"]
    assert ["modules", "readout"] in rejected["error"]["data"]["selection_paths"]
    assert "tab.select_recipe_module" in rejected["error"]["message"]
    assert owner.input.same_as(before)
