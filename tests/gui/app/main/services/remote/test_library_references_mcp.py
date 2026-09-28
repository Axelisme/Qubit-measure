"""Library mutations and active draft projections through the live MCP socket."""

from __future__ import annotations

from dataclasses import replace

import pytest
from zcu_tools.gui.app.main.cfg_schemas import (
    module_cfg_to_value,
    waveform_cfg_to_value,
)
from zcu_tools.gui.cfg import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    ReferenceSpec,
    ReferenceValue,
)
from zcu_tools.meta_tool import MetaDict, ModuleLibrary
from zcu_tools.program.v2 import ModuleCfgFactory, WaveformCfgFactory

from ._helpers import Fixture, mcp_client


@pytest.fixture(params=["module", "waveform"])
def reference_library(request):
    """Real program configs shared by linked and locally modified references."""
    kind = request.param
    library = ModuleLibrary()
    if kind == "module":
        cfg = ModuleCfgFactory.from_raw(
            {
                "type": "pulse",
                "ch": 0,
                "nqz": 1,
                "freq": 100.0,
                "gain": 0.5,
                "phase": 0.0,
                "pre_delay": 0.0,
                "post_delay": 0.0,
                "waveform": {"style": "const", "length": 0.1},
            }
        )
        library.modules["seed"] = cfg
        spec, value = module_cfg_to_value(cfg)
        field = "gain"
    else:
        waveform = WaveformCfgFactory.from_raw({"style": "const", "length": 0.1})
        library.waveforms["seed"] = waveform
        spec, value = waveform_cfg_to_value(waveform)
        field = "length"
    return kind, library, spec, value, field


@pytest.mark.parametrize("tool", ["ml_rename", "ml_delete"])
def test_library_mutation_refreshes_linked_and_modified_tab_drafts(
    qapp, tmp_path, monkeypatch, reference_library, tool
):
    kind, library, spec, value, field = reference_library
    schema = CfgSchema(
        CfgSectionSpec(
            fields={
                name: ReferenceSpec(kind, [spec]) for name in ("linked", "modified")
            }
        ),
        CfgSectionValue(
            {name: ReferenceValue("seed", value) for name in ("linked", "modified")}
        ),
    )
    fx = Fixture(active_label="ctx001")
    fx.state.set_context(replace(fx.state.exp_context, md=MetaDict(), ml=library))
    tab_id = fx.ctrl.new_tab("fake")
    editor_id, _ = fx.ctrl.open_seeded_cfg_editor(schema, gc=False, owner_key=tab_id)
    fx.ctrl.cfg_editor_set_field(editor_id, f"modified.{field}", 0.75)
    fx.start()
    monkeypatch.setattr("zcu_tools.mcp.measure.tools_lifecycle.status", lambda *_: {})
    bridge, invoke = mcp_client(fx.service.port, tmp_path)
    try:
        invoke("connect", {"port": fx.service.port})
        before = invoke(
            "rpc_call", {"method": "tab.get_cfg", "params": {"tab_id": tab_id}}
        )["tree"]
        assert before["children"]["linked"]["valid"] is True
        assert before["children"]["linked"]["ref"] == "seed"
        assert before["children"]["modified"]["valid"] is True
        invoke("rpc_call", {"method": "context.snapshot"})
        arguments = {"name": "seed", "kind": kind}
        if tool == "ml_rename":
            arguments["new_name"] = "moved"
        invoke(tool, arguments)
        after = invoke(
            "rpc_call", {"method": "tab.get_cfg", "params": {"tab_id": tab_id}}
        )["tree"]
        linked = after["children"]["linked"]
        assert linked["ref"] == "seed"
        assert linked["valid"] is False
        assert linked["error"] is not None
        modified = after["children"]["modified"]
        assert modified["ref"] == f"<Custom:{spec.label}>"
        assert modified["valid"] is True
        assert modified["children"][field]["input"]["resolved"] == 0.75
        published = fx.state.get_tab(tab_id).cfg_schema.value.fields
        published_linked = published["linked"]
        published_modified = published["modified"]
        assert isinstance(published_linked, ReferenceValue)
        assert isinstance(published_modified, ReferenceValue)
        assert published_linked.chosen_key == "seed"
        assert published_linked.error is not None
        assert published_modified.chosen_key == f"<Custom:{spec.label}>"
    finally:
        bridge.disconnect()
        fx.stop()
