"""Library mutations and active draft projections through the live MCP socket."""

from __future__ import annotations

import pytest
from zcu_tools.cfg_model import module_spec_from_raw, waveform_spec_from_raw
from zcu_tools.gui.app.measure.cfg_schemas import (
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
from zcu_tools.gui.session.context_types import LibraryChange
from zcu_tools.program.v2 import ModuleCfgFactory, WaveformCfgFactory
from zcu_tools.resources.entry.registry import component_registry

from zcu_lab.components import QubitSchema

from ._helpers import Fixture, mcp_client


@pytest.fixture(params=["module", "waveform"])
def reference_library(request, qapp):
    """Real program configs shared by linked and locally modified references."""
    kind = request.param
    before = component_registry.kinds()
    added = "qubit/transmon" not in before
    if added:
        component_registry.register("qubit/transmon", QubitSchema)
    fx = Fixture(active_label="ctx001")
    fx.ctrl.context_control.add_component("Q1", "qubit/transmon", {})
    parameters = fx.ctrl.read_context()
    assert parameters is not None
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
        spec, value = module_cfg_to_value(cfg)
        field = "gain"
        fx.ctrl.context_control.set_module(
            parameters.target,
            "Q1.pulses.seed",
            module_spec_from_raw(cfg.model_dump(mode="python")),
        )
    else:
        waveform = WaveformCfgFactory.from_raw({"style": "const", "length": 0.1})
        spec, value = waveform_cfg_to_value(waveform)
        field = "length"
        fx.ctrl.context_control.set_waveform(
            parameters.target,
            "Q1.waveforms.seed",
            waveform_spec_from_raw(waveform.model_dump(mode="python")),
        )
    return fx, kind, spec, value, field


@pytest.mark.parametrize("action", ["rename", "del"])
def test_library_mutation_refreshes_linked_and_modified_tab_drafts(
    qapp, tmp_path, monkeypatch, reference_library, action
):
    fx, kind, spec, value, field = reference_library
    path = "Q1.pulses.seed" if kind == "module" else "Q1.waveforms.seed"
    moved = "Q1.pulses.moved" if kind == "module" else "Q1.waveforms.moved"
    schema = CfgSchema(
        CfgSectionSpec(
            fields={
                name: ReferenceSpec(kind, [spec]) for name in ("linked", "modified")
            }
        ),
        CfgSectionValue(
            {name: ReferenceValue(path, value) for name in ("linked", "modified")}
        ),
    )
    from zcu_tools.gui.cfg.resource import CfgEdit

    from zcu_lab.v2.fake.stub.gui import FakeAdapter

    tab_id = "references"
    cfg = fx.prepare_tab(tab_id, FakeAdapter(), schema)
    cfg.edit(cfg.observe().ref.revision, (CfgEdit(("modified", field), 0.75),))
    fx.start()
    monkeypatch.setattr("zcu_tools.mcp.measure.tools_lifecycle.status", lambda *_: {})
    bridge, invoke = mcp_client(fx.service.port, tmp_path)
    try:
        invoke("connect", {"port": fx.service.port})
        before = invoke(
            "rpc_call", {"method": "tab.get_cfg", "params": {"tab_id": tab_id}}
        )["tree"]
        assert before["children"]["linked"]["valid"] is True
        assert before["children"]["linked"]["ref"] == path
        assert before["children"]["modified"]["valid"] is True
        invoke("rpc_call", {"method": "context.snapshot"})
        parameters = fx.ctrl.read_context()
        assert parameters is not None
        fx.ctrl.apply_library_change(
            parameters,
            LibraryChange(
                kind,
                "rename" if action == "rename" else "delete",
                path,
                new_path=moved if action == "rename" else None,
            ),
        )
        after = invoke(
            "rpc_call", {"method": "tab.get_cfg", "params": {"tab_id": tab_id}}
        )["tree"]
        linked = after["children"]["linked"]
        assert linked["ref"] == path
        assert linked["valid"] is False
        assert linked["error"] is not None
        modified = after["children"]["modified"]
        assert modified["ref"] == path
        assert modified["is_overridden"] is True
        assert modified["valid"] is True
        assert modified["children"][field]["input"]["resolved"] == 0.75
        published = cfg.snapshot_inputs().value.fields
        published_linked = published["linked"]
        published_modified = published["modified"]
        assert isinstance(published_linked, ReferenceValue)
        assert isinstance(published_modified, ReferenceValue)
        assert published_linked.chosen_key == path
        assert linked["error"] is not None
        assert published_modified.chosen_key == path
        assert published_modified.is_overridden is True
    finally:
        bridge.disconnect()
        fx.stop()
