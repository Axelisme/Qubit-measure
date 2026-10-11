"""Native library mutations projected through the live MCP socket."""

from __future__ import annotations

import pytest
from zcu_tools.cfg_model import ModuleCfgFactory, WaveformCfgFactory
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
from zcu_tools.gui.cfg.resource import CfgEdit
from zcu_tools.gui.session.context_types import LibraryChange

from ._helpers import Fixture, mcp_client


@pytest.fixture(params=["module", "waveform"])
def reference_library(request):
    """Real program schemas for ref and explicit inline references."""
    kind = request.param
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
        field, path = "gain", "Q1.pulses.seed"
    else:
        cfg = WaveformCfgFactory.from_raw({"style": "const", "length": 0.1})
        spec, value = waveform_cfg_to_value(cfg)
        field, path = "length", "Q1.waveforms.seed"
    return kind, spec, value, field, path


@pytest.mark.parametrize("action", ["rename", "del"])
def test_library_mutation_refreshes_ref_and_explicit_inline_tab_cfg(
    qapp, tmp_path, monkeypatch, reference_library, action
):
    kind, spec, value, field, path = reference_library
    fx = Fixture(project_root=str(tmp_path), active_label="ctx001")
    fx.ctrl.apply_library_change(
        fx.ctrl.read_context(),
        LibraryChange(kind, "put", path, schema=CfgSchema(spec, value)),
    )
    schema = CfgSchema(
        CfgSectionSpec(
            fields={name: ReferenceSpec(kind, [spec]) for name in ("linked", "inline")}
        ),
        CfgSectionValue(
            {name: ReferenceValue(path, value) for name in ("linked", "inline")}
        ),
    )
    from zcu_lab.v2.fake.stub.gui import FakeAdapter

    tab_id = "references"
    cfg = fx.prepare_tab(tab_id, FakeAdapter(), schema)
    converted = cfg.convert_reference_to_inline(cfg.observe().ref.revision, ("inline",))
    cfg.edit(converted.ref.revision, (CfgEdit(("inline", field), 0.75),))
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
        assert before["children"]["linked"]["mode"] == "ref"
        assert before["children"]["inline"]["mode"] == "inline"
        fx.ctrl.apply_library_change(
            fx.ctrl.read_context(),
            LibraryChange(
                kind,
                "rename" if action == "rename" else "delete",
                path,
                new_path=path.replace("seed", "moved") if action == "rename" else None,
            ),
        )
        after = invoke(
            "rpc_call", {"method": "tab.get_cfg", "params": {"tab_id": tab_id}}
        )["tree"]
        linked = after["children"]["linked"]
        assert linked["ref"] == path
        assert linked["valid"] is False
        assert linked["error"] is not None
        inline = after["children"]["inline"]
        assert inline["mode"] == "inline"
        assert inline["valid"] is True
        assert inline["children"][field]["input"]["resolved"] == 0.75
        published = cfg.snapshot_inputs().value.fields
        published_linked, published_inline = published["linked"], published["inline"]
        assert isinstance(published_linked, ReferenceValue)
        assert isinstance(published_inline, ReferenceValue)
        assert published_linked.chosen_key == path
        assert published_inline.chosen_key == f"<Custom:{spec.label}>"
        published_inline.value.fields.clear()
        again = invoke(
            "rpc_call", {"method": "tab.get_cfg", "params": {"tab_id": tab_id}}
        )["tree"]
        assert (
            again["children"]["inline"]["children"][field]["input"]["resolved"] == 0.75
        )
    finally:
        bridge.disconnect()
        fx.stop()
