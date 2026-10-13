"""ModuleLibrary tools through the public MCP handler and live GUI socket."""

from __future__ import annotations

import pytest
from zcu_tools.cfg_model import ConstWaveformCfg
from zcu_tools.gui.app.measure.template_catalog import TemplateCatalog
from zcu_tools.gui.session.context_types import LibraryChange
from zcu_tools.mcp.measure.session import GuiRpcError

from tests.gui.app.measure._templates import register_all_templates

from ._helpers import Fixture, mcp_client


@pytest.fixture()
def library_client(qapp, tmp_path, monkeypatch):
    catalog = TemplateCatalog()
    register_all_templates(catalog)
    fx = Fixture(active_label="ctx001", template_catalog=catalog)
    context = fx.ctrl.context_control
    context.set_waveform(
        context.read_context().target, "Q1.waveforms.seed", ConstWaveformCfg(length=0.1)
    )
    fx.start()
    monkeypatch.setattr("zcu_tools.mcp.measure.tools_lifecycle.status", lambda *_: {})
    bridge, invoke = mcp_client(fx.service.port, tmp_path)
    try:
        invoke("connect", {"port": fx.service.port})
        yield invoke, fx
    finally:
        bridge.disconnect()
        fx.stop()


def waveforms(fx):
    library = fx.ctrl.read_context().library
    return {
        key: library.waveform_spec(key).model_dump(mode="python")
        for key in library.waveform_keys()
    }


def modules(fx):
    library = fx.ctrl.read_context().library
    return {
        key: library.module_spec(key).model_dump(mode="python")
        for key in library.module_keys()
    }


@pytest.mark.parametrize("kind", ["module", "waveform"])
def test_ml_rename_then_delete_updates_only_the_selected_collection(
    library_client, kind
):
    invoke, fx = library_client
    invoke(
        "rpc_call",
        {
            "method": "context.ml_create_from_role",
            "params": {"role_id": "none_reset", "name": "Q1.reset.seed"},
        },
    )
    invoke("rpc_call", {"method": "context.snapshot"})
    selected = modules if kind == "module" else waveforms
    other = waveforms if kind == "module" else modules
    family = "reset" if kind == "module" else "waveforms"
    other_family = "waveforms" if kind == "module" else "reset"
    seed = f"Q1.{family}.seed"
    moved = f"Q1.{family}.moved"
    other_seed = f"Q1.{other_family}.seed"
    before = selected(fx)[seed]
    other_before = other(fx)[other_seed]
    fx.ctrl.apply_library_change(
        fx.ctrl.read_context(), LibraryChange(kind, "rename", seed, new_path=moved)
    )
    assert seed not in selected(fx)
    assert selected(fx)[moved] == before
    assert other(fx)[other_seed] == other_before
    fx.ctrl.apply_library_change(
        fx.ctrl.read_context(), LibraryChange(kind, "delete", moved)
    )
    assert moved not in selected(fx)
    assert other(fx)[other_seed] == other_before


def test_ml_rename_rejects_collision_without_overwriting(library_client):
    invoke, fx = library_client
    fx.ctrl.context_control.set_waveform(
        fx.ctrl.read_context().target,
        "Q1.waveforms.occupied",
        ConstWaveformCfg(length=0.3),
    )
    invoke("rpc_call", {"method": "context.snapshot"})
    before = waveforms(fx)
    with pytest.raises(ValueError, match="exists|collision|already"):
        fx.ctrl.apply_library_change(
            fx.ctrl.read_context(),
            LibraryChange(
                "waveform",
                "rename",
                "Q1.waveforms.seed",
                new_path="Q1.waveforms.occupied",
            ),
        )
    assert waveforms(fx) == before


def test_ml_get_index_and_named_cfg_are_read_only(library_client):
    invoke, fx = library_client
    listed = invoke("rpc_call", {"method": "context.snapshot", "params": {}})
    seed = listed["library"]["components"]["Q1"]["waveforms"]["seed"]
    assert seed["style"] == "const"
    named = invoke(
        "rpc_call",
        {
            "method": "context.snapshot",
            "params": {"path": "library.components.Q1.waveforms.seed"},
        },
    )
    assert named["value"] == waveforms(fx)["Q1.waveforms.seed"]
    with pytest.raises(GuiRpcError, match="missing"):
        invoke(
            "rpc_call",
            {
                "method": "context.snapshot",
                "params": {"path": "library.components.Q1.waveforms.missing"},
            },
        )
    assert sorted(waveforms(fx)) == ["Q1.waveforms.seed"]


def test_ml_roles_and_create_use_gui_role_defaults(library_client):
    invoke, fx = library_client
    roles = invoke("rpc_call", {"method": "context.ml_list_roles", "params": {}})[
        "roles"
    ]
    role = next(item for item in roles if item["role_id"] == "none_reset")
    assert role["item_kind"] == "module"
    assert role["default_name"] == "reset_none"
    created = invoke(
        "rpc_call",
        {
            "method": "context.ml_create_from_role",
            "params": {"role_id": "none_reset", "name": "Q1.reset.reset_none"},
        },
    )
    assert created == {"created": "Q1.reset.reset_none"}
    stored = invoke(
        "rpc_call",
        {
            "method": "context.snapshot",
            "params": {"path": "library.components.Q1.reset.reset_none"},
        },
    )
    assert stored["value"] == modules(fx)["Q1.reset.reset_none"]
