"""ModuleLibrary tools through the public MCP handler and live GUI socket."""

from __future__ import annotations

from dataclasses import replace

import pytest
from zcu_tools.experiment.v2_gui.role_registry import register_all_roles
from zcu_tools.gui.app.main.role_catalog import RoleCatalog
from zcu_tools.mcp.measure.session import GuiRpcError
from zcu_tools.meta_tool import MetaDict, ModuleLibrary
from zcu_tools.program.v2 import WaveformCfgFactory

from ._helpers import Fixture, mcp_client


@pytest.fixture()
def library_client(qapp, tmp_path, monkeypatch):
    catalog = RoleCatalog()
    register_all_roles(catalog)
    fx = Fixture(active_label="ctx001", role_catalog=catalog)
    library = ModuleLibrary()
    library.waveforms["seed"] = WaveformCfgFactory.from_raw(
        {"style": "const", "length": 0.1}
    )
    fx.state.set_context(replace(fx.state.exp_context, md=MetaDict(), ml=library))
    fx.start()
    monkeypatch.setattr("zcu_tools.mcp.measure.tools_lifecycle.status", lambda *_: {})
    bridge, invoke = mcp_client(fx.service.port, tmp_path)
    try:
        invoke("connect", {"port": fx.service.port})
        yield invoke, library
    finally:
        bridge.disconnect()
        fx.stop()


@pytest.mark.parametrize("kind", ["module", "waveform"])
def test_ml_rename_then_delete_updates_only_the_selected_collection(
    library_client, kind
):
    invoke, library = library_client
    invoke("ml_create", {"role_id": "none_reset", "name": "seed"})
    invoke("rpc_call", {"method": "context.snapshot"})
    selected = library.modules if kind == "module" else library.waveforms
    other = library.waveforms if kind == "module" else library.modules
    before = selected["seed"].to_dict()
    other_before = other["seed"].to_dict()

    renamed = invoke("ml_rename", {"name": "seed", "new_name": "moved", "kind": kind})
    assert renamed["renamed"] == "moved"
    assert "seed" not in selected
    assert selected["moved"].to_dict() == before
    assert other["seed"].to_dict() == other_before

    deleted = invoke("ml_delete", {"name": "moved"})
    assert deleted["deleted"] == "moved"
    assert "moved" not in selected
    assert other["seed"].to_dict() == other_before


@pytest.mark.parametrize("tool", ["ml_rename", "ml_delete"])
def test_ml_mutation_rejects_ambiguous_names_without_changing_library(
    library_client, tool
):
    invoke, library = library_client
    invoke("ml_create", {"role_id": "none_reset", "name": "seed"})
    invoke("rpc_call", {"method": "context.snapshot"})
    before = (library.modules["seed"].to_dict(), library.waveforms["seed"].to_dict())
    arguments = {"name": "seed"}
    if tool == "ml_rename":
        arguments["new_name"] = "moved"
    with pytest.raises(ValueError, match="ambiguous"):
        invoke(tool, arguments)
    assert (
        library.modules["seed"].to_dict(),
        library.waveforms["seed"].to_dict(),
    ) == before


def test_ml_rename_rejects_collision_without_overwriting(library_client):
    invoke, library = library_client
    library.waveforms["occupied"] = WaveformCfgFactory.from_raw(
        {"style": "const", "length": 0.3}
    )
    invoke("rpc_call", {"method": "context.snapshot"})
    before = {name: cfg.to_dict() for name, cfg in library.waveforms.items()}
    with pytest.raises((ValueError, GuiRpcError), match="exists|collision|already"):
        invoke("ml_rename", {"name": "seed", "new_name": "occupied"})
    assert {name: cfg.to_dict() for name, cfg in library.waveforms.items()} == before


def test_ml_get_index_and_named_cfg_are_read_only(library_client):
    invoke, library = library_client
    listed = invoke("ml_get", {})
    seed = next(item for item in listed["waveforms"] if item["name"] == "seed")
    assert seed["style"] == "const"
    assert isinstance(seed["description"], str) and seed["description"]
    named = invoke("ml_get", {"name": "seed"})
    assert named == {
        "name": "seed",
        "kind": "waveform",
        "cfg": library.waveforms["seed"].to_dict(),
    }
    with pytest.raises(GuiRpcError, match="missing"):
        invoke("ml_get", {"name": "missing"})
    assert sorted(library.waveforms) == ["seed"]


def test_ml_roles_and_create_use_gui_role_defaults(library_client):
    invoke, library = library_client
    roles = invoke("ml_roles", {})
    role = next(item for item in roles if item["role_id"] == "none_reset")
    assert role["kind"] == "module"
    assert role["default_name"] == "reset_none"
    created = invoke("ml_create", {"role_id": "none_reset"})
    assert created == {
        "name": "reset_none",
        "kind": "module",
        "cfg": library.modules["reset_none"].to_dict(),
    }
    with pytest.raises((GuiRpcError, ValueError), match="name"):
        invoke("ml_create", {"role_id": "const:blank"})


def test_ml_edit_commits_only_full_draft_and_save_as_preserves_source(library_client):
    invoke, library = library_client
    before = library.waveforms["seed"].to_dict()
    saved = invoke(
        "ml_edit",
        {
            "name": "seed",
            "edits": [{"path": "length", "value": 0.25}],
            "save_as": "copy",
        },
    )
    assert saved == {
        "name": "copy",
        "cfg": library.waveforms["copy"].to_dict(),
    }
    assert saved["cfg"]["length"] == pytest.approx(0.25)
    assert library.waveforms["seed"].to_dict() == before
    with pytest.raises(
        GuiRpcError, match="unknown.*after 1 applied|after 1 applied.*unknown"
    ):
        invoke(
            "ml_edit",
            {
                "name": "seed",
                "edits": [
                    {"path": "length", "value": 0.5},
                    {"path": "missing", "value": 1.0},
                ],
            },
        )
    assert library.waveforms["seed"].to_dict() == before
    assert library.waveforms["copy"].to_dict() == saved["cfg"]


@pytest.mark.parametrize("destination", ["seed", "occupied"])
def test_ml_edit_save_as_rejects_existing_name_without_overwriting(
    library_client, destination
):
    invoke, library = library_client
    original = library.waveforms["seed"].to_dict()
    library.waveforms["occupied"] = WaveformCfgFactory.from_raw(
        {"style": "const", "length": 0.3}
    )
    occupied = library.waveforms["occupied"].to_dict()

    with pytest.raises(ValueError, match="already exists"):
        invoke(
            "ml_edit",
            {
                "name": "seed",
                "edits": [{"path": "length", "value": 0.5}],
                "save_as": destination,
            },
        )

    assert library.waveforms["seed"].to_dict() == original
    assert library.waveforms["occupied"].to_dict() == occupied
