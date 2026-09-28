"""ModuleLibrary tools through the public MCP handler and live GUI socket."""

from __future__ import annotations

from dataclasses import replace

import pytest
from zcu_tools.experiment.v2_gui.role_registry import register_all_roles
from zcu_tools.gui.app.main.role_catalog import RoleCatalog
from zcu_tools.mcp.measure.session import GuiRpcError
from zcu_tools.meta_tool import MetaDict, ModuleLibrary
from zcu_tools.program.v2 import ModuleCfgFactory, WaveformCfgFactory

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


def test_ml_edit_commits_prefix_and_save_as_preserves_source(library_client):
    invoke, library = library_client
    before = library.waveforms["seed"].to_dict()
    invoke("rpc_call", {"method": "context.snapshot"})
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
        "applied": 1,
        "failed": None,
        "skipped": [],
    }
    assert saved["cfg"]["length"] == pytest.approx(0.25)
    assert library.waveforms["seed"].to_dict() == before
    partial = invoke(
        "ml_edit",
        {
            "name": "seed",
            "edits": [
                {"path": "length", "value": 0.5},
                {"path": "missing", "value": 1.0},
                {"path": "length", "value": 0.9},
            ],
        },
    )
    assert partial["applied"] == 1
    assert partial["failed"]["index"] == 1
    assert partial["failed"]["path"] == "missing"
    assert partial["skipped"] == [2]
    assert partial["cfg"]["length"] == 0.5
    assert library.waveforms["seed"].to_dict()["length"] == 0.5
    assert library.waveforms["copy"].to_dict() == saved["cfg"]
    continued = invoke(
        "ml_edit", {"name": "seed", "edits": [{"path": "length", "value": 0.75}]}
    )
    assert continued["cfg"]["length"] == 0.75


def test_ml_edit_first_failure_leaves_save_as_uncreated(library_client):
    invoke, library = library_client
    invoke("rpc_call", {"method": "context.snapshot"})
    result = invoke(
        "ml_edit",
        {
            "name": "seed",
            "save_as": "copy",
            "edits": [
                {"path": "missing", "value": 1.0},
                {"path": "length", "value": 0.9},
            ],
        },
    )
    assert result["applied"] == 0
    assert result["failed"]["index"] == 0
    assert result["skipped"] == [1]
    assert result["cfg"] is None
    assert "copy" not in library.waveforms
    assert library.waveforms["seed"].to_dict()["length"] == 0.1


def test_ml_edit_module_and_eval_use_shared_lowering(library_client):
    invoke, library = library_client
    library.modules["pulse"] = ModuleCfgFactory.from_raw(
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
    invoke("rpc_call", {"method": "context.snapshot"})
    result = invoke(
        "ml_edit",
        {
            "name": "pulse",
            "kind": "module",
            "edits": [
                {"path": "gain", "value": {"__kind": "eval", "expr": "0.25 + 0.5"}}
            ],
        },
    )
    assert result["applied"] == 1
    assert result["failed"] is None
    assert result["cfg"]["gain"] == 0.75
    assert library.modules["pulse"].to_dict()["gain"] == 0.75


def test_ml_edit_requires_explicit_context_observation(library_client):
    invoke, library = library_client
    before = library.waveforms["seed"].to_dict()
    with pytest.raises(GuiRpcError) as exc:
        invoke("ml_edit", {"name": "seed", "edits": [{"path": "length", "value": 0.5}]})
    assert exc.value.reason == "stale_version"
    assert library.waveforms["seed"].to_dict() == before
    invoke("rpc_call", {"method": "context.snapshot"})
    saved = invoke(
        "ml_edit", {"name": "seed", "edits": [{"path": "length", "value": 0.5}]}
    )
    assert saved["cfg"]["length"] == pytest.approx(0.5)


def test_ml_edit_rejects_context_changed_by_another_connection(
    qapp, tmp_path, monkeypatch
):
    fx = Fixture(active_label="ctx001")
    library = ModuleLibrary()
    library.waveforms["seed"] = WaveformCfgFactory.from_raw(
        {"style": "const", "length": 0.1}
    )
    fx.state.set_context(replace(fx.state.exp_context, md=MetaDict(), ml=library))
    fx.start()
    monkeypatch.setattr("zcu_tools.mcp.measure.tools_lifecycle.status", lambda *_: {})
    first_bridge, first = mcp_client(fx.service.port, tmp_path / "first")
    second_bridge, second = mcp_client(fx.service.port, tmp_path / "second")
    try:
        for invoke in (first, second):
            invoke("connect", {"port": fx.service.port})
            invoke("rpc_call", {"method": "context.snapshot"})
        second(
            "ml_edit", {"name": "seed", "edits": [{"path": "length", "value": 0.25}]}
        )
        with pytest.raises(GuiRpcError) as error:
            first(
                "ml_edit", {"name": "seed", "edits": [{"path": "length", "value": 0.9}]}
            )
        assert error.value.reason == "stale_version"
        assert library.waveforms["seed"].to_dict()["length"] == 0.25
    finally:
        first_bridge.disconnect()
        second_bridge.disconnect()
        fx.stop()


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
