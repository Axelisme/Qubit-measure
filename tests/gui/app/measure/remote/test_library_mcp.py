"""ModuleLibrary tools through the public MCP handler and live GUI socket."""

from __future__ import annotations

from dataclasses import replace

import pytest
from zcu_tools.gui.app.measure.role_catalog import RoleCatalog
from zcu_tools.mcp.measure.session import GuiRpcError
from zcu_tools.program.v2 import ModuleCfgFactory, WaveformCfgFactory
from zcu_tools.resources.context import MetaDict, ModuleLibrary

from zcu_lab.roles import register_all_roles

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
    fx.state.set_context(replace(fx.state.session_env, md=MetaDict(), ml=library))
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
    invoke(
        "rpc_call",
        {
            "method": "context.ml_create_from_role",
            "params": {"role_id": "none_reset", "name": "seed"},
        },
    )
    invoke("rpc_call", {"method": "context.snapshot"})
    selected = library.modules if kind == "module" else library.waveforms
    other = library.waveforms if kind == "module" else library.modules
    before = selected["seed"].to_dict()
    other_before = other["seed"].to_dict()

    renamed = invoke(
        "rpc_call",
        {
            "method": f"context.ml_rename_{kind}",
            "params": {"old": "seed", "new": "moved"},
        },
    )
    assert renamed["renamed"] == "moved"
    assert "seed" not in selected
    assert selected["moved"].to_dict() == before
    assert other["seed"].to_dict() == other_before

    deleted = invoke(
        "rpc_call", {"method": f"context.ml_del_{kind}", "params": {"name": "moved"}}
    )
    assert deleted["deleted"] == "moved"
    assert "moved" not in selected
    assert other["seed"].to_dict() == other_before


def test_ml_rename_rejects_collision_without_overwriting(library_client):
    invoke, library = library_client
    library.waveforms["occupied"] = WaveformCfgFactory.from_raw(
        {"style": "const", "length": 0.3}
    )
    invoke("rpc_call", {"method": "context.snapshot"})
    before = {name: cfg.to_dict() for name, cfg in library.waveforms.items()}
    with pytest.raises((ValueError, GuiRpcError), match="exists|collision|already"):
        invoke(
            "rpc_call",
            {
                "method": "context.ml_rename_waveform",
                "params": {"old": "seed", "new": "occupied"},
            },
        )
    assert {name: cfg.to_dict() for name, cfg in library.waveforms.items()} == before


def test_ml_get_index_and_named_cfg_are_read_only(library_client):
    invoke, library = library_client
    listed = invoke("rpc_call", {"method": "context.ml_get", "params": {}})
    seed = next(item for item in listed["waveforms"] if item["name"] == "seed")
    assert seed["style"] == "const"
    assert isinstance(seed["description"], str) and seed["description"]
    named = invoke("rpc_call", {"method": "context.ml_get", "params": {"name": "seed"}})
    assert named == {
        "name": "seed",
        "kind": "waveform",
        "cfg": library.waveforms["seed"].to_dict(),
    }
    with pytest.raises(GuiRpcError, match="missing"):
        invoke("rpc_call", {"method": "context.ml_get", "params": {"name": "missing"}})
    assert sorted(library.waveforms) == ["seed"]


def test_ml_roles_and_create_use_gui_role_defaults(library_client):
    invoke, library = library_client
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
            "params": {"role_id": "none_reset", "name": "reset_none"},
        },
    )
    assert created == {"created": "reset_none"}
    stored = invoke(
        "rpc_call", {"method": "context.ml_get", "params": {"name": "reset_none"}}
    )
    assert stored["cfg"] == library.modules["reset_none"].to_dict()


def test_ml_edit_commits_prefix_and_save_as_preserves_source(library_client):
    invoke, library = library_client
    before = library.waveforms["seed"].to_dict()
    invoke("rpc_call", {"method": "context.snapshot"})
    saved = invoke(
        "rpc_call",
        {
            "method": "context.ml_edit",
            "params": {
                "name": "seed",
                "edits": [{"path": "length", "value": 0.25}],
                "save_as": "copy",
                "kind": "waveform",
            },
        },
    )
    assert saved["valid"] is True
    assert saved["applied"] == 1
    copy_cfg = invoke(
        "rpc_call", {"method": "context.ml_get", "params": {"name": "copy"}}
    )["cfg"]
    assert copy_cfg["length"] == pytest.approx(0.25)
    assert library.waveforms["seed"].to_dict() == before
    partial = invoke(
        "rpc_call",
        {
            "method": "context.ml_edit",
            "params": {
                "name": "seed",
                "edits": [
                    {"path": "length", "value": 0.5},
                    {"path": "missing", "value": 1.0},
                    {"path": "length", "value": 0.9},
                ],
                "kind": "waveform",
            },
        },
    )
    assert partial["applied"] == 1
    assert partial["valid"] is False
    assert partial["errors"][0]["path"] == "missing"
    assert library.waveforms["seed"].to_dict()["length"] == 0.5
    assert library.waveforms["copy"].to_dict() == copy_cfg
    continued = invoke(
        "rpc_call",
        {
            "method": "context.ml_edit",
            "params": {
                "name": "seed",
                "edits": [{"path": "length", "value": 0.75}],
                "kind": "waveform",
            },
        },
    )
    assert continued["valid"] is True
    assert library.waveforms["seed"].to_dict()["length"] == 0.75


def test_ml_edit_first_failure_leaves_save_as_uncreated(library_client):
    invoke, library = library_client
    invoke("rpc_call", {"method": "context.snapshot"})
    result = invoke(
        "rpc_call",
        {
            "method": "context.ml_edit",
            "params": {
                "name": "seed",
                "save_as": "copy",
                "edits": [
                    {"path": "missing", "value": 1.0},
                    {"path": "length", "value": 0.9},
                ],
                "kind": "waveform",
            },
        },
    )
    assert result["applied"] == 0
    assert result["valid"] is False
    assert result["errors"][0]["path"] == "missing"
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
        "rpc_call",
        {
            "method": "context.ml_edit",
            "params": {
                "name": "pulse",
                "kind": "module",
                "edits": [
                    {"path": "gain", "value": {"__kind": "eval", "expr": "0.25 + 0.5"}}
                ],
            },
        },
    )
    assert result["applied"] == 1
    assert result["valid"] is True
    assert library.modules["pulse"].to_dict()["gain"] == 0.75


def test_ml_edit_requires_explicit_context_observation(library_client):
    invoke, library = library_client
    before = library.waveforms["seed"].to_dict()
    with pytest.raises(GuiRpcError) as exc:
        invoke(
            "rpc_call",
            {
                "method": "context.ml_edit",
                "params": {
                    "name": "seed",
                    "edits": [{"path": "length", "value": 0.5}],
                    "kind": "waveform",
                },
            },
        )
    assert exc.value.reason == "stale_version"
    assert library.waveforms["seed"].to_dict() == before
    invoke("rpc_call", {"method": "context.snapshot"})
    saved = invoke(
        "rpc_call",
        {
            "method": "context.ml_edit",
            "params": {
                "name": "seed",
                "edits": [{"path": "length", "value": 0.5}],
                "kind": "waveform",
            },
        },
    )
    assert saved["valid"] is True
    assert library.waveforms["seed"].to_dict()["length"] == pytest.approx(0.5)


def test_ml_edit_rejects_context_changed_by_another_connection(
    qapp, tmp_path, monkeypatch
):
    fx = Fixture(active_label="ctx001")
    library = ModuleLibrary()
    library.waveforms["seed"] = WaveformCfgFactory.from_raw(
        {"style": "const", "length": 0.1}
    )
    fx.state.set_context(replace(fx.state.session_env, md=MetaDict(), ml=library))
    fx.start()
    monkeypatch.setattr("zcu_tools.mcp.measure.tools_lifecycle.status", lambda *_: {})
    first_bridge, first = mcp_client(fx.service.port, tmp_path / "first")
    second_bridge, second = mcp_client(fx.service.port, tmp_path / "second")
    try:
        for invoke in (first, second):
            invoke("connect", {"port": fx.service.port})
            invoke("rpc_call", {"method": "context.snapshot"})
        second(
            "rpc_call",
            {
                "method": "context.ml_edit",
                "params": {
                    "name": "seed",
                    "edits": [{"path": "length", "value": 0.25}],
                    "kind": "waveform",
                },
            },
        )
        with pytest.raises(GuiRpcError) as error:
            first(
                "rpc_call",
                {
                    "method": "context.ml_edit",
                    "params": {
                        "name": "seed",
                        "edits": [{"path": "length", "value": 0.9}],
                        "kind": "waveform",
                    },
                },
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

    invoke("rpc_call", {"method": "context.snapshot"})
    with pytest.raises(GuiRpcError, match="already exists"):
        invoke(
            "rpc_call",
            {
                "method": "context.ml_edit",
                "params": {
                    "name": "seed",
                    "edits": [{"path": "length", "value": 0.5}],
                    "save_as": destination,
                    "kind": "waveform",
                },
            },
        )

    assert library.waveforms["seed"].to_dict() == original
    assert library.waveforms["occupied"].to_dict() == occupied
