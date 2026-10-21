"""ModuleLibrary tools through the public MCP handler and live GUI socket."""

from __future__ import annotations

import pytest
from zcu_tools.cfg_model import (
    ConstWaveformCfg,
    MuxReadoutSpec,
    MuxReadoutTone,
    TemplateDef,
)
from zcu_tools.gui.session.context_types import LibraryChange
from zcu_tools.mcp.measure.session import GuiRpcError

from tests.gui.app.measure._templates import TEMPLATES
from tests.resources.entry.fakes import register_fakes, registry_state, restore_registry

from ._helpers import Fixture, call, mcp_client, open_client


@pytest.fixture()
def library_client(qapp, tmp_path):
    fx = Fixture(active_label="ctx001", templates=TEMPLATES)
    context = fx.ctrl.context_control
    context.set_waveform(
        context.read_context().target, "Q1.waveforms.seed", ConstWaveformCfg(length=0.1)
    )
    fx.start()
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


MUX_TONES = [{"freq": 100.0, "gain": 0.2, "ro_ch": 0}]


@pytest.fixture()
def component_registry():
    before = registry_state()
    try:
        register_fakes()
        installed = registry_state()
        yield
        assert registry_state() == installed
    finally:
        restore_registry(before)


@pytest.fixture()
def role_socket(qapp, tmp_path, component_registry):
    template = TemplateDef(
        MuxReadoutSpec,
        "Synthetic mux readout",
        {"gen_ch": 0, "nqz": 1, "mixer_freq": 0.0, "length": 1.0, "ro_length": 1.0},
    )
    fx = Fixture(project_root=str(tmp_path), templates={"mux:blank": template})
    fx.ctrl.context_control.add_component("R1", "fake/drive/a", {})
    fx.start()
    sock = open_client(fx.service.port)
    try:
        yield fx, sock
    finally:
        sock.close()
        fx.stop()


def test_ml_create_from_role_applies_edits_before_validation(role_socket):
    fx, sock = role_socket
    reply = call(
        sock,
        "context.ml_create_from_role",
        {
            "role_id": "mux:blank",
            "name": "R1.readout.mux_readout",
            "edits": [
                {"path": ["tones"], "value": MUX_TONES},
                {"path": ["length"], "value": {"expr": "0.25 + 0.5"}},
            ],
        },
    )
    assert reply["ok"] is True, reply
    assert reply["result"] == {"created": "R1.readout.mux_readout"}
    written = fx.ctrl.read_context().library.module_spec("R1.readout.mux_readout")
    assert written.model_dump(mode="json")["tones"] == [
        MuxReadoutTone.model_validate(tone).model_dump(mode="json")
        for tone in MUX_TONES
    ]
    assert written.model_dump(mode="json")["length"] == 0.75


@pytest.mark.parametrize(
    ("edits", "error_path"),
    [
        (None, ["tones"]),
        (
            [
                {"path": ["tones"], "value": MUX_TONES},
                {"path": ["missing"], "value": 1},
            ],
            ["missing"],
        ),
        (
            [
                {"path": ["tones"], "value": MUX_TONES},
                {"path": ["length"], "value": -1.0},
            ],
            ["length"],
        ),
        ([{"path": ["tones"], "value": []}], ["tones"]),
        ([{"path": ["tones"], "value": {"freq": 100.0}}], ["tones"]),
    ],
    ids=[
        "required-list-missing",
        "unknown-path",
        "invalid-value",
        "invalid-form",
        "invalid-structure",
    ],
)
def test_ml_create_from_role_rejects_invalid_input_without_writing(
    role_socket, edits, error_path
):
    fx, sock = role_socket
    before = fx.ctrl.read_context().library.document()
    params = {"role_id": "mux:blank", "name": "R1.readout.mux_readout"}
    if edits is not None:
        params["edits"] = edits
    reply = call(sock, "context.ml_create_from_role", params)
    assert reply["ok"] is False, reply
    error = reply["error"]
    assert error["code"] == "invalid_params"
    assert error.get("data") is not None, error
    assert [item["path"] for item in error["data"]["errors"]] == [error_path]
    assert all(item["message"] for item in error["data"]["errors"])
    assert fx.ctrl.read_context().library.document() == before


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
