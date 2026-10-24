"""ModuleLibrary tools through the public MCP handler and live GUI socket."""

from __future__ import annotations

import pytest
from pydantic import Field
from zcu_tools.cfg_model import (
    ConstWaveformCfg,
    Expr,
    MuxReadoutSpec,
    MuxReadoutTone,
    PulseSpec,
    TemplateDef,
)
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_types import LibraryChange
from zcu_tools.mcp.measure.session import GuiRpcError
from zcu_tools.resources.entry import ComponentSchema

from tests.gui.app.measure._templates import TEMPLATES
from tests.resources.entry.fakes import register_fakes, registry_state, restore_registry

from ._helpers import Fixture, call, mcp_client, open_client


@pytest.fixture()
def library_client(qapp, tmp_path):
    fx = Fixture(
        active_label="ctx001",
        templates=TEMPLATES,
        project_root=str(tmp_path / "project"),
    )
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
    library = fx.ctrl.context_control.read_context().library
    return {
        key: library.waveform_spec(key).model_dump(mode="python")
        for key in library.waveform_keys()
    }


def modules(fx):
    library = fx.ctrl.context_control.read_context().library
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
            "params": {"role_id": "reset", "name": "Q1.reset.seed"},
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
    fx.ctrl.context_control.apply_library_change(
        fx.ctrl.context_control.read_context(),
        LibraryChange(kind, "rename", seed, new_path=moved),
    )
    assert seed not in selected(fx)
    assert selected(fx)[moved] == before
    assert other(fx)[other_seed] == other_before
    fx.ctrl.context_control.apply_library_change(
        fx.ctrl.context_control.read_context(), LibraryChange(kind, "delete", moved)
    )
    assert moved not in selected(fx)
    assert other(fx)[other_seed] == other_before


def test_ml_rename_rejects_collision_without_overwriting(library_client):
    invoke, fx = library_client
    fx.ctrl.context_control.set_waveform(
        fx.ctrl.context_control.read_context().target,
        "Q1.waveforms.occupied",
        ConstWaveformCfg(length=0.3),
    )
    invoke("rpc_call", {"method": "context.snapshot"})
    before = waveforms(fx)
    with pytest.raises(ValueError, match="exists|collision|already"):
        fx.ctrl.context_control.apply_library_change(
            fx.ctrl.context_control.read_context(),
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


class SlottedComponent(ComponentSchema):
    module: dict[str, str | None] = Field(default_factory=dict)
    rate: float


@pytest.fixture()
def component_registry():
    before = registry_state()
    try:
        register_fakes()
        from zcu_tools.resources.entry import component_registry as registry

        registry.register("synthetic/slotted", SlottedComponent)
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
    probe = TemplateDef(
        PulseSpec,
        "Synthetic alias probe",
        {
            "ch": 2,
            "nqz": 1,
            "freq": Expr("{qubit}.rate"),
            "gain": 0.2,
            "waveform.style": "const",
            "waveform.length": 0.5,
        },
    )
    fx = Fixture(
        project_root=str(tmp_path),
        templates={"mux:blank": template, "probe": probe},
    )
    fx.ctrl.context_control.add_component("R1", "synthetic/slotted", {"rate": 234.0})
    fx.start()
    sock = open_client(fx.service.port)
    try:
        yield fx, sock
    finally:
        sock.close()
        fx.stop()


def test_ml_create_binds_placeholders_to_resolved_alias_not_entry_name(role_socket):
    fx, sock = role_socket
    fx.ctrl.context_control.add_component("Q2", "synthetic/slotted", {"rate": 567.0})
    fx.ctrl.context_control.set_alias_rows((AliasRow("qubit", "R1"),))
    reply = call(
        sock,
        "context.ml_create_from_role",
        {"role_id": "probe", "name": "Q2.pulses.probe"},
    )
    assert reply["ok"] is True, reply
    written = fx.ctrl.context_control.read_context().library.module_spec(
        "Q2.pulses.probe"
    )
    assert written.model_dump(mode="json")["freq"] == 234.0


@pytest.mark.parametrize("alias_value", [None, ""])
def test_ml_create_missing_alias_reports_binding_without_writing(
    role_socket, alias_value
):
    fx, sock = role_socket
    if alias_value is not None:
        fx.ctrl.context_control.set_alias_rows((AliasRow("qubit", alias_value),))
    before = fx.ctrl.context_control.read_context().library.document()
    reply = call(
        sock,
        "context.ml_create_from_role",
        {"role_id": "probe", "name": "R1.pulses.probe"},
    )
    assert reply["ok"] is False, reply
    assert reply["error"]["code"] == "invalid_params"
    assert "qubit" in reply["error"]["message"]
    assert "Aliases" in reply["error"]["message"]
    assert fx.ctrl.context_control.read_context().library.document() == before


@pytest.mark.parametrize("whole_module", [False, True])
def test_parameter_set_binds_missing_module_slot_and_returns_previous_value(
    role_socket, whole_module
):
    fx, sock = role_socket
    key = "R1.readout.mux_readout"
    fx.ctrl.create_from_template(
        "module", "mux:blank", key, edits={("tones",): MUX_TONES}
    )
    path = "R1.module" if whole_module else "R1.module.readout"
    value = {"readout": key} if whole_module else key
    reply = call(sock, "context.set_parameter", {"path": path, "value": value})
    assert reply["ok"] is True, reply
    assert reply["result"] == {
        "path": path,
        "before": {} if whole_module else None,
        "value": value,
    }
    assert (
        fx.ctrl.context_control.read_context().library_key("R1.module.readout") == key
    )
    replaced = call(sock, "context.set_parameter", {"path": path, "value": value})
    assert replaced["ok"] is True, replaced
    assert replaced["result"]["before"] == value


@pytest.mark.parametrize("name", ["mux_ro", "R1.mux_ro", "R1.invalid.mux_ro"])
def test_ml_create_reports_name_format_without_writing(role_socket, name):
    fx, sock = role_socket
    before = fx.ctrl.context_control.read_context().library.document()
    reply = call(
        sock,
        "context.ml_create_from_role",
        {
            "role_id": "mux:blank",
            "name": name,
            "edits": [{"path": ["tones"], "value": MUX_TONES}],
        },
    )
    assert reply["ok"] is False, reply
    error = reply["error"]
    assert error["code"] == "invalid_params"
    assert "component.partition.name" in error["message"]
    assert all(
        partition in error["message"] for partition in ("pulses", "readout", "reset")
    )
    assert fx.ctrl.context_control.read_context().library.document() == before


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
    written = fx.ctrl.context_control.read_context().library.module_spec(
        "R1.readout.mux_readout"
    )
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
    before = fx.ctrl.context_control.read_context().library.document()
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
    assert fx.ctrl.context_control.read_context().library.document() == before


def test_ml_roles_and_create_use_gui_role_defaults(library_client):
    invoke, fx = library_client
    roles = invoke("rpc_call", {"method": "context.ml_list_roles", "params": {}})[
        "roles"
    ]
    role = next(item for item in roles if item["role_id"] == "reset")
    assert role["item_kind"] == "module"
    assert role["default_name"] == "reset"
    created = invoke(
        "rpc_call",
        {
            "method": "context.ml_create_from_role",
            "params": {"role_id": "reset", "name": "Q1.reset.reset"},
        },
    )
    assert created == {"created": "Q1.reset.reset"}
    stored = invoke(
        "rpc_call",
        {
            "method": "context.snapshot",
            "params": {"path": "library.components.Q1.reset.reset"},
        },
    )
    assert stored["value"] == modules(fx)["Q1.reset.reset"]
