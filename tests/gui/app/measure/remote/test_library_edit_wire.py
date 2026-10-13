"""context.ml_edit writes library entries through the shared Form over the socket."""

from pathlib import Path
from time import monotonic
from typing import Any

import pytest
from zcu_tools.cfg_model import ConstWaveformCfg, PulseSpec

from tests.resources.entry.fakes import register_fakes, registry_state, restore_registry

from ._helpers import Fixture, call, open_client

PULSE = "Q1.pulses.pi"
WAVEFORM = "Q1.waveforms.seed"


@pytest.fixture()
def library_socket(qapp, tmp_path: Path):
    before = registry_state()
    fx = None
    sock = None
    try:
        register_fakes()
        installed = registry_state()
        fx = Fixture(active_label="ctx001", project_root=str(tmp_path))
        control = fx.ctrl.context_control
        control.add_component("Q1", "fake/drive/a", {})
        parameters = control.read_context()
        control.set_waveform(parameters.target, WAVEFORM, ConstWaveformCfg(length=0.1))
        control.set_module(
            parameters.target,
            PULSE,
            PulseSpec(ch=0, nqz=1, freq=100.0, gain=0.2, waveform=WAVEFORM),
        )
        fx.start()
        sock = open_client(fx.service.port)
        yield fx, sock
        assert registry_state() == installed
    finally:
        try:
            if sock is not None:
                sock.close()
            if fx is not None:
                closed = False

                def on_closed() -> None:
                    nonlocal closed
                    closed = True

                fx.ctrl.begin_shutdown(on_closed)
                deadline = monotonic() + 1.0
                while not closed and monotonic() < deadline:
                    qapp.processEvents()
                if not closed:
                    raise RuntimeError("Synchronous library fixture shutdown timed out")
                fx.stop()
                qapp.processEvents()
        finally:
            restore_registry(before)


def ml_edit(
    sock, kind: str, name: str, edits: list[dict[str, object]], **extra
) -> dict[str, Any]:
    params = {"kind": kind, "name": name, "edits": edits, **extra}
    return call(sock, "context.ml_edit", params)


def test_ml_edit_writes_path_array_edits_and_evaluated_expressions(
    library_socket,
) -> None:
    fx, sock = library_socket
    reply = ml_edit(
        sock,
        "module",
        PULSE,
        [
            {"path": ["gain"], "value": {"expr": "0.25 + 0.5"}},
            {"path": ["freq"], "value": 120.0},
        ],
    )
    assert reply["ok"] is True, reply
    assert reply["result"] == {"valid": True, "errors": []}
    written = fx.ctrl.read_context().library.module_spec(PULSE)
    assert isinstance(written, PulseSpec)
    assert (written.gain, written.freq, written.waveform) == (0.75, 120.0, WAVEFORM)


def test_ml_edit_invalid_field_leaves_library_unchanged_and_names_its_path(
    library_socket,
) -> None:
    fx, sock = library_socket
    before = fx.ctrl.read_context().library.document()
    reply = ml_edit(
        sock,
        "module",
        PULSE,
        [
            {"path": ["freq"], "value": 130.0},
            {"path": ["gain"], "value": "bad"},
        ],
    )
    assert reply["ok"] is True, reply
    assert reply["result"]["valid"] is False
    assert [error["path"] for error in reply["result"]["errors"]] == [["gain"]]
    assert all(error["message"] for error in reply["result"]["errors"])
    assert fx.ctrl.read_context().library.document() == before


def test_ml_edit_unknown_path_is_rejected_without_a_partial_write(
    library_socket,
) -> None:
    fx, sock = library_socket
    before = fx.ctrl.read_context().library.document()
    reply = ml_edit(
        sock,
        "waveform",
        WAVEFORM,
        [
            {"path": ["length"], "value": 0.5},
            {"path": ["missing"], "value": 1.0},
        ],
    )
    assert reply["ok"] is False
    assert reply["error"]["code"] == "invalid_params"
    assert reply["error"]["reason"] == "form_errors"
    assert "missing" in reply["error"]["message"]
    assert fx.ctrl.read_context().library.document() == before


def test_ml_edit_save_as_creates_a_new_entry_and_keeps_the_source(
    library_socket,
) -> None:
    fx, sock = library_socket
    reply = ml_edit(
        sock,
        "waveform",
        WAVEFORM,
        [{"path": ["length"], "value": 0.25}],
        save_as="Q1.waveforms.copy",
    )
    assert reply["ok"] is True, reply
    assert reply["result"] == {"valid": True, "errors": []}
    library = fx.ctrl.read_context().library
    assert library.waveform_spec("Q1.waveforms.copy").length == 0.25
    assert library.waveform_spec(WAVEFORM).length == 0.1


def test_ml_edit_save_as_rejects_an_existing_name(library_socket) -> None:
    fx, sock = library_socket
    before = fx.ctrl.read_context().library.document()
    reply = ml_edit(
        sock,
        "waveform",
        WAVEFORM,
        [{"path": ["length"], "value": 0.5}],
        save_as=WAVEFORM,
    )
    assert reply["ok"] is False
    assert reply["error"]["code"] == "invalid_params"
    assert "already exists" in reply["error"]["message"]
    assert fx.ctrl.read_context().library.document() == before


@pytest.mark.parametrize(
    "value",
    [
        {"__kind": "eval", "expr": "0.25 + 0.5"},
        {"__kind": "value_ref", "key": "device.flux.value", "type": "float"},
    ],
    ids=["eval", "value_ref"],
)
def test_ml_edit_rejects_retired_value_tags_with_the_expression_form(
    library_socket, value
) -> None:
    fx, sock = library_socket
    before = fx.ctrl.read_context().library.document()
    reply = ml_edit(sock, "module", PULSE, [{"path": ["gain"], "value": value}])
    assert reply["ok"] is False
    assert reply["error"]["code"] == "invalid_params"
    assert '{"expr": ' in reply["error"]["message"]
    assert fx.ctrl.read_context().library.document() == before
