"""Atomic cfg edits through the GUI socket and the MCP boundary."""

from __future__ import annotations

from threading import Event

import pytest
from zcu_tools.cfg_model import Range
from zcu_tools.gui.app.measure.adapter import Form
from zcu_tools.gui.app.measure.remote import ControlOptions
from zcu_tools.mcp.measure.session import GuiRpcError

from tests.gui.app.measure._synthetic import SampleAdapter, SampleCfg

from ._helpers import Fixture, call, mcp_client, observe_run_inputs, open_client


@pytest.fixture()
def live_tab(qapp):
    fixture = Fixture()
    tab_id = fixture.ctrl.new_tab("fake")
    fixture.start()
    yield fixture, tab_id
    fixture.stop()


@pytest.fixture()
def mcp_tab(live_tab, tmp_path, monkeypatch):
    fixture, tab_id = live_tab
    monkeypatch.setattr("zcu_tools.mcp.measure.tools_lifecycle.status", lambda *_: {})
    bridge, invoke = mcp_client(fixture.service.port, tmp_path)
    sock = open_client(fixture.service.port)
    try:
        invoke("connect", {"port": fixture.service.port})
        yield fixture, tab_id, invoke, sock
    finally:
        bridge.disconnect()
        sock.close()


def test_tab_get_mcp_preserves_complete_publication_without_focusing(mcp_tab):
    _, tab_id, invoke, sock = mcp_tab
    before = call(sock, "tab.list_all")["result"]["active_tab_id"]
    observed = invoke(
        "rpc_call", {"method": "tab.get_cfg", "params": {"tab_id": tab_id}}
    )
    assert observed["input"]["gain"] == 0.1
    assert {tuple(item["path"]): item["value"] for item in observed["values"]}[
        ("gain",)
    ] == 0.1
    assert observed["schema"]["properties"]["gain"]
    assert observed["errors"] == []
    assert observed == call(sock, "tab.get_cfg", {"tab_id": tab_id})["result"]
    assert call(sock, "tab.list_all")["result"]["active_tab_id"] == before


def test_tab_edit_mcp_writes_range_leaves_in_one_observation(mcp_tab):
    _, tab_id, invoke, sock = mcp_tab
    call(sock, "tab.get_cfg", {"tab_id": tab_id})["result"]
    result = invoke(
        "rpc_call",
        {
            "method": "tab.edit_cfg",
            "params": {
                "tab_id": tab_id,
                "edits": [
                    {"path": ["sweep", "start"], "value": 2.0},
                    {"path": ["sweep", "stop"], "value": 8.0},
                    {"path": ["sweep", "n"], "value": 4},
                ],
            },
        },
    )
    assert result["errors"] == []
    assert result["input"]["sweep"] == {
        "kind": "linrange",
        "start": 2.0,
        "stop": 8.0,
        "n": 4,
    }
    assert result == call(sock, "tab.get_cfg", {"tab_id": tab_id})["result"]


def test_reset_mcp_restores_gui_defaults(mcp_tab):
    _, tab_id, invoke, sock = mcp_tab
    defaults = call(sock, "tab.get_cfg", {"tab_id": tab_id})["result"]
    invoke(
        "rpc_call",
        {
            "method": "tab.edit_cfg",
            "params": {
                "tab_id": tab_id,
                "edits": [{"path": ["gain"], "value": 0.75}],
            },
        },
    )

    reset = invoke(
        "rpc_call",
        {
            "method": "tab.reset_cfg",
            "params": {
                "tab_id": tab_id,
            },
        },
    )
    assert reset == defaults
    assert call(sock, "tab.get_cfg", {"tab_id": tab_id})["result"] == reset

    unchanged = invoke(
        "rpc_call",
        {
            "method": "tab.reset_cfg",
            "params": {
                "tab_id": tab_id,
            },
        },
    )
    assert unchanged == defaults


def test_reset_during_active_run_preserves_the_publication(live_tab, monkeypatch):
    fixture, tab_id = live_tab
    release = Event()
    original_run = SampleAdapter.run

    def blocked_run(self, req, raw_cfg, *, context):
        if not release.wait(timeout=5):
            raise TimeoutError("test did not release the fake run")
        return original_run(self, req, raw_cfg, context=context)

    monkeypatch.setattr(SampleAdapter, "run", blocked_run)
    sock = open_client(fixture.service.port)
    try:
        observe_run_inputs(
            fixture, tab_id, lambda method, params: call(sock, method, params)["result"]
        )
        before = call(sock, "tab.get_cfg", {"tab_id": tab_id})["result"]
        params = {
            "tab_id": tab_id,
        }
        assert call(sock, "tab.run_start", params)["ok"]
        rejected = call(sock, "tab.reset_cfg", params)
        assert rejected["error"]["reason"] == "mutation_blocked"
        assert call(sock, "tab.get_cfg", {"tab_id": tab_id})["result"] == before
    finally:
        release.set()
        fixture.ctrl.quiesce()
        sock.close()


def test_tab_edit_mcp_rejection_preserves_entire_previous_publication(mcp_tab):
    _, tab_id, invoke, sock = mcp_tab
    before = call(sock, "tab.get_cfg", {"tab_id": tab_id})["result"]
    with pytest.raises(GuiRpcError) as raised:
        invoke(
            "rpc_call",
            {
                "method": "tab.edit_cfg",
                "params": {
                    "tab_id": tab_id,
                    "edits": [
                        {"path": ["gain"], "value": 0.25},
                        {"path": ["missing"], "value": 1},
                    ],
                },
            },
        )
    assert raised.value.code == "invalid_params"
    assert raised.value.reason == "form_errors"
    assert call(sock, "tab.get_cfg", {"tab_id": tab_id})["result"] == before


def test_cfg_edit_requires_authentication(qapp):
    fixture = Fixture(ControlOptions(port=0, token="cfg-auth-test"))
    tab_id = fixture.ctrl.new_tab("fake")
    fixture.start()
    authorized = open_client(fixture.service.port)
    unauthorized = open_client(fixture.service.port)
    try:
        assert call(authorized, "auth", {"token": "cfg-auth-test"})["ok"]
        before = call(authorized, "tab.get_cfg", {"tab_id": tab_id})["result"]
        rejected = call(
            unauthorized,
            "tab.edit_cfg",
            {
                "tab_id": tab_id,
                "edits": [{"path": ["gain"], "value": 0.25}],
            },
        )
        assert rejected["error"]["code"] == "unauthorized"
        assert call(authorized, "tab.get_cfg", {"tab_id": tab_id})["result"] == before
    finally:
        authorized.close()
        unauthorized.close()
        fixture.stop()


def test_invalid_publication_is_success_but_cannot_run(mcp_tab):
    fixture, tab_id, invoke, sock = mcp_tab
    call(sock, "tab.get_cfg", {"tab_id": tab_id})["result"]
    invalid = invoke(
        "rpc_call",
        {
            "method": "tab.edit_cfg",
            "params": {
                "tab_id": tab_id,
                "edits": [{"path": ["gain"], "value": {"expr": "1 / 0"}}],
            },
        },
    )
    assert invalid["input"]["gain"] == {"expr": "1 / 0"}
    assert invalid["errors"][0]["path"] == ["gain"]
    assert (
        invoke("rpc_call", {"method": "tab.get_cfg", "params": {"tab_id": tab_id}})
        == invalid
    )
    observe_run_inputs(
        fixture, tab_id, lambda name, params: call(sock, name, params)["result"]
    )
    rejected = call(
        sock,
        "tab.run_start",
        {
            "tab_id": tab_id,
        },
    )
    assert rejected["error"]["reason"] == "not_valid"
    assert fixture.ctrl.active_operation_count() == 0


class TextCfg(SampleCfg):
    text: str = "before"


class TextAdapter(SampleAdapter):
    form = Form(TextCfg, {"sweep": Range.between(1.0, 2.0, expts=3)})


def test_reply_encoding_failure_does_not_roll_back_the_cfg_publication(qapp):
    fixture = Fixture()
    fixture.registry.register("text", TextAdapter)
    tab_id = fixture.ctrl.new_tab("text")
    owner = fixture.ctrl.cfg_resources.lookup(tab_id)
    fixture.start()
    sock = open_client(fixture.service.port)
    try:
        call(
            sock,
            "tab.get_cfg",
            {
                "tab_id": tab_id,
            },
        )["result"]
        text = "x" * (4 * 1024 * 1024)
        # The read helper switches to nonblocking; a large request needs sendall
        # to wait for the server's reader thread before pumping owner events.
        sock.settimeout(10.0)
        rejected_reply = call(
            sock,
            "tab.edit_cfg",
            {
                "tab_id": tab_id,
                "edits": [{"path": ["text"], "value": text}],
            },
            timeout_s=10.0,
        )
        assert rejected_reply["error"]["reason"] == "response_encoding_failed"
        assert owner.input.data["text"] == text
        assert owner.accept().values["text"] == text

        owner.write({("text",): "small"})
    finally:
        sock.close()
        fixture.stop()


def test_edit_error_identifies_failed_path_and_index_without_committing(live_tab):
    fixture, tab_id = live_tab
    sock = open_client(fixture.service.port)
    try:
        before = call(sock, "tab.get_cfg", {"tab_id": tab_id})["result"]
        reply = call(
            sock,
            "tab.edit_cfg",
            {
                "tab_id": tab_id,
                "edits": [
                    {"path": ["gain"], "value": 0.25},
                    {"path": ["missing"], "value": 1},
                ],
            },
        )
        assert reply["error"]["code"] == "invalid_params"
        assert reply["error"]["reason"] == "form_errors"
        assert call(sock, "tab.get_cfg", {"tab_id": tab_id})["result"] == before
    finally:
        sock.close()


def test_library_editor_batch_retains_its_independent_prefix_contract(live_tab):
    fixture, _ = live_tab
    schema = SampleAdapter().make_default_cfg(fixture.state.session_env)
    parameters = fixture.ctrl.read_context()
    editor_id, _ = fixture.ctrl.open_seeded_cfg_editor(
        schema, parameters=parameters, gc=False
    )
    sock = open_client(fixture.service.port)
    try:
        result = call(
            sock,
            "editor.set_fields",
            {
                "editor_id": editor_id,
                "edits": [
                    {
                        "path": "sweep",
                        "value": {"start": 2.0, "stop": 8.0, "step": 2.2},
                    },
                ],
            },
        )
        assert result["ok"] is True
        assert result["result"]["actual"]["sweep"]["expts"] == 4
        rejected = call(
            sock,
            "editor.set_fields",
            {
                "editor_id": editor_id,
                "edits": [
                    {"path": "gain", "value": 0.25},
                    {"path": "sweep", "value": {"start": 3.0, "stop": 9.0}},
                ],
            },
        )
        assert rejected["error"]["reason"] == "invalid_settable_path"
        tree = call(sock, "editor.get", {"editor_id": editor_id})["result"]["tree"]
        assert tree["children"]["gain"]["input"]["resolved"] == pytest.approx(0.25)
        assert tree["children"]["sweep"]["inputs"]["expts"]["resolved"] == 4
    finally:
        sock.close()


@pytest.mark.parametrize("failure", ["unavailable", "provider_fault"])
def test_independent_editor_batch_preserves_precondition_and_prior_prefix(
    live_tab, monkeypatch, failure
):
    from zcu_tools.gui.session.value_lookup import ProviderError, UnavailableValue

    fixture, _ = live_tab
    schema = SampleAdapter().make_default_cfg(fixture.state.session_env)
    parameters = fixture.ctrl.read_context()
    editor_id, _ = fixture.ctrl.open_seeded_cfg_editor(
        schema, parameters=parameters, gc=False
    )

    def reject(*_args):
        if failure == "unavailable":
            raise UnavailableValue("device.bias.value", "not ready")
        raise ProviderError(
            "device.bias.value", "device:bias", ValueError("driver fault")
        )

    monkeypatch.setattr(fixture.ctrl, "read_value_source", reject)
    sock = open_client(fixture.service.port)
    try:
        reply = call(
            sock,
            "editor.set_fields",
            {
                "editor_id": editor_id,
                "edits": [
                    {"path": "gain", "value": 0.25},
                    {
                        "path": "gain",
                        "value": {
                            "__kind": "value_ref",
                            "key": "device.bias.value",
                            "type": "float",
                        },
                    },
                ],
            },
        )
        assert reply["error"]["code"] == (
            "precondition_failed" if failure == "unavailable" else "controller_error"
        )
        tree = call(sock, "editor.get", {"editor_id": editor_id})["result"]["tree"]
        assert tree["children"]["gain"]["input"]["resolved"] == 0.25
    finally:
        sock.close()


def test_holder_edits_resets_and_runs_current_cfg_without_ref(live_tab):
    fixture, tab_id = live_tab
    sock = open_client(fixture.service.port)
    try:
        for method, params in (
            ("tab.snapshot", {"tab_id": tab_id}),
            ("soc.info", {"include_cfg": True}),
        ):
            observed = call(sock, method, params)
            assert observed["ok"], observed
        for device in call(sock, "device.list")["result"]["devices"]:
            observed = call(sock, "device.snapshot", {"name": device["name"]})
            assert observed["ok"], observed
        edited = call(
            sock,
            "tab.edit_cfg",
            {"tab_id": tab_id, "edits": [{"path": ["gain"], "value": 0.75}]},
        )
        assert edited["ok"], edited
        assert edited["result"]["input"]["gain"] == 0.75
        reset = call(sock, "tab.reset_cfg", {"tab_id": tab_id})
        assert reset["ok"], reset
        assert reset["result"]["input"]["gain"] != 0.75
        started = call(sock, "tab.run_start", {"tab_id": tab_id})
        assert started["ok"], started
        assert started["result"]["operation_id"]
        call(
            sock,
            "operation.await",
            {
                "operation_id": started["result"]["operation_id"],
                "timeout": 3.0,
            },
        )
    finally:
        sock.close()
