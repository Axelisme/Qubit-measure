"""Atomic cfg edits through the GUI socket and the MCP boundary."""

from __future__ import annotations

import pytest
from zcu_tools.experiment.v2_gui.measure.adapters.fake import FakeAdapter
from zcu_tools.gui.app.measure.cfg_binding import MeasureCfgBindings
from zcu_tools.gui.app.measure.remote import ControlOptions
from zcu_tools.gui.cfg import (
    CenteredSweepSpec,
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    ReferenceSpec,
    ScalarSpec,
    make_default_value,
)
from zcu_tools.gui.cfg.edit_codec import encode_ref
from zcu_tools.gui.cfg.resource import CfgEdit, CfgStaleError
from zcu_tools.mcp.measure.session import GuiRpcError

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


def test_tab_get_mcp_preserves_complete_publication_without_focusing(
    qapp, tmp_path, monkeypatch
):
    fixture = Fixture()
    spec = CfgSectionSpec(
        fields={
            "axis": CenteredSweepSpec(center_editable=False, locked_center=0.0),
            "nqz": ScalarSpec("Nyquist zone", int, choices=[1, 2]),
            "drive": ReferenceSpec(
                "module",
                [
                    CfgSectionSpec(
                        label="Pulse", fields={"gain": ScalarSpec("Gain", float)}
                    )
                ],
                optional=True,
            ),
        }
    )
    tab_id = "tab-agent-cfg-read"
    fixture.prepare_tab(
        tab_id, FakeAdapter(), CfgSchema(spec, make_default_value(spec))
    )
    fixture.start()
    monkeypatch.setattr("zcu_tools.mcp.measure.tools_lifecycle.status", lambda *_: {})
    bridge, invoke = mcp_client(fixture.service.port, tmp_path)
    sock = open_client(fixture.service.port)
    try:
        invoke("connect", {"port": fixture.service.port})
        before = call(sock, "tab.list_all")
        result = invoke("tab_get", {"tab": tab_id, "include": ["cfg"]})
        children = result["cfg"]["tree"]["children"]
        assert children["axis"]["kind"] == "centered_sweep"
        assert children["axis"]["locked_center"] == 0.0
        assert children["axis"]["center_editable"] is False
        assert children["nqz"]["type"] == "int"
        assert children["nqz"]["choices"] == [1, 2]
        assert children["drive"]["kind"] == "reference"
        assert children["drive"]["choices"] == ["Pulse"]
        assert result["cfg"] == call(sock, "tab.get_cfg", {"tab_id": tab_id})["result"]
        assert (
            call(sock, "tab.list_all")["result"]["active_tab_id"]
            == before["result"]["active_tab_id"]
        )
    finally:
        bridge.disconnect()
        sock.close()
        fixture.stop()


def test_tab_edit_mcp_normalizes_sweep_in_one_publication(mcp_tab):
    _, tab_id, invoke, sock = mcp_tab
    before = call(sock, "tab.get_cfg", {"tab_id": tab_id})["result"]
    result = invoke(
        "tab_edit",
        {
            "tab": tab_id,
            "expected": before["cfg_ref"],
            "edits": [
                {"path": ["sweep"], "value": {"start": 2.0, "stop": 8.0, "step": 2.2}},
            ],
        },
    )
    assert result["status"] == "Valid"
    assert result["cfg_ref"]["cfg_id"] == before["cfg_ref"]["cfg_id"]
    assert int(result["cfg_ref"]["revision"]) == int(before["cfg_ref"]["revision"]) + 1
    inputs = result["tree"]["children"]["sweep"]["inputs"]
    assert inputs["expts"]["resolved"] == 4
    assert inputs["step"]["resolved"] == pytest.approx(2.0)
    assert result == call(sock, "tab.get_cfg", {"tab_id": tab_id})["result"]


def test_tab_edit_mcp_rejection_preserves_entire_previous_publication(mcp_tab):
    _, tab_id, invoke, sock = mcp_tab
    before = call(sock, "tab.get_cfg", {"tab_id": tab_id})["result"]
    with pytest.raises(GuiRpcError) as raised:
        invoke(
            "tab_edit",
            {
                "tab": tab_id,
                "expected": before["cfg_ref"],
                "edits": [
                    {"path": ["gain"], "value": 0.25},
                    {
                        "path": ["sweep"],
                        "value": {"start": 3.0, "stop": 9.0, "step": 2.0, "expts": 4},
                    },
                ],
            },
        )
    assert raised.value.code == "invalid_params"
    assert raised.value.reason == "invalid_value"
    assert call(sock, "tab.get_cfg", {"tab_id": tab_id})["result"] == before


def test_stale_reference_returns_expected_actual_and_never_retries(mcp_tab):
    _, tab_id, invoke, sock = mcp_tab
    expected = call(sock, "tab.get_cfg", {"tab_id": tab_id})["result"]["cfg_ref"]
    actual = invoke(
        "tab_edit",
        {
            "tab": tab_id,
            "expected": expected,
            "edits": [{"path": ["gain"], "value": 0.25}],
        },
    )
    rejected = call(
        sock,
        "tab.edit_cfg",
        {
            "tab_id": tab_id,
            "expected": expected,
            "edits": [{"path": ["gain"], "value": 0.5}],
        },
    )
    assert rejected["error"]["code"] == "precondition_failed"
    assert rejected["error"]["reason"] == "stale_revision"
    assert rejected["error"]["data"] == {
        "expected": expected,
        "actual": actual["cfg_ref"],
    }
    assert call(sock, "tab.get_cfg", {"tab_id": tab_id})["result"] == actual


@pytest.mark.parametrize(
    "remote_first", [False, True], ids=["gui-first", "remote-first"]
)
def test_gui_and_remote_competing_edits_preserve_the_first_publication(
    mcp_tab, remote_first
):
    fixture, tab_id, _, sock = mcp_tab
    owner = fixture.ctrl.cfg_resources.lookup(tab_id)
    before = call(sock, "tab.get_cfg", {"tab_id": tab_id})["result"]
    base = owner.observe().ref
    params = {
        "tab_id": tab_id,
        "expected": before["cfg_ref"],
        "edits": [{"path": ["gain"], "value": 0.25}],
    }
    if remote_first:
        winner = call(sock, "tab.edit_cfg", params)["result"]
        with pytest.raises(CfgStaleError) as raised:
            owner.edit(base.revision, (CfgEdit(("gain",), DirectValue(0.5)),))
        assert encode_ref(raised.value.expected) == before["cfg_ref"]
        assert encode_ref(raised.value.actual) == winner["cfg_ref"]
    else:
        owner.edit(base.revision, (CfgEdit(("gain",), DirectValue(0.5)),))
        winner = call(sock, "tab.get_cfg", {"tab_id": tab_id})["result"]
        rejected = call(sock, "tab.edit_cfg", params)
        assert rejected["error"]["reason"] == "stale_revision"
        assert rejected["error"]["data"] == {
            "expected": before["cfg_ref"],
            "actual": winner["cfg_ref"],
        }
    assert call(sock, "tab.get_cfg", {"tab_id": tab_id})["result"] == winner
    assert winner["tree"]["children"]["gain"]["input"]["resolved"] == (
        0.25 if remote_first else 0.5
    )


@pytest.mark.parametrize("method", ["tab.edit_cfg", "tab.run_start"])
def test_recreated_tab_rejects_the_retired_cfg_identity(live_tab, method):
    fixture, tab_id = live_tab
    sock = open_client(fixture.service.port)
    try:
        old = call(sock, "tab.get_cfg", {"tab_id": tab_id})["result"]
        fixture.ctrl.close_tab(tab_id)
        replacement = fixture.prepare_tab(
            tab_id,
            FakeAdapter(),
            FakeAdapter().make_default_cfg(fixture.state.session_env),
        )
        current = call(sock, "tab.get_cfg", {"tab_id": tab_id})["result"]
        assert current["cfg_ref"]["cfg_id"] != old["cfg_ref"]["cfg_id"]
        if method == "tab.run_start":
            observe_run_inputs(
                fixture, tab_id, lambda name, params: call(sock, name, params)["result"]
            )
        params = {"tab_id": tab_id, "expected": old["cfg_ref"]}
        if method == "tab.edit_cfg":
            params["edits"] = [{"path": ["gain"], "value": 0.25}]
        rejected = call(sock, method, params)
        assert rejected["error"]["reason"] == "stale_revision"
        assert rejected["error"]["data"] == {
            "expected": old["cfg_ref"],
            "actual": current["cfg_ref"],
        }
        assert encode_ref(replacement.observe().ref) == current["cfg_ref"]
        assert call(sock, "tab.get_cfg", {"tab_id": tab_id})["result"] == current
    finally:
        sock.close()


@pytest.mark.parametrize("method", ["tab.edit_cfg", "tab.run_start"])
@pytest.mark.parametrize("missing", ["expected", "revision", "canonical-revision"])
def test_mutations_require_an_explicit_canonical_cfg_reference(
    live_tab, method, missing
):
    fixture, tab_id = live_tab
    sock = open_client(fixture.service.port)
    try:
        before = call(sock, "tab.get_cfg", {"tab_id": tab_id})["result"]
        if method == "tab.run_start":
            observe_run_inputs(
                fixture, tab_id, lambda name, params: call(sock, name, params)["result"]
            )
        params = {"tab_id": tab_id}
        if missing != "expected":
            params["expected"] = {"cfg_id": before["cfg_ref"]["cfg_id"]}
            if missing == "canonical-revision":
                params["expected"]["revision"] = "01"
        if method == "tab.edit_cfg":
            params["edits"] = [{"path": ["gain"], "value": 0.25}]
        rejected = call(sock, method, params)
        assert rejected["error"]["code"] == "invalid_params"
        assert call(sock, "tab.get_cfg", {"tab_id": tab_id})["result"] == before
        assert fixture.ctrl.active_operation_count() == 0
    finally:
        sock.close()


def test_cfg_reference_is_not_an_authentication_token(qapp):
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
                "expected": before["cfg_ref"],
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
    before = call(sock, "tab.get_cfg", {"tab_id": tab_id})["result"]
    invalid = invoke(
        "tab_edit",
        {
            "tab": tab_id,
            "expected": before["cfg_ref"],
            "edits": [{"path": ["gain"], "value": {"__text": "-"}}],
        },
    )
    assert invalid["status"] == "Invalid"
    assert invalid["cfg_ref"] != before["cfg_ref"]
    assert invalid["tree"]["children"]["gain"]["input"]["raw"] == "-"
    assert invoke("tab_get", {"tab": tab_id, "include": ["cfg"]})["cfg"] == invalid
    observe_run_inputs(
        fixture, tab_id, lambda name, params: call(sock, name, params)["result"]
    )
    rejected = call(
        sock, "tab.run_start", {"tab_id": tab_id, "expected": invalid["cfg_ref"]}
    )
    assert rejected["error"]["reason"] == "not_valid"
    assert fixture.ctrl.active_operation_count() == 0


def test_unavailable_publication_is_preserved_by_mcp_read_and_blocks_run(
    mcp_tab, monkeypatch
):
    fixture, tab_id, invoke, sock = mcp_tab
    owner = fixture.ctrl.cfg_resources.lookup(tab_id)
    before = call(sock, "tab.get_cfg", {"tab_id": tab_id})["result"]
    observe_run_inputs(
        fixture, tab_id, lambda name, params: call(sock, name, params)["result"]
    )

    def source_fault(*args, **kwargs):
        raise RuntimeError("source snapshot failed")

    with monkeypatch.context() as patch:
        patch.setattr(MeasureCfgBindings, "snapshot_from_state", source_fault)
        owner.refresh(owner.observe().ref.revision)
        unavailable = call(sock, "tab.get_cfg", {"tab_id": tab_id})["result"]
        assert unavailable["status"] == "Unavailable"
        assert unavailable["cfg_ref"] != before["cfg_ref"]
        assert (
            unavailable["tree"]["children"]["gain"]["input"]
            == (before["tree"]["children"]["gain"]["input"])
        )
        assert unavailable["diagnostics"]
        assert (
            invoke("tab_get", {"tab": tab_id, "include": ["cfg"]})["cfg"] == unavailable
        )
        rejected = call(
            sock,
            "tab.run_start",
            {"tab_id": tab_id, "expected": unavailable["cfg_ref"]},
        )
        assert rejected["error"]["reason"] == "not_valid"
        assert fixture.ctrl.active_operation_count() == 0
    owner.refresh(owner.observe().ref.revision)
    assert call(sock, "tab.get_cfg", {"tab_id": tab_id})["result"]["status"] == "Valid"


def test_edit_source_fault_does_not_publish_a_successful_prefix(mcp_tab, monkeypatch):
    _, tab_id, invoke, sock = mcp_tab
    before = call(sock, "tab.get_cfg", {"tab_id": tab_id})["result"]

    def source_fault(*args, **kwargs):
        raise RuntimeError("source snapshot failed")

    with monkeypatch.context() as patch:
        patch.setattr(MeasureCfgBindings, "snapshot_from_state", source_fault)
        with pytest.raises(GuiRpcError) as raised:
            invoke(
                "tab_edit",
                {
                    "tab": tab_id,
                    "expected": before["cfg_ref"],
                    "edits": [{"path": ["gain"], "value": 0.25}],
                },
            )
        assert raised.value.code == "controller_error"
        assert call(sock, "tab.get_cfg", {"tab_id": tab_id})["result"] == before


def test_reply_encoding_failure_does_not_roll_back_the_cfg_publication(qapp):
    fixture = Fixture()
    tab_id = "reply-failure-tab"
    owner = fixture.prepare_tab(
        tab_id,
        FakeAdapter(),
        CfgSchema(
            CfgSectionSpec(fields={"text": ScalarSpec("Text", str)}),
            CfgSectionValue({"text": DirectValue("before")}),
        ),
    )
    fixture.start()
    sock = open_client(fixture.service.port)
    try:
        before = call(sock, "tab.get_cfg", {"tab_id": tab_id})["result"]
        text = "x" * (4 * 1024 * 1024)
        # The read helper switches to nonblocking; a large request needs sendall
        # to wait for the server's reader thread before pumping owner events.
        sock.settimeout(10.0)
        rejected_reply = call(
            sock,
            "tab.edit_cfg",
            {
                "tab_id": tab_id,
                "expected": before["cfg_ref"],
                "edits": [{"path": ["text"], "value": text}],
            },
            timeout_s=10.0,
        )
        assert rejected_reply["error"]["reason"] == "response_encoding_failed"
        current = owner.observe().ref
        assert encode_ref(current) != before["cfg_ref"]
        assert owner.accept(current.revision).values == {"text": text}
        deliberate_retry = call(
            sock,
            "tab.edit_cfg",
            {
                "tab_id": tab_id,
                "expected": before["cfg_ref"],
                "edits": [{"path": ["text"], "value": "do not overwrite"}],
            },
        )
        assert deliberate_retry["error"]["reason"] == "stale_revision"
        assert deliberate_retry["error"]["data"]["actual"] == encode_ref(current)
        assert owner.observe().ref == current
        restored = owner.edit(
            current.revision, (CfgEdit(("text",), DirectValue("small")),)
        )
        assert call(sock, "tab.get_cfg", {"tab_id": tab_id})["result"]["cfg_ref"] == (
            encode_ref(restored.ref)
        )
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
                "expected": before["cfg_ref"],
                "edits": [
                    {"path": ["gain"], "value": 0.25},
                    {"path": ["missing"], "value": 1},
                ],
            },
        )
        assert reply["error"]["reason"] == "unknown_path"
        assert reply["error"]["data"] == {"path": ["missing"], "edit_index": 1}
        assert call(sock, "tab.get_cfg", {"tab_id": tab_id})["result"] == before
    finally:
        sock.close()


def test_malformed_envelope_precedes_resource_lookup(live_tab):
    fixture, _ = live_tab
    sock = open_client(fixture.service.port)
    try:
        reply = call(
            sock,
            "tab.edit_cfg",
            {
                "tab_id": "missing",
                "expected": {"cfg_id": "missing", "revision": "01"},
                "edits": [],
            },
        )
        assert reply["error"]["code"] == "invalid_params"
        assert reply["error"]["reason"] == "malformed_input"
    finally:
        sock.close()


def test_library_editor_batch_retains_its_independent_prefix_contract(live_tab):
    fixture, _ = live_tab
    schema = FakeAdapter().make_default_cfg(fixture.state.session_env)
    editor_id, _ = fixture.ctrl.open_seeded_cfg_editor(schema, gc=False)
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
    schema = FakeAdapter().make_default_cfg(fixture.state.session_env)
    editor_id, _ = fixture.ctrl.open_seeded_cfg_editor(schema, gc=False)

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
