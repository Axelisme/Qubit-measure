"""Agent whole-sweep cfg edit through the live tab's GUI socket."""

from __future__ import annotations

import pytest
from zcu_tools.experiment.v2_gui.measure.adapters.fake import FakeAdapter
from zcu_tools.gui.app.measure.state import Session
from zcu_tools.gui.cfg import (
    CenteredSweepSpec,
    CfgSchema,
    CfgSectionSpec,
    ReferenceSpec,
    ScalarSpec,
    make_default_value,
)
from zcu_tools.mcp.measure.session import GuiRpcError

from ._helpers import Fixture, call, mcp_client, open_client


@pytest.fixture()
def live_tab(qapp):
    fixture = Fixture()
    tab_id = "tab-agent-sweep"
    cfg = FakeAdapter().make_default_cfg(fixture.state.session_env)
    fixture.state.add_tab(
        tab_id, Session(adapter_name="fake", adapter=FakeAdapter(), cfg_schema=cfg)
    )
    fixture.ctrl.open_seeded_cfg_editor(cfg, gc=False, owner_key=tab_id)
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


def test_tab_get_mcp_projects_gui_types_choices_and_locks_without_focusing(
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
    cfg = CfgSchema(spec, make_default_value(spec))
    tab_id = "tab-agent-cfg-read"
    fixture.state.add_tab(
        tab_id, Session(adapter_name="fake", adapter=FakeAdapter(), cfg_schema=cfg)
    )
    fixture.ctrl.open_seeded_cfg_editor(cfg, gc=False, owner_key=tab_id)
    fixture.start()
    monkeypatch.setattr("zcu_tools.mcp.measure.tools_lifecycle.status", lambda *_: {})
    bridge, invoke = mcp_client(fixture.service.port, tmp_path)
    sock = open_client(fixture.service.port)
    try:
        invoke("connect", {"port": fixture.service.port})
        before = call(sock, "tab.list_all")
        result = invoke("tab_get", {"tab": tab_id, "include": ["cfg"]})
        children = result["cfg"]["children"]
        assert children["axis"]["kind"] == "centered_sweep"
        assert children["axis"]["locked_center"] == 0.0
        assert children["axis"]["center_editable"] is False
        assert children["nqz"]["kind"] == "scalar"
        assert children["nqz"]["type"] == "int"
        assert children["nqz"]["choices"] == [1, 2]
        assert children["drive"]["kind"] == "reference"
        assert children["drive"]["choices"] == ["Pulse"]
        assert result.get("partial") is None
        assert (
            result["cfg"]
            == call(sock, "tab.get_cfg", {"tab_id": tab_id})["result"]["tree"]
        )
        assert (
            call(sock, "tab.list_all")["result"]["active_tab_id"]
            == before["result"]["active_tab_id"]
        )
    finally:
        bridge.disconnect()
        sock.close()
        fixture.stop()


def test_tab_edit_mcp_normalizes_sweep_on_live_gui_draft(mcp_tab):
    _, tab_id, invoke, sock = mcp_tab
    result = invoke(
        "tab_edit",
        {
            "tab": tab_id,
            "edits": [
                {"path": "sweep", "value": {"start": 2.0, "stop": 8.0, "step": 2.2}}
            ],
        },
    )
    assert result["applied"] == 1 and result["valid"] is True
    assert result["actual"]["sweep"] == {
        "start": 2.0,
        "stop": 8.0,
        "expts": 4,
        "step": 2.0,
    }
    observed = call(sock, "tab.get_cfg", {"tab_id": tab_id})
    assert observed["ok"] is True
    assert (
        observed["result"]["tree"]["children"]["sweep"]["inputs"]["expts"]["resolved"]
        == 4
    )


def test_tab_edit_mcp_failure_preserves_successful_prefix(mcp_tab):
    _, tab_id, invoke, sock = mcp_tab
    with pytest.raises(GuiRpcError) as raised:
        invoke(
            "tab_edit",
            {
                "tab": tab_id,
                "edits": [
                    {
                        "path": "sweep",
                        "value": {"start": 2.0, "stop": 8.0, "expts": 4},
                    },
                    {"path": "sweep", "value": {"start": 3.0, "stop": 9.0}},
                ],
            },
        )
    assert raised.value.reason == "invalid_settable_path"
    assert "'sweep' failed after 1 applied" in str(raised.value)
    observed = call(sock, "tab.get_cfg", {"tab_id": tab_id})
    assert observed["ok"] is True
    inputs = observed["result"]["tree"]["children"]["sweep"]["inputs"]
    assert inputs["start"]["resolved"] == pytest.approx(2.0)
    assert inputs["expts"]["resolved"] == 4


@pytest.mark.parametrize("method", ["tab.set_cfg", "editor.set_fields"])
@pytest.mark.parametrize("source", ["value_ref", "reference_child"])
def test_agent_batch_preserves_precondition_category_and_reason(
    live_tab, monkeypatch, method, source
):
    from zcu_tools.gui.session.value_lookup import UnavailableValue

    fixture, tab_id = live_tab
    spec = CfgSectionSpec(
        fields={
            "gain": ScalarSpec("Gain", float),
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
    editor_id, _ = fixture.ctrl.open_seeded_cfg_editor(
        CfgSchema(spec, make_default_value(spec)), gc=False, owner_key=tab_id
    )
    if source == "value_ref":

        def unavailable(*_args):
            raise UnavailableValue("device.flux.value", "device unavailable")

        monkeypatch.setattr(fixture.ctrl, "read_value_source", unavailable)
        failing = {
            "path": "gain",
            "value": {
                "__kind": "value_ref",
                "key": "device.flux.value",
                "type": "float",
            },
        }
        reason = ""
    else:
        from zcu_tools.gui.cfg.binding.targets import SettableTargetUnavailable

        draft = fixture.ctrl.get_cfg_editor_draft(editor_id)
        resolve = draft.resolve_agent_target

        def resolve_available(path):
            if path == "drive.gain":
                raise SettableTargetUnavailable("reference shape currently unavailable")
            return resolve(path)

        monkeypatch.setattr(draft, "resolve_agent_target", resolve_available)
        failing = {"path": "drive.gain", "value": 0.5}
        reason = "settable_target_unavailable"
    params = (
        {"tab_id": tab_id, "agent_edit": True}
        if method == "tab.set_cfg"
        else {"editor_id": editor_id}
    )
    params["edits"] = [{"path": "gain", "value": 0.25}, failing]
    sock = open_client(fixture.service.port)
    try:
        reply = call(sock, method, params)
        assert reply["error"]["code"] == "precondition_failed"
        assert reply["error"].get("reason", "") == reason
        assert (
            f"{failing['path']!r} failed after 1 applied" in reply["error"]["message"]
        )
        tree = call(sock, "editor.get", {"editor_id": editor_id})["result"]["tree"]
        assert tree["children"]["gain"]["input"]["resolved"] == 0.25
    finally:
        sock.close()


def test_library_editor_batch_uses_shared_agent_sweep_and_preserves_prefix(live_tab):
    fixture, tab_id = live_tab
    editor_id = fixture.ctrl.editor_id_for_owner(tab_id)
    assert editor_id is not None
    sock = open_client(fixture.service.port)
    try:
        result = call(
            sock,
            "editor.set_fields",
            {
                "editor_id": editor_id,
                "edits": [
                    {"path": "sweep", "value": {"start": 2.0, "stop": 8.0, "step": 2.2}}
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
        assert rejected["ok"] is False
        assert rejected["error"]["reason"] == "invalid_settable_path"
        assert "'sweep' failed after 1 applied" in rejected["error"]["message"]
        tree = call(sock, "editor.get", {"editor_id": editor_id})["result"]["tree"]
        assert tree["children"]["gain"]["input"]["resolved"] == pytest.approx(0.25)
        assert tree["children"]["sweep"]["inputs"]["expts"]["resolved"] == 4
    finally:
        sock.close()


def test_agent_edit_normalizes_whole_sweep_in_the_shared_tab_draft(live_tab):
    fixture, tab_id = live_tab
    sock = open_client(fixture.service.port)
    try:
        resp = call(
            sock,
            "tab.set_cfg",
            {
                "tab_id": tab_id,
                "agent_edit": True,
                "edits": [
                    {"path": "sweep", "value": {"start": 2.0, "stop": 8.0, "step": 2.2}}
                ],
            },
        )
        assert resp["ok"] is True, resp
        assert resp["result"]["applied"] == 1
        assert resp["result"]["actual"]["sweep"] == {
            "start": 2.0,
            "stop": 8.0,
            "expts": 4,
            "step": 2.0,
        }
        observed = call(sock, "tab.get_cfg", {"tab_id": tab_id})
        assert observed["ok"] is True
        inputs = observed["result"]["tree"]["children"]["sweep"]["inputs"]
        assert inputs["expts"]["resolved"] == 4
        assert inputs["step"]["resolved"] == pytest.approx(2.0)
    finally:
        sock.close()


def test_agent_batch_failure_preserves_applied_sweep_and_names_failed_path(live_tab):
    fixture, tab_id = live_tab
    sock = open_client(fixture.service.port)
    try:
        resp = call(
            sock,
            "tab.set_cfg",
            {
                "tab_id": tab_id,
                "agent_edit": True,
                "edits": [
                    {
                        "path": "sweep",
                        "value": {"start": 2.0, "stop": 8.0, "step": 2.0},
                    },
                    {
                        "path": "sweep",
                        "value": {
                            "start": 3.0,
                            "stop": 9.0,
                            "step": 2.0,
                            "expts": 4,
                        },
                    },
                ],
            },
        )
        assert resp["ok"] is False
        assert resp["error"]["code"] == "invalid_params"
        assert resp["error"]["reason"] == "invalid_settable_path"
        assert "'sweep' failed after 1 applied" in resp["error"]["message"]
        observed = call(sock, "tab.get_cfg", {"tab_id": tab_id})
        assert observed["ok"] is True
        inputs = observed["result"]["tree"]["children"]["sweep"]["inputs"]
        assert inputs["start"]["resolved"] == pytest.approx(2.0)
        assert inputs["stop"]["resolved"] == pytest.approx(8.0)
        assert inputs["expts"]["resolved"] == 4
        assert inputs["step"]["resolved"] == pytest.approx(2.0)
    finally:
        sock.close()
