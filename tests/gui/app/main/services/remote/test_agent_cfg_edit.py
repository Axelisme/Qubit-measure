"""Agent whole-sweep cfg edit through the live tab's GUI socket."""

from __future__ import annotations

import pytest
from zcu_tools.experiment.v2_gui.adapters.fake import FakeAdapter
from zcu_tools.gui.app.main.state import Session
from zcu_tools.mcp.measure.session import GuiRpcError

from ._helpers import Fixture, call, mcp_client, open_client


@pytest.fixture()
def live_tab(qapp):
    fixture = Fixture()
    tab_id = "tab-agent-sweep"
    cfg = FakeAdapter().make_default_cfg(fixture.state.exp_context)
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
