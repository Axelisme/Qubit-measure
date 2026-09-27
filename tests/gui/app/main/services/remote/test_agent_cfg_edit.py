"""Agent whole-sweep cfg edit through the live tab's GUI socket."""

from __future__ import annotations

import pytest
from zcu_tools.experiment.v2_gui.adapters.fake import FakeAdapter
from zcu_tools.gui.app.main.state import Session

from ._helpers import Fixture, call, open_client


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
