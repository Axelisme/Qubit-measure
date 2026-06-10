"""mcp_server-side optimistic-concurrency bookkeeping.

Drives send_gui_rpc over a synchronous FakeTransport injected into the bridge:
each send_line echoes a per-method crafted reply, so we assert the mcp policy
(attach expected_versions for guarded ops, translate a stale rejection, refresh
_LAST_SEEN after every successful RPC) without a real GUI.

Post-E4/F: socket I/O lives behind the McpBridge transport seam — tests inject a
synchronous ``FakeTransport`` (no socket, no thread) and run the bridge's REAL
``send_rpc_raw``; the mcp policy (``send_gui_rpc`` / ``_LAST_SEEN`` / guard) stays
on ``mcp_server`` and is asserted there. No socket internals are patched.
"""

from __future__ import annotations

from typing import Any

import pytest
from zcu_tools.mcp.measure import server as mcp_server

from ._helpers import FakeTransport


@pytest.fixture()
def wired(monkeypatch):
    """Inject a FakeTransport into the bridge; reset the guard baseline.

    Returns a dict you populate as ``{method: reply_envelope}``; ``["sent"]``
    records every outgoing ``(method, params)`` so tests can assert what the guard
    attached.
    """
    fake = FakeTransport()
    mcp_server._BRIDGE.set_transport(fake)
    monkeypatch.setattr(mcp_server, "_LAST_SEEN", {}, raising=False)
    replies: dict[str, dict[str, Any]] = fake.replies
    replies["sent"] = fake.sent  # type: ignore[assignment]
    yield replies
    mcp_server._BRIDGE.set_transport(None)


def _versions_reply(table: dict[str, int]) -> dict[str, Any]:
    return {"ok": True, "result": {"versions": table}}


def test_guarded_op_attaches_expected_versions(wired):
    sent = wired["sent"]
    # Baseline the agent has observed.
    mcp_server._LAST_SEEN.update(
        {
            "tab:t:cfg": 3,
            "tab:t": 1,
            "soc": 2,
            "context": 4,
            "device:yoko": 5,
            "devices:__set__": 6,
        }
    )
    wired["run.start"] = {"ok": True, "result": {}}
    wired["resources.versions"] = _versions_reply(dict(mcp_server._LAST_SEEN))

    mcp_server.send_gui_rpc("run.start", {"tab_id": "t"})

    run_params = next(p for (m, p) in sent if m == "run.start")
    assert run_params["expected_versions"] == {
        "tab:t:cfg": 3,
        "tab:t": 1,
        "soc": 2,
        "context": 4,
        "device:yoko": 5,
        "devices:__set__": 6,
    }


def test_run_start_declares_device_set_cardinality_key(wired):
    """run.start must declare devices:__set__ so a concurrently-added device
    (which device:* glob cannot reveal) is caught by the guard."""
    sent = wired["sent"]
    # Agent observed an empty device set (cardinality key unseen → 0).
    mcp_server._LAST_SEEN.update({"tab:t:cfg": 1, "tab:t": 1, "soc": 1, "context": 1})
    wired["run.start"] = {"ok": True, "result": {}}
    wired["resources.versions"] = _versions_reply(dict(mcp_server._LAST_SEEN))

    mcp_server.send_gui_rpc("run.start", {"tab_id": "t"})

    expected = next(p for (m, p) in sent if m == "run.start")["expected_versions"]
    # Declared at its last-seen baseline of 0; the server rejects if a device was
    # added since (cardinality now ≥ 1).
    assert expected["devices:__set__"] == 0


def test_save_depends_on_result_and_save_path_not_cfg(wired):
    sent = wired["sent"]
    mcp_server._LAST_SEEN.update(
        {"tab:t:result": 7, "tab:t:save_path": 2, "tab:t:cfg": 9}
    )
    wired["save.data"] = {"ok": True, "result": {}}
    wired["resources.versions"] = _versions_reply(dict(mcp_server._LAST_SEEN))

    mcp_server.send_gui_rpc("save.data", {"tab_id": "t"})

    params = next(p for (m, p) in sent if m == "save.data")
    assert params["expected_versions"] == {"tab:t:result": 7, "tab:t:save_path": 2}
    assert "tab:t:cfg" not in params["expected_versions"]


def test_writeback_apply_depends_on_result_analyze_and_context(wired):
    sent = wired["sent"]
    mcp_server._LAST_SEEN.update(
        {"tab:t:result": 7, "tab:t:analyze": 4, "context": 9, "tab:t:save_path": 2}
    )
    wired["writeback.apply"] = {"ok": True, "result": {}}
    wired["resources.versions"] = _versions_reply(dict(mcp_server._LAST_SEEN))

    mcp_server.send_gui_rpc("writeback.apply", {"tab_id": "t", "selections": []})

    params = next(p for (m, p) in sent if m == "writeback.apply")
    assert params["expected_versions"] == {
        "tab:t:result": 7,
        "tab:t:analyze": 4,
        "context": 9,
    }
    # save_path is irrelevant to writeback.
    assert "tab:t:save_path" not in params["expected_versions"]


def test_unguarded_op_attaches_nothing(wired):
    sent = wired["sent"]
    wired["tab.snapshot"] = {"ok": True, "result": {"x": 1}}
    wired["resources.versions"] = _versions_reply({})

    mcp_server.send_gui_rpc("tab.snapshot", {"tab_id": "t"})

    params = next(p for (m, p) in sent if m == "tab.snapshot")
    assert "expected_versions" not in params


def test_stale_rejection_translated_and_refreshes(wired):
    mcp_server._LAST_SEEN.update({"tab:t:cfg": 3, "tab:t": 1, "soc": 1, "context": 1})
    wired["run.start"] = {
        "ok": False,
        "error": {
            "code": "precondition_failed",
            "reason": "stale_version",
            "message": "stale",
        },
    }
    # After rejection the bridge re-reads the table; the human's edit bumped cfg.
    wired["resources.versions"] = _versions_reply({"tab:t:cfg": 4})

    with pytest.raises(RuntimeError) as ei:
        mcp_server.send_gui_rpc("run.start", {"tab_id": "t"})

    msg = str(ei.value)
    assert "PRECONDITION_FAILED" in msg
    # No raw version numbers leak into the agent-facing message.
    assert "4" not in msg and "tab:t:cfg" not in msg
    # _LAST_SEEN was resynced from the post-rejection read.
    assert mcp_server._LAST_SEEN == {"tab:t:cfg": 4}


def test_successful_rpc_refreshes_last_seen(wired):
    wired["state.has_soc"] = {"ok": True, "result": {"value": True}}
    wired["resources.versions"] = _versions_reply({"soc": 9, "context": 1})

    mcp_server.send_gui_rpc("state.has_soc", {})

    assert mcp_server._LAST_SEEN == {"soc": 9, "context": 1}


def test_device_glob_expands_to_all_device_keys(wired):
    sent = wired["sent"]
    mcp_server._LAST_SEEN.update(
        {
            "tab:t:cfg": 1,
            "tab:t": 1,
            "soc": 1,
            "context": 1,
            "device:yoko": 2,
            "device:sgs": 3,
        }
    )
    wired["run.start"] = {"ok": True, "result": {}}
    wired["resources.versions"] = _versions_reply(dict(mcp_server._LAST_SEEN))

    mcp_server.send_gui_rpc("run.start", {"tab_id": "t"})

    expected = next(p for (m, p) in sent if m == "run.start")["expected_versions"]
    assert expected["device:yoko"] == 2
    assert expected["device:sgs"] == 3
