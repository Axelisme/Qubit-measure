"""Timeout recovery keeps the owner submission identity instead of retrying."""

from pathlib import Path

from ._support import make_client


def test_apply_timeout_returns_queryable_owner_submission_without_retry(tmp_path: Path):
    client = make_client(tmp_path)
    for method in ("tab.get_analyze_result", "tab.get_post_analyze_result"):
        client.transport.replies[method] = {"ok": True, "result": {"summary": {}}}
    client.transport.replies["tab.writeback_preview"] = lambda params: {
        "ok": True,
        "result": {
            "has_draft": True,
            "items": [{"id": "param-1", "target_name": params["subtab_id"]}],
            "destination_context": {},
        },
    }
    client.transport.replies["tab.writeback_begin"] = {
        "ok": True,
        "result": {"submission_id": "wb-1", "status": "pending"},
    }
    client.transport.replies["tab.writeback_write"] = {
        "ok": False,
        "error": {"code": "timeout", "message": "handler budget expired"},
    }
    client.transport.replies["tab.writeback_receipt"] = {
        "ok": True,
        "result": {
            "submission_id": "wb-1",
            "status": "completed",
            "completed": [
                {
                    "subtab_id": "analysis",
                    "item": {
                        "id": "param-1",
                        "kind": "parameter",
                        "target": "Q1.ext.gain",
                        "value": 0.5,
                        "destination": {"entry_id": "entry", "point": "point"},
                    },
                }
            ],
            "error": None,
        },
    }
    try:
        result = client.call("apply_writeback", {"tab": "tab-1"})
        assert result.get("submission_id") == "wb-1"
        assert result["status"] == "unknown"
        assert result["failed_stage_may_have_partial_writes"] is True
        assert result["not_started"] == ["post"]
        assert result["recovery"] == {
            "rpc": "tab.writeback_receipt",
            "params": {"submission_id": "wb-1"},
        }
        assert (
            "Check status before deciding whether to resubmit" in result["next_action"]
        )
        writes = [
            (method, params)
            for method, params in client.transport.sent
            if method == "tab.writeback_write"
        ]
        assert len(writes) == 1
        assert writes[0][1]["submission_id"] == "wb-1"
        recovered = client.call(
            "rpc_call",
            {
                "method": result["recovery"]["rpc"],
                "params": result["recovery"]["params"],
            },
        )
        assert recovered["status"] == "completed"
        assert recovered["completed"][0]["item"]["value"] == 0.5
    finally:
        client.context.session.close()
