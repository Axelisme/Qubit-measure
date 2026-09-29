"""Save tool outcomes over the assembled MCP/session transport seam."""

from pathlib import Path

import pytest
from zcu_tools.mcp.measure.session import GuiRpcError

from ._support import make_client


@pytest.mark.parametrize("status", ["finished", "running", "failed"])
def test_save_submits_once_and_reports_only_terminal_success(
    tmp_path: Path, status: str
) -> None:
    client = make_client(tmp_path)
    destinations = {
        "analysis": str(tmp_path / "fit.png"),
        "data": str(tmp_path / "data_1.hdf5"),
    }
    client.transport.replies["tab.save_artifacts"] = {
        "ok": True,
        "result": {"operation_id": 71, "destinations": destinations},
    }
    terminal: dict[str, object] = {"reason": "completed", "status": status}
    if status == "failed":
        terminal["error"] = {"message": "disk full"}
    if status == "running":
        terminal = {"reason": "timeout"}
        client.transport.replies["operation.progress"] = {
            "ok": True,
            "result": {"active": True, "bars": []},
        }
    client.transport.replies["operation.await"] = {"ok": True, "result": terminal}
    arguments = {
        "tab": "tab-a",
        "artifacts": ["analysis:fit", "data"],
        "paths": {"data": str(tmp_path / "data.hdf5")},
        "comment": "sample",
    }
    if status == "failed":
        with pytest.raises(GuiRpcError, match="disk full") as exc:
            client.call("tab_save", arguments)
        assert exc.value.reason == "operation_failed"
    else:
        result = client.call("tab_save", arguments)
        if status == "finished":
            assert result == {"saved": destinations}
        else:
            assert set(result) == {"op"}
            assert result["op"] > 0
            client.transport.replies["operation.await"] = {
                "ok": True,
                "result": {"reason": "completed", "status": "finished"},
            }
            assert (
                client.call("wait", {"op": result["op"], "timeout": 0})["status"]
                == "finished"
            )
    submissions = [
        params
        for method, params in client.transport.sent
        if method == "tab.save_artifacts"
    ]
    assert submissions == [
        {
            "tab_id": "tab-a",
            "artifacts": arguments["artifacts"],
            "paths": arguments["paths"],
            "comment": "sample",
        }
    ]
    waits = [
        params
        for method, params in client.transport.sent
        if method == "operation.await"
    ]
    assert waits[0] == {"operation_id": 71, "timeout": 2}
    assert not any(
        method in {"tab.snapshot", "tab.get_cfg", "context.snapshot"}
        for method, _ in client.transport.sent
    )


def test_save_omission_preserves_gui_drafts_and_stale_is_not_retried(
    tmp_path: Path,
) -> None:
    client = make_client(tmp_path)
    client.transport.replies["tab.save_artifacts"] = {
        "ok": False,
        "error": {
            "code": "precondition_failed",
            "reason": "stale_version",
            "message": "read tab before saving",
            "data": {"stale": ["tab:tab-a:path:data"]},
        },
    }
    with pytest.raises(GuiRpcError) as exc:
        client.call("tab_save", {"tab": "tab-a"})
    assert exc.value.reason == "stale_version"
    domain_calls = [
        (method, params)
        for method, params in client.transport.sent
        if method.startswith(("tab.", "operation."))
    ]
    assert domain_calls == [("tab.save_artifacts", {"tab_id": "tab-a"})]
