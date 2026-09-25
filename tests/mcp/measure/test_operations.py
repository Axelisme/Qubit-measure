"""Operation handles and wait/poll results through assembled MCP handlers."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from ._support import MeasureClient, make_client


@pytest.fixture
def client(tmp_path: Path) -> MeasureClient:
    return make_client(tmp_path)


def ok(result: dict[str, Any]) -> dict[str, Any]:
    return {"ok": True, "result": result}


def timeout_reply() -> dict[str, Any]:
    return {"ok": False, "error": {"code": "timeout", "message": "not done"}}


@dataclass
class AwaitReplies:
    replies: list[dict[str, Any]]

    def __call__(self, params: dict[str, Any]) -> dict[str, Any]:
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


@pytest.mark.parametrize(
    ("method", "params", "key"),
    [
        ("device.setup", {"name": "flux", "updates": {}}, "device:flux"),
        ("device.connect", {"name": "flux"}, "device:flux"),
        ("tab.run_start", {"tab_id": "t"}, "tab:t"),
    ],
)
def test_latest_operation_handle_is_reported_only_by_its_own_session(
    tmp_path: Path,
    method: str,
    params: dict[str, Any],
    key: str,
) -> None:
    first, second = make_client(tmp_path), make_client(tmp_path)
    for client, handle in ((first, 42), (second, 91), (first, 43)):
        client.transport.replies[method] = ok({"operation_id": handle})
        assert client.context.send_gui_rpc(method, params) == {"handle": handle}
    assert first.call("gui_debug_operations", {}) == {
        "handles": {key: {"operation_id": 43}}
    }
    assert second.call("gui_debug_operations", {}) == {
        "handles": {key: {"operation_id": 91}}
    }


@pytest.mark.parametrize("settled", [True, False])
@pytest.mark.parametrize("apply", [False, True])
def test_device_start_short_wait_returns_product_or_pending_handle(
    client: MeasureClient,
    settled: bool,
    apply: bool,
) -> None:
    client.transport.replies.update(
        {
            "device.setup" if apply else "device.connect": ok({"operation_id": 9}),
            "operation.await": ok({"status": "finished"})
            if settled
            else timeout_reply(),
            "device.snapshot": ok(
                {"snapshot": {"name": "flux", "status": "connected"}}
            ),
        }
    )
    result = client.call(
        "gui_device_apply" if apply else "gui_device_connect",
        {"name": "flux", "updates": {"value": 1.0}}
        if apply
        else {
            "type_name": "FakeDevice",
            "name": "flux",
            "address": "addr",
        },
    )
    if apply:
        setup = next(
            params
            for method, params in client.transport.sent
            if method == "device.setup"
        )
        assert {
            key: value for key, value in setup.items() if key != "expected_versions"
        } == {"name": "flux", "updates": {"value": 1.0}}
    if settled:
        assert result["status"] == "finished"
        assert result["snapshot"] == {"name": "flux", "status": "connected"}
    else:
        assert result["status"] == "pending"
        assert result["handle"] == 9
        assert "flux" in result["message"]


@pytest.mark.parametrize("feedback", [None, "user pressed Stop"])
def test_wait_forwards_handle_and_preserves_cancelled_feedback(
    client: MeasureClient,
    feedback: str | None,
) -> None:
    outcome: dict[str, Any] = {"reason": "completed", "status": "cancelled"}
    if feedback is not None:
        outcome["feedback"] = feedback
    client.transport.replies["operation.await"] = ok(outcome)
    result = client.call("gui_op_wait", {"handle": 5, "timeout": 0.05})
    assert result["status"] == "cancelled"
    assert result["waited_seconds"] >= 0
    if feedback is None:
        assert "feedback" not in result
    else:
        assert result["feedback"] == feedback
    assert next(
        params
        for method, params in client.transport.sent
        if method == "operation.await"
    ) == {
        "operation_id": 5,
        "timeout": 0.05,
    }


def test_finished_wait_refreshes_versions_for_following_save(
    client: MeasureClient,
) -> None:
    client.transport.replies.update(
        {
            "operation.await": ok({"status": "finished"}),
            "resources.versions": ok({"versions": {"tab:t:result": 9}}),
            "tab.save_data": ok({}),
        }
    )
    result = client.call("gui_op_wait", {"handle": 42, "timeout": 0.1})
    assert result["status"] == "finished"
    assert result["waited_seconds"] >= 0
    client.context.send_gui_rpc("tab.save_data", {"tab_id": "t"})
    assert (
        next(p for m, p in client.transport.sent if m == "tab.save_data")[
            "expected_versions"
        ]["tab:t:result"]
        == 9
    )


@pytest.mark.parametrize("tool", ["gui_op_wait", "gui_op_poll"])
def test_failed_operation_raises_for_wait_but_is_reported_by_poll(
    client: MeasureClient,
    tool: str,
) -> None:
    client.transport.replies["operation.await"] = {
        "ok": False,
        "error": {
            "code": "precondition_failed",
            "reason": "failed",
            "message": "hardware boom",
        },
    }
    if tool == "gui_op_wait":
        with pytest.raises(RuntimeError, match="hardware boom"):
            client.call(tool, {"handle": 7})
    else:
        result = client.call(tool, {"handle": 7})
        assert result["status"] == "failed"
        assert "hardware boom" in result["message"]


@pytest.mark.parametrize("messages", [[], ["first", "second", "third"]])
def test_running_poll_drains_feedback_and_projects_progress(
    client: MeasureClient,
    messages: list[str],
) -> None:
    client.transport.replies["operation.await"] = AwaitReplies(
        [
            *[
                ok({"reason": "user_feedback", "feedback": message})
                for message in messages
            ],
            timeout_reply(),
        ]
    )
    client.transport.replies["operation.progress"] = ok(
        {
            "active": True,
            "bars": [
                {"token": "outer", "format": "2/10", "percent": 20, "maximum": 10}
            ],
        }
    )
    result = client.call("gui_op_poll", {"handle": 9})
    assert result["status"] == "running"
    assert result["active"] is True
    assert result["bars"] == [{"token": "outer", "format": "2/10", "percent": 20}]
    if messages:
        assert result["feedback"] == messages
    else:
        assert "feedback" not in result
    follow = client.call("gui_op_wait", {"handle": 9, "timeout": 0.01})
    assert follow["status"] == "timed_out"
    assert "feedback" not in follow


@pytest.mark.parametrize("status", ["finished", "cancelled"])
@pytest.mark.parametrize("messages", [[], ["note A", "note B"]])
def test_poll_drains_feedback_before_a_rereadable_terminal_result(
    client: MeasureClient,
    status: str,
    messages: list[str],
) -> None:
    terminal = {"reason": "completed", "status": status}
    if status == "cancelled":
        terminal["feedback"] = "stop: recalibrate first"
    client.transport.replies["operation.await"] = AwaitReplies(
        [
            *[
                ok({"reason": "user_feedback", "feedback": message})
                for message in messages
            ],
            ok(terminal),
        ]
    )
    result = client.call("gui_op_poll", {"handle": 13})
    assert result["status"] == status
    if messages:
        assert result["feedback"] == messages
    else:
        assert "feedback" not in result
    if status == "cancelled":
        assert result["stop_reason"] == "stop: recalibrate first"
        assert "recalibrate" in result["message"]
    follow = client.call("gui_op_wait", {"handle": 13})
    assert follow["status"] == status
    if status == "cancelled":
        assert follow["feedback"] == "stop: recalibrate first"
