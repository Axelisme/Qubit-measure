"""Public analyze tool contracts over the recording GUI transport."""

import pytest

from ._support import make_client


@pytest.mark.parametrize(
    "stage, method, result_method",
    [
        ("primary", "tab.analyze", "tab.get_analyze_result"),
        ("post", "tab.post_analyze", "tab.get_post_analyze_result"),
    ],
)
def test_finished_analysis_uses_start_facts_without_hidden_pre_reads(
    tmp_path, stage, method, result_method
):
    def respond(name, params):
        if name == method:
            assert params == {"tab_id": "t", "updates": {"gain": 2}}
            return {
                "operation_id": 71,
                "interactive": False,
                "params": {"gain": 2, "model": "fit"},
                "invalidated_on_success": ["post.writeback"],
            }
        if name == "operation.await":
            assert params["timeout"] == 2.0
            return {"reason": "completed", "status": "finished"}
        if name == result_method:
            return {"summary": {"frequency": 5.0}}
        raise AssertionError(name)

    client = make_client(tmp_path, respond)
    client.transport.replies["tab.get_figure"] = {
        "ok": False,
        "error": {"code": "precondition_failed", "message": "no figure"},
    }
    assert client.call(
        "tab_analyze", {"tab": "t", "stage": stage, "params": {"gain": 2}}
    ) == {
        "status": "finished",
        "summary": {"frequency": 5.0},
        "figure": None,
        "params": {"gain": 2, "model": "fit"},
        "invalidated": ["post.writeback"],
    }
    assert [
        name
        for name, _ in client.transport.sent
        if name not in ("wire.version", "rpc.catalog")
    ] == [
        method,
        "operation.await",
        result_method,
        "tab.get_figure",
    ]


@pytest.mark.parametrize("status", ["interactive", "running", "failed"])
def test_unfinished_analysis_never_reads_success_payload(tmp_path, status):
    def respond(method, params):
        if method == "tab.analyze":
            assert params == {"tab_id": "t", "updates": {}}
            return {
                "operation_id": 71,
                "interactive": status == "interactive",
                "params": {},
                "invalidated_on_success": [],
            }
        if method == "operation.await":
            if status == "running":
                return {"reason": "timeout"}
            return {"reason": "completed", "status": "failed", "error": "fit failed"}
        if method == "operation.progress":
            return {"active": False}
        raise AssertionError(method)

    client = make_client(tmp_path, respond)
    result = client.call("tab_analyze", {"tab": "t"})
    assert result["status"] == status
    assert isinstance(result["op"], int)
    if status == "failed":
        assert result["error"] == "fit failed"
    methods = [
        name
        for name, _ in client.transport.sent
        if name not in ("wire.version", "rpc.catalog")
    ]
    assert (
        methods
        == {
            "interactive": ["tab.analyze"],
            "running": ["tab.analyze", "operation.await", "operation.progress"],
            "failed": ["tab.analyze", "operation.await"],
        }[status]
    )
