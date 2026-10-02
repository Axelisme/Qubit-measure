"""Writeback tools use one GUI-owned preview or batch command."""

import pytest
from zcu_tools.mcp.measure.session import GuiRpcError

from ._support import make_client


@pytest.mark.parametrize(
    "stage,pane", [("primary", "analysis"), ("post", "post_analysis")]
)
def test_preview_projects_complete_gui_values(tmp_path, stage, pane):
    client = make_client(tmp_path)
    client.transport.replies["tab.writeback_preview"] = {
        "ok": True,
        "result": {
            "destination_context": {"active_label": "ctx"},
            "items": [
                {
                    "id": "md-1",
                    "kind": "metadict",
                    "target_name": "freq",
                    "description": "Frequency",
                    "current": [1, 2],
                    "proposed": [3, 4],
                }
            ],
        },
    }
    result = client.call("writeback", {"tab": "t", "stage": stage})
    assert result == {
        "destination": {"active_label": "ctx"},
        "items": [
            {
                "id": "md-1",
                "kind": "md",
                "target": "freq",
                "description": "Frequency",
                "current": [1, 2],
                "proposed": [3, 4],
            }
        ],
    }
    assert [
        entry for entry in client.transport.sent if entry[0].startswith("tab.")
    ] == [
        ("tab.writeback_preview", {"tab_id": "t", "subtab_id": pane}),
    ]


@pytest.mark.parametrize("write", [[], [{"id": "md-1", "value": None}]])
def test_write_is_one_command_without_hidden_reads(tmp_path, write):
    client = make_client(tmp_path)
    reply = {
        "written": [
            {
                "id": "ml-1",
                "kind": "module",
                "target": "same",
                "before": {},
                "after": {"gain": 0.5},
            },
            {
                "id": "wf-1",
                "kind": "waveform",
                "target": "same",
                "before": None,
                "after": {"length": 2},
            },
        ]
    }
    client.transport.replies["tab.writeback_write"] = {"ok": True, "result": reply}
    assert client.call("writeback", {"tab": "t", "write": write}) == reply
    domain_calls = [
        entry
        for entry in client.transport.sent
        if entry[0]
        not in (
            "wire.version",
            "rpc.catalog",
            "resources.versions",
        )
    ]
    assert domain_calls == [
        (
            "tab.writeback_write",
            {
                "tab_id": "t",
                "subtab_id": "analysis",
                "write": write,
            },
        )
    ]


def test_write_failure_is_not_retried(tmp_path):
    client = make_client(tmp_path)
    client.transport.replies["tab.writeback_write"] = {
        "ok": False,
        "error": {"code": "precondition_failed", "message": "draft unavailable"},
    }
    with pytest.raises(GuiRpcError, match="draft unavailable"):
        client.call("writeback", {"tab": "t", "write": [{"id": "md-1"}]})
    assert (
        sum(method == "tab.writeback_write" for method, _ in client.transport.sent) == 1
    )


def test_invalid_stage_fails_before_transport(tmp_path):
    client = make_client(tmp_path)
    with pytest.raises(ValueError, match="stage"):
        client.call("writeback", {"tab": "t", "stage": "run"})
    assert client.transport.sent == []


@pytest.fixture
def accept_client(tmp_path):
    client = make_client(tmp_path)
    for method in ("tab.get_analyze_result", "tab.get_post_analyze_result"):
        client.transport.replies[method] = {"ok": True, "result": {"summary": {}}}
    client.transport.replies["tab.writeback_preview"] = {
        "ok": True,
        "result": {
            "has_draft": True,
            "items": [
                {"id": "md-1", "selected": False},
                {"id": "wf-1", "selected": True},
            ],
        },
    }
    client.transport.replies["tab.writeback_write"] = lambda params: {
        "ok": True,
        "result": {
            "written": [
                {
                    "id": item["id"],
                    "kind": "md" if item["id"].startswith("md-") else "waveform",
                    "target": params["subtab_id"],
                    "before": None,
                    "after": {"value": 2},
                }
                for item in params["write"]
            ]
        },
    }
    return client


def test_accept_writes_all_ids_in_each_pane_without_hidden_reads(accept_client):
    result = accept_client.call("accept", {"tab": "t"})
    assert result == {
        "tab": "t",
        "status": "finished",
        "completed": [
            {
                "stage": stage,
                "written": [
                    {
                        "id": item_id,
                        "kind": kind,
                        "target": pane,
                        "before": None,
                        "after": {"value": 2},
                    }
                    for item_id, kind in (("md-1", "md"), ("wf-1", "waveform"))
                ],
            }
            for stage, pane in (("primary", "analysis"), ("post", "post_analysis"))
        ],
        "skipped": [],
        "not_started": [],
    }
    domain_calls = [
        entry
        for entry in accept_client.transport.sent
        if entry[0] not in ("wire.version", "rpc.catalog", "resources.versions")
    ]
    assert domain_calls == [
        ("tab.get_analyze_result", {"tab_id": "t"}),
        ("tab.get_post_analyze_result", {"tab_id": "t"}),
        ("tab.writeback_preview", {"tab_id": "t", "subtab_id": "analysis"}),
        (
            "tab.writeback_write",
            {
                "tab_id": "t",
                "subtab_id": "analysis",
                "write": [{"id": "md-1"}, {"id": "wf-1"}],
            },
        ),
        ("tab.writeback_preview", {"tab_id": "t", "subtab_id": "post_analysis"}),
        (
            "tab.writeback_write",
            {
                "tab_id": "t",
                "subtab_id": "post_analysis",
                "write": [{"id": "md-1"}, {"id": "wf-1"}],
            },
        ),
    ]


@pytest.mark.parametrize(
    "primary_exists,post_exists", [(True, False), (False, True), (False, False)]
)
def test_accept_skips_missing_results(accept_client, primary_exists, post_exists):
    accept_client.transport.replies["tab.get_analyze_result"] = {
        "ok": True,
        "result": {"summary": {} if primary_exists else None},
    }
    accept_client.transport.replies["tab.get_post_analyze_result"] = {
        "ok": True,
        "result": {"summary": {} if post_exists else None},
    }
    present = [
        (stage, pane)
        for stage, pane, exists in (
            ("primary", "analysis", primary_exists),
            ("post", "post_analysis", post_exists),
        )
        if exists
    ]
    result = accept_client.call("accept", {"tab": "t"})
    assert result["status"] == "finished"
    assert result["skipped"] == [
        stage
        for stage, exists in (("primary", primary_exists), ("post", post_exists))
        if not exists
    ]
    assert [pane["stage"] for pane in result["completed"]] == [
        stage for stage, _ in present
    ]
    assert result["not_started"] == []
    assert [
        params["subtab_id"]
        for method, params in accept_client.transport.sent
        if method.startswith("tab.writeback_")
    ] == [pane for _, pane in present for _ in range(2)]


@pytest.mark.parametrize(
    "preview",
    [
        {"has_draft": False, "items": []},
        {"has_draft": True, "items": []},
    ],
)
def test_accept_skips_absent_or_empty_drafts(accept_client, preview):
    accept_client.transport.replies["tab.writeback_preview"] = {
        "ok": True,
        "result": preview,
    }
    assert accept_client.call("accept", {"tab": "t"}) == {
        "tab": "t",
        "status": "finished",
        "completed": [],
        "skipped": ["primary", "post"],
        "not_started": [],
    }
    assert all(
        method != "tab.writeback_write" for method, _ in accept_client.transport.sent
    )


@pytest.mark.parametrize("reason", ["no_read", "stale_version"])
@pytest.mark.parametrize(
    "method,stage,expected_calls",
    [
        ("tab.get_analyze_result", "primary", 1),
        ("tab.get_post_analyze_result", "post", 2),
        ("tab.writeback_preview", "primary", 3),
        ("tab.writeback_write", "primary", 4),
        ("tab.writeback_preview", "post", 5),
        ("tab.writeback_write", "post", 6),
    ],
)
def test_accept_stops_at_first_rpc_error_with_confirmed_progress(
    accept_client, method, stage, expected_calls, reason
):
    previous_reply = accept_client.transport.replies[method]
    failed_pane = "analysis" if stage == "primary" else "post_analysis"

    def reply(params):
        if params.get("subtab_id", failed_pane) == failed_pane:
            return {
                "ok": False,
                "error": {
                    "code": "precondition_failed",
                    "reason": reason,
                    "message": "guard denied",
                },
            }
        return previous_reply(params) if callable(previous_reply) else previous_reply

    accept_client.transport.replies[method] = reply
    result = accept_client.call("accept", {"tab": "t"})
    assert result["status"] == "failed"
    assert result["failed_stage"] == stage
    assert result["error"]["code"] == "precondition_failed"
    assert result["error"]["reason"] == reason
    assert result["error"]["message"] == (
        "GUI Error (precondition_failed): guard denied"
        if reason == "no_read"
        else "GUI Error (PRECONDITION_FAILED): a resource changed since your last read; "
        "review then retry"
    )
    assert result["failed_stage_may_have_partial_writes"] is (
        method == "tab.writeback_write"
    )
    assert result["skipped"] == []
    assert [pane["stage"] for pane in result["completed"]] == (
        ["primary"] if expected_calls >= 5 else []
    )
    assert result["not_started"] == (
        [] if expected_calls >= 5 else ["post" if stage == "primary" else "primary"]
    )
    expected_sequence = [
        "tab.get_analyze_result",
        "tab.get_post_analyze_result",
        "tab.writeback_preview",
        "tab.writeback_write",
        "tab.writeback_preview",
        "tab.writeback_write",
    ]
    assert [
        method
        for method, _ in accept_client.transport.sent
        if method not in ("wire.version", "rpc.catalog", "resources.versions")
    ] == expected_sequence[:expected_calls]


def test_accept_reports_partial_write_after_skipped_primary(accept_client):
    accept_client.transport.replies["tab.get_analyze_result"] = {
        "ok": True,
        "result": {"summary": None},
    }
    destination = {"freq": 1}

    def write_then_fail(params):
        destination["freq"] = 2
        return {
            "ok": False,
            "error": {"code": "internal_error", "message": "context apply interrupted"},
        }

    accept_client.transport.replies["tab.writeback_write"] = write_then_fail
    result = accept_client.call("accept", {"tab": "t"})
    assert result == {
        "tab": "t",
        "status": "failed",
        "completed": [],
        "skipped": ["primary"],
        "not_started": [],
        "failed_stage": "post",
        "error": {
            "code": "internal_error",
            "reason": None,
            "message": "GUI Error (internal_error): context apply interrupted",
        },
        "failed_stage_may_have_partial_writes": True,
    }
    assert destination == {"freq": 2}
    assert (
        sum(
            method == "tab.writeback_write"
            for method, _ in accept_client.transport.sent
        )
        == 1
    )


def test_accept_uses_current_results_and_drafts_on_each_call(accept_client):
    accept_client.call("accept", {"tab": "t"})
    accept_client.transport.sent.clear()
    accept_client.transport.replies["tab.get_post_analyze_result"] = {
        "ok": True,
        "result": {"summary": None},
    }
    accept_client.transport.replies["tab.writeback_preview"] = {
        "ok": True,
        "result": {"has_draft": True, "items": [{"id": "md-new"}]},
    }
    result = accept_client.call("accept", {"tab": "t"})
    assert result["skipped"] == ["post"]
    assert [
        (params["subtab_id"], params["write"])
        for method, params in accept_client.transport.sent
        if method == "tab.writeback_write"
    ] == [("analysis", [{"id": "md-new"}])]
    assert result["completed"][0]["written"][0]["id"] == "md-new"


def test_accept_does_not_swallow_unexpected_errors(accept_client):
    accept_client.transport.replies["tab.get_analyze_result"] = {
        "ok": True,
        "result": {},
    }
    with pytest.raises(KeyError, match="summary"):
        accept_client.call("accept", {"tab": "t"})


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"tab": ""},
        {"tab": 1},
        {"tab": "t", "stage": "post"},
        {"tab": "t", "items": []},
    ],
)
def test_accept_rejects_invalid_arguments_before_transport(accept_client, arguments):
    with pytest.raises(ValueError, match="tab"):
        accept_client.call("accept", arguments)
    assert accept_client.transport.sent == []
