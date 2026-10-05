"""Standalone apply_writeback selects current drafts without changing observations."""

import pytest

from ._support import make_client


@pytest.fixture
def apply_writeback_client(tmp_path):
    client = make_client(tmp_path)
    for method in ("tab.get_analyze_result", "tab.get_post_analyze_result"):
        client.transport.replies[method] = {"ok": True, "result": {"summary": {}}}
    client.transport.replies["tab.writeback_preview"] = lambda params: {
        "ok": True,
        "result": {
            "has_draft": True,
            "items": [
                {
                    "id": item_id,
                    "target_name": f"{params['subtab_id']}.{item_id}",
                    "selected": selected,
                }
                for item_id, selected in (("md-1", False), ("wf-1", True))
            ],
            "destination_context": {},
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


@pytest.mark.parametrize("r2", [-0.25, 0.0])
def test_apply_writeback_writes_whole_drafts_despite_low_fit_quality(
    apply_writeback_client, r2
):
    for method in ("tab.get_analyze_result", "tab.get_post_analyze_result"):
        apply_writeback_client.transport.replies[method] = {
            "ok": True,
            "result": {
                "summary": {
                    "fidelity": 0.98,
                    "fit_quality": {
                        "fit": {
                            "r2": r2,
                            "normalized_residual_rms": 0.31,
                            "relative_parameter_errors": {"decay_time": None},
                            "invalid": [
                                {
                                    "path": "summary.fit_quality.fit.relative_parameter_errors.decay_time",
                                    "reason": "covariance_unavailable",
                                }
                            ],
                        }
                    },
                },
                "invalid": [],
            },
        }
    result = apply_writeback_client.call("apply_writeback", {"tab": "t"})
    assert result["status"] == "finished"
    assert result["skipped"] == result["not_started"] == []
    assert [
        (stage["stage"], [item["id"] for item in stage["written"]])
        for stage in result["completed"]
    ] == [("primary", ["md-1", "wf-1"]), ("post", ["md-1", "wf-1"])]


def test_apply_writeback_writes_all_ids_in_each_pane_without_hidden_reads(
    apply_writeback_client,
):
    result = apply_writeback_client.call("apply_writeback", {"tab": "t"})
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
        for entry in apply_writeback_client.transport.sent
        if entry[0] not in ("wire.version", "rpc.catalog", "resources.versions")
    ]
    assert domain_calls == [
        ("tab.get_analyze_result", {"tab_id": "t"}),
        ("tab.get_post_analyze_result", {"tab_id": "t"}),
        ("tab.writeback_preview", {"tab_id": "t", "subtab_id": "analysis"}),
        ("tab.writeback_preview", {"tab_id": "t", "subtab_id": "post_analysis"}),
        (
            "tab.writeback_write",
            {
                "tab_id": "t",
                "subtab_id": "analysis",
                "write": [{"id": "md-1"}, {"id": "wf-1"}],
            },
        ),
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
def test_apply_writeback_skips_missing_results(
    apply_writeback_client, primary_exists, post_exists
):
    apply_writeback_client.transport.replies["tab.get_analyze_result"] = {
        "ok": True,
        "result": {"summary": {} if primary_exists else None},
    }
    apply_writeback_client.transport.replies["tab.get_post_analyze_result"] = {
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
    result = apply_writeback_client.call("apply_writeback", {"tab": "t"})
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
        for method, params in apply_writeback_client.transport.sent
        if method.startswith("tab.writeback_")
    ] == [pane for _ in range(2) for _, pane in present]


@pytest.mark.parametrize(
    "preview",
    [
        {"has_draft": False, "items": [], "destination_context": {}},
        {"has_draft": True, "items": [], "destination_context": {}},
    ],
)
def test_apply_writeback_skips_absent_or_empty_drafts(apply_writeback_client, preview):
    apply_writeback_client.transport.replies["tab.writeback_preview"] = {
        "ok": True,
        "result": preview,
    }
    assert apply_writeback_client.call("apply_writeback", {"tab": "t"}) == {
        "tab": "t",
        "status": "finished",
        "completed": [],
        "skipped": ["primary", "post"],
        "not_started": [],
    }
    assert all(
        method != "tab.writeback_write"
        for method, _ in apply_writeback_client.transport.sent
    )


@pytest.mark.parametrize("reason", ["no_read", "stale_version"])
@pytest.mark.parametrize(
    "method,stage,expected_calls",
    [
        ("tab.get_analyze_result", "primary", 1),
        ("tab.get_post_analyze_result", "post", 2),
        ("tab.writeback_preview", "primary", 3),
        ("tab.writeback_write", "primary", 5),
        ("tab.writeback_preview", "post", 4),
        ("tab.writeback_write", "post", 6),
    ],
)
def test_apply_writeback_stops_at_first_rpc_error_with_confirmed_progress(
    apply_writeback_client, method, stage, expected_calls, reason
):
    previous_reply = apply_writeback_client.transport.replies[method]
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

    apply_writeback_client.transport.replies[method] = reply
    result = apply_writeback_client.call("apply_writeback", {"tab": "t"})
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
        ["primary"] if method == "tab.writeback_write" and stage == "post" else []
    )
    assert result["not_started"] == (
        []
        if method == "tab.writeback_write" and stage == "post"
        else ["post" if stage == "primary" else "primary"]
    )
    expected_sequence = [
        "tab.get_analyze_result",
        "tab.get_post_analyze_result",
        "tab.writeback_preview",
        "tab.writeback_preview",
        "tab.writeback_write",
        "tab.writeback_write",
    ]
    assert [
        method
        for method, _ in apply_writeback_client.transport.sent
        if method not in ("wire.version", "rpc.catalog", "resources.versions")
    ] == expected_sequence[:expected_calls]


def test_apply_writeback_reports_partial_write_after_skipped_primary(
    apply_writeback_client,
):
    apply_writeback_client.transport.replies["tab.get_analyze_result"] = {
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

    apply_writeback_client.transport.replies["tab.writeback_write"] = write_then_fail
    result = apply_writeback_client.call("apply_writeback", {"tab": "t"})
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
            for method, _ in apply_writeback_client.transport.sent
        )
        == 1
    )


def test_apply_writeback_uses_current_results_and_drafts_on_each_call(
    apply_writeback_client,
):
    apply_writeback_client.call("apply_writeback", {"tab": "t"})
    apply_writeback_client.transport.sent.clear()
    apply_writeback_client.transport.replies["tab.get_post_analyze_result"] = {
        "ok": True,
        "result": {"summary": None},
    }
    apply_writeback_client.transport.replies["tab.writeback_preview"] = {
        "ok": True,
        "result": {
            "has_draft": True,
            "items": [{"id": "md-new", "target_name": "new_frequency"}],
            "destination_context": {},
        },
    }
    result = apply_writeback_client.call("apply_writeback", {"tab": "t"})
    assert result["skipped"] == ["post"]
    assert [
        (params["subtab_id"], params["write"])
        for method, params in apply_writeback_client.transport.sent
        if method == "tab.writeback_write"
    ] == [("analysis", [{"id": "md-new"}])]
    assert result["completed"][0]["written"][0]["id"] == "md-new"


def test_apply_writeback_does_not_swallow_unexpected_errors(apply_writeback_client):
    apply_writeback_client.transport.replies["tab.get_analyze_result"] = {
        "ok": True,
        "result": {},
    }
    with pytest.raises(KeyError, match="summary"):
        apply_writeback_client.call("apply_writeback", {"tab": "t"})


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"tab": ""},
        {"tab": 1},
        {"tab": "t", "stage": "post"},
        {"tab": "t", "items": "frequency"},
        {"tab": "t", "items": [1]},
        {"tab": "t", "items": [""]},
        {"tab": "t", "items": ["duplicate", "duplicate"]},
    ],
)
def test_apply_writeback_rejects_invalid_arguments_before_transport(
    apply_writeback_client, arguments
):
    with pytest.raises(ValueError, match="tab|items|Duplicate"):
        apply_writeback_client.call("apply_writeback", arguments)
    assert apply_writeback_client.transport.sent == []


@pytest.mark.parametrize(
    "items", [None, [], ["analysis.md-1"], ["post_analysis.wf-1", "analysis.md-1"]]
)
def test_apply_writeback_selects_stable_names_in_pane_order(
    apply_writeback_client, items
):
    result = apply_writeback_client.call(
        "apply_writeback", {"tab": "t", "items": items}
    )
    expected = {
        "primary": ["md-1", "wf-1"]
        if items is None
        else [item for item in ("md-1", "wf-1") if f"analysis.{item}" in items],
        "post": ["md-1", "wf-1"]
        if items is None
        else [item for item in ("md-1", "wf-1") if f"post_analysis.{item}" in items],
    }
    assert result["status"] == "finished"
    assert result["not_started"] == []
    assert result["skipped"] == [stage for stage, ids in expected.items() if not ids]
    assert [
        (stage["stage"], [item["id"] for item in stage["written"]])
        for stage in result["completed"]
    ] == [(stage, ids) for stage, ids in expected.items() if ids]


@pytest.mark.parametrize("ambiguity", ["unknown", "across_panes", "within_pane"])
def test_apply_writeback_rejects_names_before_any_write(
    apply_writeback_client, ambiguity
):
    arguments = {"tab": "t", "items": ["absent"]}
    if ambiguity != "unknown":
        arguments = {"tab": "t"}
        apply_writeback_client.transport.replies["tab.writeback_preview"] = {
            "ok": True,
            "result": {
                "has_draft": True,
                "destination_context": {},
                "items": [{"id": "x", "target_name": "same"}]
                * (2 if ambiguity == "within_pane" else 1),
            },
        }
    with pytest.raises(ValueError, match="Unknown|Ambiguous"):
        apply_writeback_client.call("apply_writeback", arguments)
    assert not any(
        method == "tab.writeback_write"
        for method, _ in apply_writeback_client.transport.sent
    )


def test_apply_writeback_keeps_selected_confirmed_prefix_on_post_failure(
    apply_writeback_client,
):
    original_write = apply_writeback_client.transport.replies["tab.writeback_write"]
    uncertain_effects = []

    def write(params):
        if params["subtab_id"] == "post_analysis":
            uncertain_effects.extend(params["write"])
            return {
                "ok": False,
                "error": {
                    "code": "internal_error",
                    "message": "write interrupted",
                },
            }
        return original_write(params)

    apply_writeback_client.transport.replies["tab.writeback_write"] = write
    result = apply_writeback_client.call(
        "apply_writeback",
        {
            "tab": "t",
            "items": ["analysis.md-1", "post_analysis.wf-1"],
        },
    )
    assert result["status"] == "failed"
    assert result["failed_stage"] == "post"
    assert result["failed_stage_may_have_partial_writes"] is True
    assert result["not_started"] == []
    assert result["skipped"] == []
    assert [
        (stage["stage"], [item["id"] for item in stage["written"]])
        for stage in result["completed"]
    ] == [("primary", ["md-1"])]
    assert uncertain_effects == [{"id": "wf-1"}]
    assert [
        (params["subtab_id"], params["write"])
        for method, params in apply_writeback_client.transport.sent
        if method == "tab.writeback_write"
    ] == [
        ("analysis", [{"id": "md-1"}]),
        ("post_analysis", [{"id": "wf-1"}]),
    ]


@pytest.mark.parametrize("empty_stage", ["primary", "post"])
def test_apply_writeback_retains_confirmed_empty_stage_on_later_failure(
    apply_writeback_client, empty_stage
):
    preview = apply_writeback_client.transport.replies["tab.writeback_preview"]

    def read(params):
        if params["subtab_id"] == "analysis" and empty_stage == "primary":
            return {
                "ok": True,
                "result": {
                    "has_draft": True,
                    "items": [],
                    "destination_context": {},
                },
            }
        if params["subtab_id"] == "post_analysis":
            if empty_stage == "post":
                return {
                    "ok": True,
                    "result": {
                        "has_draft": False,
                        "items": [],
                        "destination_context": {},
                    },
                }
            return {
                "ok": False,
                "error": {
                    "code": "internal_error",
                    "message": "preview failed",
                },
            }
        return preview(params)

    apply_writeback_client.transport.replies["tab.writeback_preview"] = read
    if empty_stage == "post":
        apply_writeback_client.transport.replies["tab.writeback_write"] = {
            "ok": False,
            "error": {"code": "internal_error", "message": "write failed"},
        }
    result = apply_writeback_client.call("apply_writeback", {"tab": "t"})
    assert result["status"] == "failed"
    assert result["completed"] == []
    assert result["skipped"] == [empty_stage]
    assert result["not_started"] == []
    assert result["failed_stage"] == ("post" if empty_stage == "primary" else "primary")
    assert result["failed_stage_may_have_partial_writes"] is (empty_stage == "post")


@pytest.mark.parametrize(
    "pane,stage", [("analysis", "primary"), ("post_analysis", "post")]
)
def test_apply_writeback_reports_invalid_native_name_at_its_stage(
    apply_writeback_client, pane, stage
):
    preview = apply_writeback_client.transport.replies["tab.writeback_preview"]

    def read(params):
        result = preview(params)
        if params["subtab_id"] == pane:
            result["result"]["items"][0]["target_name"] = ""
        return result

    apply_writeback_client.transport.replies["tab.writeback_preview"] = read
    result = apply_writeback_client.call("apply_writeback", {"tab": "t"})
    assert result["status"] == "failed"
    assert result["failed_stage"] == stage
    assert result["error"]["reason"] == "incompatible_wire"
    assert result["failed_stage_may_have_partial_writes"] is False
    assert result["completed"] == []
    assert not any(
        method == "tab.writeback_write"
        for method, _ in apply_writeback_client.transport.sent
    )
