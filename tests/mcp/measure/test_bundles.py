"""Tab bundles preserve RPC ordering and return only settled products."""

from pathlib import Path
from typing import Any

import pytest

from ._support import MeasureClient, make_client


def stage_rpc(method: str, params: dict[str, Any]) -> dict[str, Any]:
    if method == "tab.get_figure":
        return {"bytes": 9, "saved_to": params["out_path"]}
    replies: dict[str, dict[str, Any]] = {
        "tab.snapshot": {
            "tabs": [
                {
                    "editor_id": "stage-ed",
                    "interaction": {
                        "is_running": False,
                        "has_run_result": True,
                    },
                }
            ]
        },
        "tab.set_cfg": {"valid": True, "removed": [], "added": []},
        "tab.run_start": {},
        "tab.analyze": {},
        "tab.get_analyze_params": {"analyze_params": {"smooth": 1}},
        "tab.get_analyze_result": {"summary": {"t1": 12.3}},
        "tab.writeback_preview": {
            "has_draft": True,
            "items": [{"id": "md-0", "target_name": "q_f"}],
        },
    }
    return replies[method]


@pytest.fixture
def client(tmp_path: Path) -> MeasureClient:
    return make_client(tmp_path, stage_rpc)


def calls(client: MeasureClient) -> list[tuple[str, dict[str, Any]]]:
    return [
        (method, params)
        for method, params in client.transport.sent
        if method != "resources.versions"
    ]


@pytest.mark.parametrize(
    "edits",
    [
        None,
        [
            {"path": "sweep.type", "value": "gauss"},
            {"path": "reps", "value": 100},
            {"path": "gain", "value": 0.2},
        ],
    ],
)
def test_run_applies_ordered_edits_then_folds_one_figure_and_analyze_params(
    client: MeasureClient,
    edits: list[dict[str, Any]] | None,
) -> None:
    arguments: dict[str, Any] = {"tab_id": "stage-tab"}
    if edits is not None:
        arguments["edits"] = edits
    result = client.call("gui_tab_run", arguments)
    methods = [method for method, _ in calls(client)]
    assert methods == (["tab.set_cfg"] if edits is not None else []) + [
        "tab.run_start",
        "tab.snapshot",
        "tab.get_figure",
        "tab.get_analyze_params",
    ]
    if edits is not None:
        assert calls(client)[0][1] == {"tab_id": "stage-tab", "edits": edits}
        assert type(calls(client)[0][1]["edits"][1]["value"]) is int
        assert type(calls(client)[0][1]["edits"][2]["value"]) is float
    assert result["status"] == "finished"
    assert result["figure"].endswith("measure_fig_stage-tab_run.png")
    assert result["analyze_params"] == {"smooth": 1}
    figure = next(
        params for method, params in calls(client) if method == "tab.get_figure"
    )
    assert figure["subtab_id"] == "run"


def test_run_rejects_unordered_edits_before_any_rpc(client: MeasureClient) -> None:
    with pytest.raises(ValueError, match="ordered list"):
        client.call("gui_tab_run", {"tab_id": "t", "edits": {"reps": 100}})
    assert client.transport.sent == []


@pytest.mark.parametrize(
    ("tool", "start"),
    [
        ("gui_tab_run_start", "tab.run_start"),
        ("gui_tab_run", "tab.run_start"),
        ("gui_tab_analyze_review", "tab.analyze"),
        ("gui_tab_analyze_start", "tab.analyze"),
        ("gui_tab_post_analyze_start", "tab.post_analyze"),
    ],
)
def test_pending_operations_do_not_read_unsettled_products(
    client: MeasureClient,
    tool: str,
    start: str,
) -> None:
    client.transport.replies[start] = {"ok": True, "result": {"operation_id": 9}}
    client.transport.replies["operation.await"] = {
        "ok": False,
        "error": {
            "code": "timeout",
            "message": "still running",
        },
    }
    result = client.call(tool, {"tab_id": "t"})
    assert result["status"] == "pending"
    assert result["handle"] == 9
    assert "figure" not in result
    assert "analyze_params" not in result
    assert "writeback_preview" not in result
    assert [method for method, _ in calls(client)] == [start, "operation.await"]
    if tool in {"gui_tab_run", "gui_tab_analyze_review"}:
        assert "figure" in result["owed"]


@pytest.mark.parametrize(
    ("tool", "method", "product", "pane"),
    [
        ("gui_tab_analyze_start", "tab.analyze", "tab.get_analyze_result", "analysis"),
        (
            "gui_tab_post_analyze_start",
            "tab.post_analyze",
            "tab.get_post_analyze_result",
            "post_analysis",
        ),
    ],
)
def test_analysis_start_returns_its_own_products_without_writeback(
    client: MeasureClient,
    tool: str,
    method: str,
    product: str,
    pane: str,
) -> None:
    client.transport.replies[method] = {"ok": True, "result": {}}
    client.transport.replies[product] = {"ok": True, "result": {"summary": {"t1": 5.0}}}
    result = client.call(tool, {"tab_id": "t", "updates": {"smooth": 3}})
    assert result["status"] == "finished"
    assert result["summary"] == {"t1": 5.0}
    assert result["figure"].endswith(f"measure_fig_t_{pane}.png")
    assert "writeback_preview" not in result
    assert [name for name, _ in calls(client)] == [method, product, "tab.get_figure"]
    assert calls(client)[0][1] == {"tab_id": "t", "updates": {"smooth": 3}}
    assert calls(client)[-1][1]["subtab_id"] == pane


def test_analyze_returns_summary_and_one_pane_figure(client: MeasureClient) -> None:
    client.transport.replies["tab.analyze"] = {
        "ok": True,
        "result": {"operation_id": 9},
    }
    client.transport.replies["operation.await"] = {
        "ok": True,
        "result": {"status": "finished"},
    }
    result = client.call(
        "gui_tab_analyze_review",
        {"tab_id": "az-1", "updates": {"smooth": 3}, "wait_seconds": 2.0},
    )
    assert result["status"] == "finished"
    assert result["summary"] == {"t1": 12.3}
    assert result["figure"].endswith("measure_fig_az-1_analysis.png")
    assert calls(client)[0] == (
        "tab.analyze",
        {"tab_id": "az-1", "updates": {"smooth": 3}},
    )
    assert calls(client)[1] == ("operation.await", {"operation_id": 9, "timeout": 2.0})
    figures = [params for method, params in calls(client) if method == "tab.get_figure"]
    assert len(figures) == 1
    assert figures[0]["subtab_id"] == "analysis"
    assert result["writeback_preview"] == {
        "has_draft": True,
        "items": [{"id": "md-0", "target_name": "q_f"}],
    }
    assert (
        "tab.writeback_preview",
        {"tab_id": "az-1", "subtab_id": "analysis"},
    ) in calls(client)


@pytest.mark.parametrize(
    ("tool", "failed_rpc", "field"),
    [
        ("gui_tab_run", "tab.get_figure", "figure"),
        ("gui_tab_run", "tab.get_analyze_params", "analyze_params"),
        ("gui_tab_analyze_review", "tab.writeback_preview", "writeback_preview"),
    ],
)
def test_optional_product_read_failure_does_not_mask_a_finished_operation(
    client: MeasureClient,
    tool: str,
    failed_rpc: str,
    field: str,
) -> None:
    client.transport.replies[failed_rpc] = {
        "ok": False,
        "error": {"code": "internal", "message": "read failed"},
    }
    result = client.call(tool, {"tab_id": "t"})
    assert result["status"] == "finished"
    if field == "writeback_preview":
        assert field not in result
    else:
        assert result[field] is None


@pytest.mark.parametrize("tool", ["gui_op_wait", "gui_op_poll"])
def test_generic_wait_and_poll_do_not_render_tab_products(
    client: MeasureClient, tool: str
) -> None:
    client.transport.replies["operation.await"] = {
        "ok": True,
        "result": {"status": "finished"},
    }
    result = client.call(tool, {"handle": 3})
    assert result["status"] == "finished"
    assert "figure" not in result
    assert [method for method, _ in calls(client)] == ["operation.await"]


@pytest.mark.parametrize(
    ("tool", "method", "arguments", "result"),
    [
        ("gui_tab_new", "tab.new", {"adapter_name": "fake/freq"}, {"tab_id": "t"}),
        ("gui_context_switch", "context.use", {"label": "ctx1"}, {}),
        ("gui_device_snapshot", "device.snapshot", {"name": "bias"}, {}),
        (
            "gui_adapter_guide",
            "adapter.guide",
            {"adapter_name": "amp_rabi"},
            {"guide": {"behavior": "measures X"}},
        ),
        (
            "gui_tab_writeback_apply",
            "tab.writeback_apply",
            {"tab_id": "t", "subtab_id": "analysis"},
            {"applied_ids": ["md-0", "ml-1"]},
        ),
        (
            "gui_tab_save_data",
            "tab.save_data",
            {"tab_id": "t", "data_path": "data.h5"},
            {"data_path": "data.h5"},
        ),
        (
            "gui_tab_save_image",
            "tab.save_image",
            {"tab_id": "t", "subtab_id": "post_analysis", "image_path": "image.png"},
            {"image_path": "image.png"},
        ),
    ],
)
def test_generated_actions_forward_without_bundle_side_effects(
    client: MeasureClient,
    tool: str,
    method: str,
    arguments: dict[str, Any],
    result: dict[str, Any],
) -> None:
    client.transport.replies[method] = {"ok": True, "result": result}
    assert client.call(tool, arguments) == result
    sent = calls(client)
    assert [name for name, _ in sent] == [method]
    assert {
        key: value for key, value in sent[0][1].items() if key != "expected_versions"
    } == arguments
