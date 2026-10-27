"""Fit export forwards GUI authority and projects its captured output receipt."""

import pytest
from zcu_tools.mcp.core.reply import ToolReply
from zcu_tools.mcp.core.rpc import GuiInvocationError

from tests.mcp.fluxdep._support import Client, GuiError, GuiResponse


@pytest.mark.parametrize("predictor_written", [True, False])
def test_fit_export_returns_only_gui_receipt_without_other_reads(
    client: Client, predictor_written: bool
) -> None:
    arguments: dict[str, object] = {}
    receipt = {
        "savepath": "/resolved/params.json" if predictor_written else None,
        "predictor_writeback": {
            "status": "written" if predictor_written else "skipped",
            "device": "bias",
            "reason": None
            if predictor_written
            else "No snapshot; load native data with a snapshot.",
        },
        "params": {"EJ": 2.741234567, "EC": 1.420123456, "EL": 0.148123456},
        "alignment_from": "empty-first-aligned",
        "component": "Q2",
        "analysis_id": "analysis-uuid",
        "flux_writeback": {
            "status": "skipped",
            "device": "bias",
            "reason": "Missing other device values; use fluxdep measurement writeback.",
        },
    }
    client.transport.replies.append(
        GuiResponse(result={**receipt, "unprojected": [1, 2, 3]})
    )
    tool = client.server.tools.get("fluxdep_fit_export_params")
    assert tool is not None, "The accepted fit must be exportable through MCP"
    reply = tool["handler"](arguments)
    assert isinstance(reply, ToolReply)
    assert reply.data == receipt
    assert reply.images == ()
    assert not reply.is_error
    assert [(r["method"], r["params"]) for r in client.transport.requests] == [
        ("fit.export_params", arguments)
    ]


@pytest.mark.parametrize(
    "error",
    [
        GuiError("precondition_failed", "fit is stale: filter changed", "fit_stale"),
        GuiError("precondition_failed", "no fit result to export", "no_fit_result"),
        GuiError("precondition_failed", "not the holder", "not_holder"),
        GuiError("controller_error", "Permission denied: params.json"),
    ],
    ids=["stale", "no-result", "not-holder", "io-failure"],
)
def test_fit_export_preserves_native_rejection_without_retry(
    client: Client, error: GuiError
) -> None:
    client.transport.replies.append(GuiResponse(error=error))
    tool = client.server.tools.get("fluxdep_fit_export_params")
    assert tool is not None, "The accepted fit must be exportable through MCP"
    with pytest.raises(GuiInvocationError) as caught:
        tool["handler"]({})
    assert (caught.value.code, caught.value.message, caught.value.reason) == (
        error.code,
        error.message,
        error.reason,
    )
    assert [(r["method"], r["params"]) for r in client.transport.requests] == [
        ("fit.export_params", {})
    ]
