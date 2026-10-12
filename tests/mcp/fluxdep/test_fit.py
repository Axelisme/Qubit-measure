"""Fit decisions compose native GUI search/query contracts without local state."""

import pytest
from zcu_tools.mcp.core.reply import ToolReply

from tests.mcp.fluxdep._support import Client, GuiResponse


def test_database_info_returns_native_ranges_without_search(client: Client) -> None:
    info = {
        "path": "/synthetic.h5",
        "entries": 4096,
        "levels": 15,
        "EJ": [1.0, 20.0],
        "EC": [0.1, 4.0],
        "EL": [0.05, 3.0],
        "flux": [0.0, 1.0],
    }
    client.transport.replies.append(GuiResponse(result=info))
    tool = client.server.tools.get("fluxdep_database_info")
    assert tool is not None, "The database decision must be available"
    result = tool["handler"]({"path": "/synthetic.h5"})
    data = result.data if isinstance(result, ToolReply) else result
    assert data == info
    assert [(r["method"], r["params"]) for r in client.transport.requests] == [
        ("fit.database_info", {"path": "/synthetic.h5"})
    ]


@pytest.fixture
def fit_inputs() -> dict[str, object]:
    """Current GUI choices, independent of any MCP-side defaults."""
    return {
        "search_db": "/previous.h5",
        "EJb": [2.0, 18.0],
        "ECb": [0.2, 3.5],
        "ELb": [0.06, 2.8],
        "transitions": {"transitions": [[0, 1]], "blue side": [[0, 2]]},
        "r_f": 7.25,
        "sample_f": 9.0,
    }


@pytest.mark.parametrize("status", ["failed", "cancelled"])
@pytest.mark.parametrize("frequency", [{}, {"r_f": None, "sample_f": None}])
def test_fit_overlays_current_inputs_waits_and_preserves_unsuccessful_outcome(
    client: Client,
    fit_inputs: dict[str, object],
    status: str,
    frequency: dict[str, object],
) -> None:
    caller = {"EJb": [1.0, 20.0], **frequency}
    client.transport.replies.extend(
        [
            GuiResponse(
                result={**fit_inputs, "has_result": True, "params": {"EJ": 3.0}}
            ),
            GuiResponse(result={"updated": True}),
            GuiResponse(result={"token": 42}),
            GuiResponse(
                result={
                    "token": 42,
                    "reason": "timeout",
                    "outcome": None,
                    "feedback": None,
                }
            ),
            GuiResponse(
                result={
                    "token": 42,
                    "reason": "completed",
                    "outcome": {"status": status, "error": "no feasible solution"},
                    "feedback": None,
                }
            ),
            GuiResponse(result={**fit_inputs, "has_result": False, "params": None}),
        ]
    )
    tool = client.server.tools.get("fluxdep_fit")
    assert tool is not None, "The composed fit decision must be available"
    result = tool["handler"](caller)
    data = result.data if isinstance(result, ToolReply) else result
    assert data == {"status": status, "error": "no feasible solution"}
    if isinstance(result, ToolReply):
        assert result.images == ()
        assert not result.is_error
    assert [(r["method"], r["params"]) for r in client.transport.requests] == [
        ("fit.result", {}),
        ("fit.set_params", {**fit_inputs, **caller}),
        ("fit.search", {}),
        ("operation.await", {"token": 42, "timeout": 30}),
        ("operation.await", {"token": 42, "timeout": 30}),
        ("fit.result", {}),
    ]
