"""Fit decisions compose native GUI search/query contracts without local state."""

import base64
from io import BytesIO

import pytest
from PIL import Image
from zcu_tools.mcp.core.reply import ToolReply
from zcu_tools.mcp.core.rpc import GuiInvocationError

from tests.mcp.fluxdep._support import Client, GuiError, GuiResponse


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


def tool_reply(value: ToolReply | dict[str, object]) -> ToolReply:
    """Normalize public handler containers while preserving error/image content."""
    return value if isinstance(value, ToolReply) else ToolReply(value)


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


@pytest.fixture
def recorded_fit(fit_inputs: dict[str, object]) -> dict[str, object]:
    """Recorded native quality includes duplicated labels and an unused line."""
    return {
        **fit_inputs,
        "search_db": "/accepted.h5",
        "has_result": True,
        "params": {"EJ": 2.741, "EC": 1.42, "EL": 0.148},
        "quality": {
            "mean": 0.11234,
            "outliers": 1,
            "outlier_threshold": 0.33702,
            "per_transition": [
                {
                    "transition": "transitions 0-1",
                    "points": 2,
                    "mean": 0.09549,
                    "max": 0.21023,
                },
                {
                    "transition": "transitions 0-1",
                    "points": 1,
                    "mean": 0.45549,
                    "max": 0.45549,
                },
                {"transition": "blue side 0-2", "points": 0, "mean": None, "max": None},
            ],
        },
        "stale": False,
        "stale_reasons": [],
        "point_transition_indices": [0, 0, 1],
        "residuals": [0.1, 0.2, 0.4],
    }


def queue_finished(
    client: Client, current: dict[str, object], recorded: dict[str, object]
) -> None:
    """Supply the native search receipts up to the final recorded result."""
    client.transport.replies.extend(
        [
            GuiResponse(result=current),
            GuiResponse(result={"updated": True}),
            GuiResponse(result={"token": 19}),
            GuiResponse(
                result={
                    "token": 19,
                    "reason": "timeout",
                    "outcome": None,
                    "feedback": None,
                }
            ),
            GuiResponse(
                result={
                    "token": 19,
                    "reason": "completed",
                    "outcome": {"status": "finished", "error": None},
                    "feedback": None,
                }
            ),
            GuiResponse(result=recorded),
        ]
    )


@pytest.mark.parametrize("stale,image", [(False, True), (False, False), (True, True)])
def test_fit_result_projects_freshness_and_only_renders_fresh_requested_evidence(
    client: Client,
    recorded_fit: dict[str, object],
    stale: bool,
    image: bool,
) -> None:
    recorded_fit["stale"] = stale
    recorded_fit["stale_reasons"] = (
        ["one re-aligned", "filter changed"] if stale else []
    )
    client.transport.replies.append(GuiResponse(result=recorded_fit))
    png = b""
    if not stale and image:
        buffer = BytesIO()
        with Image.new("RGB", (12, 8), "red") as evidence:
            evidence.save(buffer, format="PNG")
        png = buffer.getvalue()
        client.transport.replies.append(
            GuiResponse(
                result={
                    "png_b64": base64.b64encode(png).decode("ascii"),
                    "bytes": len(png),
                }
            )
        )
    tool = client.server.tools.get("fluxdep_fit_result")
    if tool is None:
        pytest.fail("missing composed fit_result decision tool")
    reply = tool_reply(tool["handler"]({"image": image}))
    assert reply.data["stale"] is stale
    assert reply.data["stale_reasons"] == recorded_fit["stale_reasons"]
    assert reply.data["params"] == recorded_fit["params"]
    assert reply.data["residual"]["mean"] == 0.112
    assert reply.data["inputs"]["points"] == 3
    assert [item.data for item in reply.images] == ([png] if png else [])
    assert [request["method"] for request in client.transport.requests] == (
        ["fit.result", "fit.render"] if not stale and image else ["fit.result"]
    )
    if stale:
        assert reply.data["note"] == "rerun fluxdep_fit to refresh"


@pytest.mark.parametrize("has_result,stale", [(False, False), (True, False), (True, True)])
def test_fit_result_without_recorded_quality_is_observation_data_not_zero_statistics(
    client: Client, recorded_fit: dict[str, object], has_result: bool, stale: bool,
) -> None:
    recorded_fit.update({
        "has_result": has_result, "quality": None, "stale": stale,
        "stale_reasons": ["filter changed"] if stale else [],
    })
    if not has_result:
        recorded_fit["params"] = None
    client.transport.replies.append(GuiResponse(result=recorded_fit))
    try:
        reply = tool_reply(client.server.tools["fluxdep_fit_result"]["handler"]({"image": True}))
    except RuntimeError as error:
        if str(error) != "Finished fit has no recorded quality":
            raise
        pytest.fail("fit_result must project absent recorded quality as observation data")
    assert reply.data["status"] == ("finished" if has_result else "no_result")
    assert reply.data["params"] == recorded_fit["params"]
    assert reply.data["stale"] is stale
    assert reply.data["stale_reasons"] == recorded_fit["stale_reasons"]
    if has_result:
        assert reply.data["inputs"]["points"] is None
        assert reply.data["residual"] is None
        assert reply.data["per_transition"] is None
    else:
        assert reply.data == {
            "status": "no_result", "params": None, "stale": False, "stale_reasons": [],
        }
    assert reply.images == ()
    assert [request["method"] for request in client.transport.requests] == ["fit.result"]


def test_finished_fit_returns_bounded_quality_and_native_png(
    client: Client, fit_inputs: dict[str, object], recorded_fit: dict[str, object]
) -> None:
    buffer = BytesIO()
    with Image.new("RGB", (12, 8), color="red") as image:
        image.save(buffer, format="PNG")
    png = buffer.getvalue()
    queue_finished(client, fit_inputs, recorded_fit)
    client.transport.replies.append(
        GuiResponse(
            result={"png_b64": base64.b64encode(png).decode("ascii"), "bytes": len(png)}
        )
    )
    reply = tool_reply(
        client.server.tools["fluxdep_fit"]["handler"]({"search_db": "/accepted.h5"})
    )
    assert reply.data == {
        "status": "finished",
        "params": {"EJ": 2.741, "EC": 1.42, "EL": 0.148},
        "inputs": {
            "search_db": "/accepted.h5",
            "EJb": [2.0, 18.0],
            "ECb": [0.2, 3.5],
            "ELb": [0.06, 2.8],
            "r_f": 7.25,
            "sample_f": 9.0,
            "points": 3,
        },
        "residual": {
            "mean": 0.112,
            "unit": "GHz",
            "outliers": 1,
            "outlier_threshold": 0.337,
        },
        "per_transition": [
            {"transition": "transitions 0-1", "points": 2, "mean": 0.095, "max": 0.210},
            {"transition": "transitions 0-1", "points": 1, "mean": 0.455, "max": 0.455},
            {"transition": "blue side 0-2", "points": 0, "mean": None, "max": None},
        ],
    }
    assert [image.data for image in reply.images] == [png]
    assert not reply.is_error
    assert [(r["method"], r["params"]) for r in client.transport.requests] == [
        ("fit.result", {}),
        ("fit.set_params", {**fit_inputs, "search_db": "/accepted.h5"}),
        ("fit.search", {}),
        ("operation.await", {"token": 19, "timeout": 30}),
        ("operation.await", {"token": 19, "timeout": 30}),
        ("fit.result", {}),
        ("fit.render", {}),
    ]


def test_finished_fit_without_recorded_quality_does_not_invent_statistics(
    client: Client, fit_inputs: dict[str, object], recorded_fit: dict[str, object]
) -> None:
    recorded_fit["quality"] = None
    queue_finished(client, fit_inputs, recorded_fit)
    with pytest.raises(RuntimeError, match="quality"):
        client.server.tools["fluxdep_fit"]["handler"]({})
    assert [r["method"] for r in client.transport.requests] == [
        "fit.result",
        "fit.set_params",
        "fit.search",
        "operation.await",
        "operation.await",
        "fit.result",
    ]


def test_native_render_refusal_propagates_without_replaying_search(
    client: Client, fit_inputs: dict[str, object], recorded_fit: dict[str, object]
) -> None:
    queue_finished(client, fit_inputs, recorded_fit)
    client.transport.replies.append(
        GuiResponse(
            error=GuiError("failed_precondition", "GUI refused evidence", "fit_missing")
        )
    )
    with pytest.raises(GuiInvocationError) as error:
        client.server.tools["fluxdep_fit"]["handler"]({})
    assert (error.value.code, error.value.message, error.value.reason) == (
        "failed_precondition",
        "GUI refused evidence",
        "fit_missing",
    )
    assert [r["method"] for r in client.transport.requests] == [
        "fit.result",
        "fit.set_params",
        "fit.search",
        "operation.await",
        "operation.await",
        "fit.result",
        "fit.render",
    ]
