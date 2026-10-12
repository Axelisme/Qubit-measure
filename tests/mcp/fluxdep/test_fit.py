"""Fit decisions compose native GUI search/query contracts without local state."""

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
