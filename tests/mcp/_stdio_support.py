"""In-memory MCP requests through the public library stdio loop."""

import json
from io import BytesIO, TextIOWrapper
from typing import Any

from zcu_tools.mcp.core.stdio_server import McpServerConfig, ToolTable, run_stdio_loop


def invoke_stdio(
    config: McpServerConfig, tools: ToolTable, name: str, arguments: dict[str, object]
) -> dict[str, Any]:
    """Send one tool call without replacing global streams or starting a process."""
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }
    with (
        TextIOWrapper(
            BytesIO((json.dumps(request) + "\n").encode("utf-8")), encoding="utf-8"
        ) as incoming,
        TextIOWrapper(BytesIO(), encoding="utf-8") as outgoing,
    ):
        run_stdio_loop(
            config,
            tools,
            input_stream=incoming,
            output_stream=outgoing,
        )
        outgoing.flush()
        outgoing.seek(0)
        return json.load(outgoing)
