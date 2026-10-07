"""Shared request/reply invocation for generated GUI tools."""

from dataclasses import dataclass

from zcu_tools.mcp.core.bridge import McpBridge
from zcu_tools.mcp.core.reply import ToolReply


@dataclass(frozen=True)
class GuiRpcCall:
    """One invocation on the bridge's current GUI binding.

    method is the dotted wire name. params contains caller-supplied decoded JSON
    keys/values from that method's declaration; no default/guard data is invented.
    timeout_seconds is the transport ceiling, including handler and reply budget.
    """

    method: str
    params: dict[str, object]
    timeout_seconds: float


def call_gui(bridge: McpBridge, call: GuiRpcCall) -> ToolReply:
    """Return one successful native result as reply data, with no images yet.

    GUI invocation failure raises RuntimeError preserving code/message/reason.
    Operation outcomes inside result are not invocation errors. Transport failure
    propagates unchanged. This function does not preread, retry, reconnect, decode
    images or update observations. A timeout does not prove no GUI effects occurred.
    """
    response = bridge.send_rpc_raw(call.method, call.params, call.timeout_seconds)
    if not response["ok"]:
        error = response["error"]
        message = f"GUI Error ({error['code']}): {error['message']}"
        if error.get("reason"):
            message += f" (reason: {error['reason']})"
        raise RuntimeError(message)
    return ToolReply(dict(response["result"]))
