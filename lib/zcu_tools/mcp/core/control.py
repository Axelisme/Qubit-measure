"""Identity handshake helpers for the two writable GUI MCP frontends."""

from dataclasses import dataclass
from typing import Literal

from zcu_tools.mcp.core.bridge import McpBridge
from zcu_tools.mcp.core.rpc import GuiRpcCall, call_gui


@dataclass(frozen=True)
class ControlConnection:
    """Identity preferences retained across explicit and lazy reconnects.

    label is the nonempty agent identity, shared by same-label connections.
    mode is owner (may acquire) or readonly (never acquires). Invalid values
    raise ValueError before opening a socket. Changing identity requires reconnect.
    """

    label: str = "agent"
    mode: Literal["owner", "readonly"] = "owner"

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("label must be a non-empty string")
        if self.mode not in ("owner", "readonly"):
            raise ValueError("mode must be owner or readonly")

    def initialize(self, bridge: McpBridge) -> dict[str, object]:
        """Initialize after auth/wire checks; return native holder projection.

        GUI decides acquisition. A locked Local or an online remote holder does
        not fail connect. Repeating the same initialization is inert at the GUI.
        Native invocation errors propagate without retry or automatic acquire.
        """
        return call_gui(
            bridge,
            GuiRpcCall(
                "control.connect", {"label": self.label, "mode": self.mode}, 6.0
            ),
        ).data


def probe_control_wire(bridge: McpBridge) -> dict[str, object]:
    """Require the configured wire version before identity initialization.

    Return the native versions; GUI revision is reported, never compared.
    A mismatch raises RuntimeError and callers must discard this socket.
    """
    versions = call_gui(bridge, GuiRpcCall("wire.version", {}, 5.0)).data
    wire = versions.get("wire_version")
    gui = versions.get("gui_version")
    if (
        isinstance(wire, bool)
        or not isinstance(wire, int)
        or wire != bridge.config.wire_version
    ):
        raise RuntimeError(
            f"incompatible GUI wire version {wire!r}; expected {bridge.config.wire_version}"
        )
    if isinstance(gui, bool) or not isinstance(gui, int):
        raise RuntimeError("invalid GUI revision")
    return versions
