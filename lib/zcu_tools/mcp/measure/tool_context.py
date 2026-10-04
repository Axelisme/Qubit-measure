"""One fixed measure tool table's bound GUI session and catalog timeout policy."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from zcu_tools.mcp.core.bridge import McpBridge, MCPBridgeConfig
from zcu_tools.mcp.measure.session import GuiConnection, MeasureMcpSession


@dataclass(frozen=True)
class MeasureToolContext:
    config: MCPBridgeConfig
    session: MeasureMcpSession
    resolve_connect_port: Callable[[MCPBridgeConfig, int | None], int]
    connection: GuiConnection | None = None

    def bound(self, *, require_connected: bool = False) -> MeasureToolContext:
        """Return this bound context, or a copy capturing this session's GUI.

        require_connected=True refuses initial attach/reconnect and raises
        GuiRpcError when no GUI transport is open. False preserves lazy attach.
        Catalog/handshake errors propagate. An inherited binding is not refreshed;
        its next RPC rejects any lost or replacement GUI without reconnecting.
        """
        if self.connection is not None:
            return self
        return replace(
            self, connection=self.session.bind(require_connected=require_connected)
        )

    @property
    def gui(self) -> GuiConnection:
        return self.connection if self.connection is not None else self.session.bind()

    @property
    def bridge(self) -> McpBridge:
        return self.session.bridge

    def send_gui_rpc(
        self,
        method: str,
        params: dict[str, Any],
        timeout_seconds: float | None = None,
        *,
        operation_handle: int | None = None,
    ) -> dict[str, Any]:
        """Send a known GUI method using its live timeout unless overridden."""
        if timeout_seconds is None and method in {"operation.await", "notify.await"}:
            raise ValueError(f"{method!r} requires explicit timeout_seconds")
        return self.gui.send_gui_rpc(
            method, params, timeout_seconds, operation_handle=operation_handle
        )
