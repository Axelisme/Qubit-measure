"""One fixed measure tool table's bound GUI session and catalog timeout policy."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from zcu_tools.mcp.core.bridge import McpBridge, MCPBridgeConfig
from zcu_tools.mcp.measure.session import GuiRpcError, MeasureMcpSession


@dataclass(frozen=True)
class MeasureToolContext:
    config: MCPBridgeConfig
    session: MeasureMcpSession
    resolve_connect_port: Callable[[MCPBridgeConfig, int | None], int]

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
        if timeout_seconds is None:
            if method in {"operation.await", "notify.await"}:
                raise ValueError(f"{method!r} requires explicit timeout_seconds")
            self.session.ensure_connected()
            entry = self.session.catalog.get(method)
            if entry is None:
                raise GuiRpcError(
                    f"unknown GUI method {method!r}", reason="unknown_method"
                )
            timeout_seconds = entry["timeout_seconds"] + 1.0
        return self.session.send_gui_rpc(
            method, params, timeout_seconds, operation_handle=operation_handle
        )
