"""Measure-gui MCP app session.

``McpBridge`` owns transport state only.  ``MeasureMcpSession`` owns the
measure-gui MCP policy state described by ADR-0014/ADR-0026: diagnostics,
optimistic-concurrency baselines, guarded send flow, and the debug-only latest
operation handle projection.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable, MutableMapping
from typing import Any

from zcu_tools.mcp.core.bridge import (
    GuiTransportTimeoutError,
    McpBridge,
    MCPBridgeConfig,
)
from zcu_tools.mcp.measure.session_policy import (
    DEFAULT_POLICY,
    MeasureMcpPolicy,
    describe_stale_keys,
    expand_pattern_keys,
)


class GuiRpcError(RuntimeError):
    """A wire-level GUI error with structured ``reason`` / ``code`` tags."""

    def __init__(
        self, message: str, *, reason: str | None = None, code: str | None = None
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.code = code


ResolveConnectPortFn = Callable[[MCPBridgeConfig, int | None], int]
PortIsOpenFn = Callable[[int], bool]


class MeasureMcpSession:
    """App-local policy session for the measure-gui MCP server."""

    def __init__(
        self,
        config: MCPBridgeConfig,
        *,
        policy: MeasureMcpPolicy = DEFAULT_POLICY,
        bridge: McpBridge | None = None,
        resolve_connect_port: ResolveConnectPortFn,
        port_is_open: PortIsOpenFn,
        diagnostic_queue_max: int = 1024,
    ) -> None:
        self._config = config
        self._policy = policy
        self._bridge = bridge
        self._resolve_connect_port = resolve_connect_port
        self._port_is_open = port_is_open
        self._diagnostic_queue: deque[dict[str, Any]] = deque(
            maxlen=diagnostic_queue_max
        )
        self._diagnostic_cond = threading.Condition()
        self._last_seen: dict[str, int] = {}
        self._operation_handles: dict[str, int] = {}

    @property
    def policy(self) -> MeasureMcpPolicy:
        return self._policy

    @property
    def bridge(self) -> McpBridge:
        if self._bridge is None:
            raise RuntimeError("MeasureMcpSession has no attached McpBridge")
        return self._bridge

    @property
    def last_seen_versions(self) -> MutableMapping[str, int]:
        return self._last_seen

    @property
    def operation_handles(self) -> MutableMapping[str, int]:
        return self._operation_handles

    def attach_bridge(self, bridge: McpBridge) -> None:
        if self._bridge is not None and self._bridge is not bridge:
            raise RuntimeError("MeasureMcpSession bridge is already attached")
        self._bridge = bridge

    def clear_diagnostics(self) -> None:
        with self._diagnostic_cond:
            self._diagnostic_queue.clear()

    def clear_policy_state(self) -> None:
        """Reset mutable policy state for tests and fresh sessions."""

        self.clear_diagnostics()
        self._last_seen.clear()
        self._operation_handles.clear()

    def deliver_event(self, msg: dict[str, Any]) -> None:
        """Queue only diagnostics; drop resource-change events."""

        if msg.get("event") != "diagnostic":
            return
        with self._diagnostic_cond:
            self._diagnostic_queue.append(msg)
            self._diagnostic_cond.notify_all()

    def drain_pending(self) -> dict[str, list[dict[str, Any]]]:
        """Drain diagnostics buffered for piggybacking on the next tool reply."""

        with self._diagnostic_cond:
            diagnostics = list(self._diagnostic_queue)
            self._diagnostic_queue.clear()
        return {"diagnostics": diagnostics}

    def read_version_table(self) -> dict[str, int] | None:
        """Read the full resource version table without applying guard policy."""

        try:
            resp = self.bridge.send_rpc_raw("resources.versions", {}, 5.0)
        except Exception:  # pragma: no cover - best-effort resync
            return None
        if not resp.get("ok", False):
            return None
        versions = resp.get("result", {}).get("versions")
        return versions if isinstance(versions, dict) else None

    def refresh_versions(self) -> None:
        versions = self.read_version_table()
        if versions is not None:
            self._last_seen.clear()
            self._last_seen.update(versions)

    def build_expected_versions(
        self, method: str, params: dict[str, Any]
    ) -> dict[str, int]:
        deps = self._policy.guard_deps.get(method)
        if not deps:
            return {}
        return expand_pattern_keys(deps, params, self._last_seen)

    def refresh_revealed_versions(self, method: str, params: dict[str, Any]) -> None:
        reveals = self._policy.read_reveals[method]
        versions = self.read_version_table()
        if versions is None:
            return
        self._last_seen.update(expand_pattern_keys(reveals, params, versions))

    def ensure_connected(self) -> None:
        """Lazily attach to a running measure-gui control channel."""

        if self.bridge.is_connected:
            return
        port = self._resolve_connect_port(self._config, None)
        if not self._port_is_open(port):
            raise RuntimeError(
                "no running measure-gui found to attach to "
                f"(tried 127.0.0.1:{port}); start one with gui_launch."
            )
        try:
            self.bridge.connect(port)
        except RuntimeError as exc:
            raise RuntimeError(
                "no running measure-gui found to attach to "
                f"(tried 127.0.0.1:{port}); start one with gui_launch."
            ) from exc

    def send_gui_rpc(
        self,
        method: str,
        params: dict[str, Any],
        timeout_seconds: float = 30.0,
    ) -> dict[str, Any]:
        """Issue one guarded RPC against the GUI; raises on error or timeout."""

        self.ensure_connected()

        send_params = params
        if method in self._policy.guard_deps:
            send_params = dict(params)
            send_params["expected_versions"] = self.build_expected_versions(
                method, params
            )

        try:
            resp = self.bridge.send_rpc_raw(method, send_params, timeout_seconds)
        except GuiTransportTimeoutError as exc:
            raise GuiRpcError(
                (
                    f"GUI Transport Timeout: {exc}. The MCP bridge closed the "
                    "stale control socket and will reconnect on the next call."
                ),
                reason="gui_transport_timeout",
                code="timeout",
            ) from exc
        if not resp.get("ok", False):
            err = resp.get("error", {})
            if err.get("reason") == "stale_version":
                self.refresh_versions()
                data = err.get("data") or {}
                stale = describe_stale_keys(data.get("stale", []))
                detail = f" ({', '.join(stale)})" if stale else ""
                raise RuntimeError(
                    "GUI Error (PRECONDITION_FAILED): a resource you depend on was "
                    f"changed in the GUI since you last saw it{detail}; review then "
                    "retry"
                )
            code = err.get("code")
            reason = err.get("reason")
            if code == "timeout" and reason is None:
                reason = "gui_handler_timeout"
            msg = f"GUI Error ({code}): {err.get('message')}"
            raise GuiRpcError(msg, reason=reason, code=code)

        if method in self._policy.read_reveals:
            self.refresh_revealed_versions(method, params)
        else:
            self.refresh_versions()

        result = dict(resp.get("result", {}))
        key_of = self._policy.operation_key_of.get(method)
        if key_of is not None and "operation_id" in result:
            handle = int(result.pop("operation_id"))
            self._operation_handles[key_of(params)] = handle
            result["handle"] = handle
        return result

    def operation_handle_for_key(self, key: str) -> int | None:
        return self._operation_handles.get(key)

    def debug_operations(self) -> dict[str, dict[str, dict[str, int]]]:
        handles = {
            key: {"operation_id": op_id}
            for key, op_id in self._operation_handles.items()
        }
        return {"handles": handles}
