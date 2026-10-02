"""Measure MCP connection, live GUI catalog and operation handles."""

from __future__ import annotations

import math
import tempfile
import uuid
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock, RLock
from types import MappingProxyType
from typing import Any, Literal, TypedDict

from zcu_tools.mcp.core.bridge import (
    GuiAuthenticationError,
    GuiTransportTimeoutError,
    McpBridge,
    MCPBridgeConfig,
)
from zcu_tools.mcp.measure.session_policy import describe_stale_keys


class GuiRpcError(RuntimeError):
    """A GUI or MCP boundary error with a stable agent-facing reason."""

    def __init__(
        self, message: str, *, reason: str | None = None, code: str | None = None
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.code = code


class CatalogEntry(TypedDict):
    method: str
    description: str
    params: dict[str, object]
    timeout_seconds: float
    exposure: Literal["rpc", "tool"]
    tool_names: list[str]
    operation_key: str | None


ResolveConnectPortFn = Callable[[MCPBridgeConfig, int | None], int]
PortIsOpenFn = Callable[[int], bool]


def _catalog_strings(raw: object) -> list[str]:
    """Validate catalog names before storing typed lists."""
    if not isinstance(raw, list):
        raise GuiRpcError("invalid GUI rpc.catalog list", reason="incompatible_wire")
    strings: list[str] = []
    for item in raw:
        if not isinstance(item, str) or not item:
            raise GuiRpcError(
                "invalid GUI rpc.catalog list entry", reason="incompatible_wire"
            )
        strings.append(item)
    return strings


def _parse_catalog(raw: object) -> dict[str, CatalogEntry]:
    """Validate the untrusted GUI reply before installing the live catalog."""
    if not isinstance(raw, dict) or not isinstance(raw.get("methods"), list):
        raise GuiRpcError("invalid GUI rpc.catalog reply", reason="incompatible_wire")
    methods: dict[str, CatalogEntry] = {}
    for value in raw["methods"]:
        if not isinstance(value, dict):
            raise GuiRpcError(
                "invalid GUI rpc.catalog entry", reason="incompatible_wire"
            )
        method = value.get("method")
        exposure = value.get("exposure")
        timeout = value.get("timeout_seconds")
        tools = _catalog_strings(value.get("tool_names"))
        schema = value.get("params")
        operation_key = value.get("operation_key")
        if (
            not isinstance(method, str)
            or not method
            or method in methods
            or exposure not in ("rpc", "tool")
            or not isinstance(value.get("description"), str)
            or not isinstance(schema, dict)
            or schema.get("type") != "object"
            or isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
            or (exposure == "tool") != bool(tools)
            or (operation_key is not None and not isinstance(operation_key, str))
        ):
            raise GuiRpcError(
                "invalid or duplicate GUI rpc.catalog entry", reason="incompatible_wire"
            )
        methods[method] = CatalogEntry(
            method=method,
            description=value["description"],
            params=schema,
            timeout_seconds=float(timeout),
            exposure=exposure,
            tool_names=tools,
            operation_key=operation_key,
        )
    return methods


class MeasureMcpSession:
    """One MCP session; bridge transport remains independent of method policy."""

    @dataclass(frozen=True)
    class GuiConnection:
        """A captured GUI incarnation; calls serialize and never reconnect.

        Obtain this capability with session.bind(), then share it across the steps
        of one operation. A changed or disconnected GUI expires the capability.
        """

        _session: MeasureMcpSession
        _generation: int

        @property
        def catalog(self) -> Mapping[str, CatalogEntry]:
            with self._session._rpc_lock:
                self._session._require_connection(self._generation)
                return self._session.catalog

        def read_internal(
            self,
            method: str,
            params: dict[str, Any],
            *,
            operation_handle: int | None = None,
        ) -> dict[str, Any]:
            """Read orientation or control on this GUI only."""
            with self._session._rpc_lock:
                return self._session._read_internal(
                    self._generation, method, params, operation_handle=operation_handle
                )

        def send_gui_rpc(
            self,
            method: str,
            params: dict[str, Any],
            timeout_seconds: float | None = None,
            *,
            operation_handle: int | None = None,
            before_send: Callable[[], None] | None = None,
        ) -> dict[str, Any]:
            """Send once, admitting immediately before dispatch under the RPC lock.

            before_send may reject by raising. It must not send RPCs or block.
            """
            with self._session._rpc_lock:
                return self._session._send_gui_rpc(
                    self._generation,
                    method,
                    params,
                    timeout_seconds,
                    operation_handle=operation_handle,
                    before_send=before_send,
                )

        def expose_operation(self, gui_id: object) -> int:
            with self._session._rpc_lock:
                self._session._require_generation(self._generation)
                return self._session._expose_operation(gui_id)

    def __init__(
        self,
        config: MCPBridgeConfig,
        *,
        bridge: McpBridge | None = None,
        resolve_connect_port: ResolveConnectPortFn,
        port_is_open: PortIsOpenFn,
    ) -> None:
        from zcu_tools.mcp.measure.analysis_execution import AnalysisExecutions

        self._config = config
        self._rpc_lock = RLock()
        self._resource_lock = RLock()
        self._close_lock = Lock()
        self._closed = Event()
        self.executions = AnalysisExecutions(self, self._closed)
        self._generation = 0
        self._bridge = bridge
        self._resolve_connect_port = resolve_connect_port
        self._port_is_open = port_is_open
        self._operation_handles: dict[str, int] = {}
        # GUI IDs restart at 1. Agent handles stay unique within this MCP session.
        self._gui_operations: dict[int, int] = {}
        self._next_operation_handle = 1
        self._catalog: dict[str, CatalogEntry] = {}
        self._connected_port: int | None = None
        self._requested_port: int | None = None
        self._auth_token: str | None = None
        self._versions: dict[str, int] = {}
        self._launched = False
        self._png_directory: tempfile.TemporaryDirectory[str] | None = None

    def new_png_path(self) -> Path:
        """Reserve a fresh session-owned path; the GUI writes the image."""
        with self._resource_lock:
            self._require_open()
            if self._png_directory is None:
                self._png_directory = tempfile.TemporaryDirectory(
                    prefix="measure-mcp-png-"
                )
            return Path(self._png_directory.name) / f"{uuid.uuid4().hex}.png"

    def write_png(self, png: bytes) -> Path:
        """Write one validated image under session ownership; reject after close."""
        with self._resource_lock:
            path = self.new_png_path()
            path.write_bytes(png)
            return path

    def close(self) -> None:
        """Permanently stop admission, disconnect, drain work, then remove PNGs."""
        self._closed.set()
        self.executions.stop_admission()
        with self._close_lock:
            if self._bridge is not None:
                self._bridge.disconnect()
            # Disconnect must wake the holder before waiting for this lock.
            # A connect already in flight may still attach; drain it, then detach.
            with self._rpc_lock:
                pass
            if self._bridge is not None:
                self._bridge.disconnect()
            self.executions.join()
            self.cleanup_pngs()

    def _require_open(self) -> None:
        if self._closed.is_set():
            raise GuiRpcError("MCP session is closed", reason="session_closed")

    def cleanup_pngs(self) -> None:
        with self._resource_lock:
            if self._png_directory is not None:
                self._png_directory.cleanup()
                self._png_directory = None

    @property
    def bridge(self) -> McpBridge:
        if self._bridge is None:
            raise RuntimeError("MeasureMcpSession has no attached McpBridge")
        return self._bridge

    @property
    def catalog(self) -> Mapping[str, CatalogEntry]:
        """A detached snapshot; nested schema and tool lists are not live aliases."""
        with self._rpc_lock:
            return MappingProxyType(deepcopy(self._catalog))

    @property
    def operation_handles(self) -> Mapping[str, int]:
        with self._rpc_lock:
            return MappingProxyType(dict(self._operation_handles))

    def attach_bridge(self, bridge: McpBridge) -> None:
        with self._rpc_lock:
            self._require_open()
            if self._bridge is not None and self._bridge is not bridge:
                raise RuntimeError("MeasureMcpSession bridge is already attached")
            self._bridge = bridge

    def bind(self) -> GuiConnection:
        """Ensure once and capture the current GUI for a multi-step operation."""
        with self._rpc_lock:
            self.ensure_connected()
            return GuiConnection(self, self._generation)

    def _require_generation(self, generation: int) -> None:
        if generation != self._generation:
            raise GuiRpcError(
                "GUI connection changed; binding expired", reason="connection_lost"
            )

    def _require_connection(self, generation: int) -> None:
        self._require_open()
        self._require_generation(generation)
        if not self._catalog or not self.bridge.is_connected:
            raise GuiRpcError(
                "GUI is not connected; binding expired", reason="connection_lost"
            )

    def _clear_connection(self) -> None:
        self._generation += 1
        self._catalog = {}
        self._operation_handles.clear()
        self._gui_operations.clear()
        self._connected_port = None
        self._requested_port = None
        self._versions = {}
        self._launched = False

    def _load_catalog(
        self, port: int, *, launched: bool, requested_port: int | None = None
    ) -> dict[str, Any]:
        """Wire compatibility must be established before reading live policy."""
        self._clear_connection()
        try:
            version = self.bridge.send_rpc_raw("wire.version", {}, 5.0)
            result = version.get("result")
            if not version.get("ok") or not isinstance(result, dict):
                raise GuiRpcError(
                    "invalid GUI wire.version reply", reason="incompatible_wire"
                )
            wire = result.get("wire_version")
            gui = result.get("gui_version")
            if (
                isinstance(wire, bool)
                or not isinstance(wire, int)
                or wire != self._config.wire_version
            ):
                raise GuiRpcError(
                    f"incompatible GUI wire version {wire!r}; expected {self._config.wire_version}",
                    reason="incompatible_wire",
                )
            if isinstance(gui, bool) or not isinstance(gui, int):
                raise GuiRpcError(
                    "invalid GUI wire.version revision", reason="incompatible_wire"
                )
            reply = self.bridge.send_rpc_raw("rpc.catalog", {}, 5.0)
            if not reply.get("ok"):
                error = reply.get("error")
                if isinstance(error, dict) and error.get("code") == "unauthorized":
                    raise GuiRpcError(
                        "GUI authentication is required; provide its control token",
                        reason="unauthorized",
                        code="unauthorized",
                    )
                raise GuiRpcError("GUI rpc.catalog failed", reason="incompatible_wire")
            self._catalog = _parse_catalog(reply.get("result"))
            self._connected_port = port
            self._requested_port = requested_port
            self._versions = {"wire": wire, "gui": gui, "mcp": self._config.mcp_version}
            self._launched = launched
            return {
                "launched": launched,
                "port": port,
                "versions": dict(self._versions),
            }
        except Exception:  # Any failed handshake invalidates this socket.
            self.bridge.disconnect()
            self._clear_connection()
            self._auth_token = None
            raise

    def connect_to_gui(
        self, *, port: int | None, launch: str, clean: bool, token: str | None = None
    ) -> dict[str, Any]:
        """Serialize attach and handshake with all calls on the current GUI."""
        with self._rpc_lock:
            self._require_open()
            return self._connect_to_gui(
                port=port, launch=launch, clean=clean, token=token
            )

    def _connect_to_gui(
        self, *, port: int | None, launch: str, clean: bool, token: str | None = None
    ) -> dict[str, Any]:
        """Attach or launch once; incompatible GUI contracts fail before mutations.

        A repeated attach returns the connected GUI. GUI code revision is shown
        but never compared. No failed or stale mutation is resent automatically.
        """
        if (
            launch != "new"
            and self.bridge.is_connected
            and self._catalog
            and (port is None or port == self._connected_port)
            and (token is None or token == self._auth_token)
        ):
            return {
                "launched": self._launched,
                "port": self._connected_port,
                "versions": dict(self._versions),
            }
        selected = self._resolve_connect_port(self._config, port)
        existing = self._port_is_open(selected)
        if launch == "new" and existing:
            raise GuiRpcError(f"GUI port {selected} is in use", reason="port_in_use")
        if launch == "never" and not existing:
            raise GuiRpcError(
                f"no GUI is listening on port {selected}", reason="no_gui"
            )
        launched = launch == "new" or (launch == "if_missing" and not existing)
        if launched:
            if self.bridge.launched_gui:
                raise GuiRpcError(
                    "this MCP process already launched a GUI; attach to its port or close it before launching another",
                    reason="busy",
                )
            repo_root = Path(__file__).resolve().parents[4]
            self.bridge.launch(
                repo_root,
                selected,
                token=token,
                auto_connect=True,
                extra_args=["--clean"] if clean else None,
            )
            if not self.bridge.is_connected:
                raise GuiRpcError(
                    "GUI launched but is not ready to attach", reason="no_gui"
                )
        else:
            try:
                self.bridge.connect(selected, token=token)
            except GuiAuthenticationError as exc:
                self.bridge.disconnect()
                self._clear_connection()
                self._auth_token = None
                reason = (
                    "unauthorized" if exc.code == "unauthorized" else "auth_disabled"
                )
                raise GuiRpcError(
                    f"GUI authentication failed: {exc}", reason=reason, code=exc.code
                ) from exc
            except RuntimeError as exc:
                self.bridge.disconnect()
                self._clear_connection()
                self._auth_token = None
                raise GuiRpcError(f"GUI attach failed: {exc}", reason="no_gui") from exc
        connection = self._load_catalog(
            selected, launched=launched, requested_port=port
        )
        self._auth_token = token
        return connection

    def ensure_connected(self) -> None:
        """A lazy attach always reloads the catalog after GUI restart."""
        with self._rpc_lock:
            self._ensure_connected()

    def _ensure_connected(self) -> None:
        self._require_open()
        if self.bridge.is_connected and self._catalog:
            return
        if self.bridge.is_connected:
            selected = self._resolve_connect_port(self._config, self._requested_port)
            self._load_catalog(
                selected, launched=False, requested_port=self._requested_port
            )
            return
        self.connect_to_gui(
            port=self._requested_port,
            launch="never",
            clean=False,
            token=self._auth_token,
        )

    def read_internal(
        self,
        method: str,
        params: dict[str, Any],
        *,
        operation_handle: int | None = None,
    ) -> dict[str, Any]:
        """A single orientation/control call, lazily attaching when needed."""
        return self.bind().read_internal(
            method, params, operation_handle=operation_handle
        )

    def _read_internal(
        self,
        generation: int,
        method: str,
        params: dict[str, Any],
        *,
        operation_handle: int | None = None,
    ) -> dict[str, Any]:
        self._require_connection(generation)
        if operation_handle is not None:
            params = self._params_for_operation(params, operation_handle)
        reply = self._bound_rpc(generation, method, params, 6.0)
        if not reply.get("ok"):
            error = reply.get("error", {})
            raise GuiRpcError(
                f"GUI Error ({error.get('code')}): {error.get('message')}",
                reason=error.get("reason"),
                code=error.get("code"),
            )
        result = reply.get("result")
        if not isinstance(result, dict):
            raise GuiRpcError(
                f"invalid GUI reply for {method}", reason="incompatible_wire"
            )
        return result

    def send_gui_rpc(
        self,
        method: str,
        params: dict[str, Any],
        timeout_seconds: float | None = None,
        *,
        operation_handle: int | None = None,
    ) -> dict[str, Any]:
        """A single catalog call; never retry an ambiguous mutation."""
        return self.bind().send_gui_rpc(
            method, params, timeout_seconds, operation_handle=operation_handle
        )

    def _send_gui_rpc(
        self,
        generation: int,
        method: str,
        params: dict[str, Any],
        timeout_seconds: float | None = None,
        *,
        operation_handle: int | None = None,
        before_send: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        self._require_connection(generation)
        if operation_handle is not None:
            params = self._params_for_operation(params, operation_handle)
        entry = self._catalog.get(method)
        if entry is None:
            raise GuiRpcError(f"unknown GUI method {method!r}", reason="unknown_method")
        if timeout_seconds is None:
            timeout_seconds = entry["timeout_seconds"] + 1.0
        if before_send is not None:
            before_send()
        try:
            resp = self._bound_rpc(generation, method, params, timeout_seconds)
        except GuiTransportTimeoutError as exc:
            raise GuiRpcError(
                f"GUI Transport Timeout: {exc}. Reconnect on the next call; review before retrying.",
                reason="gui_transport_timeout",
                code="timeout",
            ) from exc
        if not resp.get("ok", False):
            err = resp.get("error", {})
            if err.get("reason") == "stale_version":
                stale = describe_stale_keys((err.get("data") or {}).get("stale", []))
                detail = f" ({', '.join(stale)})" if stale else ""
                raise GuiRpcError(
                    "GUI Error (PRECONDITION_FAILED): a resource changed since your last read"
                    f"{detail}; review then retry",
                    reason="stale_version",
                    code="precondition_failed",
                )
            code = err.get("code")
            reason = err.get("reason")
            if code == "timeout" and reason is None:
                reason = "gui_handler_timeout"
            raise GuiRpcError(
                f"GUI Error ({code}): {err.get('message')}", reason=reason, code=code
            )
        result = resp.get("result")
        if not isinstance(result, dict):
            raise GuiRpcError(
                f"invalid GUI reply for {method}", reason="incompatible_wire"
            )
        pattern = entry["operation_key"]
        if pattern is not None and "operation_id" in result:
            # A delivered reply stays known even if its socket has just closed.
            self._require_generation(generation)
            result = dict(result)
            handle = self._expose_operation(result.pop("operation_id"))
            key = pattern.format(
                tab_id=params.get("tab_id", ""), name=params.get("name", "")
            )
            self._operation_handles[key] = handle
            result["handle"] = handle
        return result

    def expose_operation(self, gui_id: object) -> int:
        """Expose a GUI ID after ensuring the current connection."""
        return self.bind().expose_operation(gui_id)

    def _expose_operation(self, gui_id: object) -> int:
        """Issue one stable integer handle for a GUI operation in this connection."""
        if isinstance(gui_id, bool) or not isinstance(gui_id, int) or gui_id <= 0:
            raise GuiRpcError("invalid GUI operation id", reason="incompatible_wire")
        handle = self._gui_operations.get(gui_id)
        if handle is None:
            handle = self._next_operation_handle
            self._next_operation_handle += 1
            self._gui_operations[gui_id] = handle
        return handle

    def _bound_rpc(
        self,
        generation: int,
        method: str,
        params: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        try:
            reply = self.bridge.send_rpc_raw(method, params, timeout_seconds)
        except GuiTransportTimeoutError:
            # Preserve the ambiguous timeout classification; never replay.
            raise
        except (RuntimeError, OSError):
            # A disconnect may wake a pending RPC before its reply arrives.
            self._require_connection(generation)
            raise
        # A received reply remains known even if this socket subsequently closes.
        # A replacement GUI, however, must never supply this call's result.
        self._require_generation(generation)
        return reply

    def _params_for_operation(
        self, params: dict[str, Any], handle: int
    ) -> dict[str, Any]:
        """Resolve the handle on the selected GUI before sending, without reconnecting."""
        if "operation_id" in params:
            raise ValueError("pass an operation handle, not a GUI operation id")
        for gui_id, exposed in self._gui_operations.items():
            if exposed == handle:
                return {**params, "operation_id": gui_id}
        raise GuiRpcError("unknown or expired operation", reason="unknown_op")

    def operation_handle_for_key(self, key: str) -> int | None:
        with self._rpc_lock:
            return self._operation_handles.get(key)

    def debug_operations(self) -> dict[str, dict[str, dict[str, int]]]:
        with self._rpc_lock:
            return {
                "handles": {
                    key: {"operation_id": op_id}
                    for key, op_id in self._operation_handles.items()
                }
            }


# The capability is nested with its owner so session state stays private.
GuiConnection = MeasureMcpSession.GuiConnection
