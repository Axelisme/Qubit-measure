"""Measure MCP connection, live GUI catalog and guarded request state."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, MutableMapping
from pathlib import Path
from string import Formatter
from typing import Any, Literal, TypedDict

from zcu_tools.mcp.core.bridge import (
    GuiAuthenticationError,
    GuiTransportTimeoutError,
    McpBridge,
    MCPBridgeConfig,
)
from zcu_tools.mcp.measure.session_policy import (
    describe_stale_keys,
    expand_pattern_keys,
)


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
    guard_deps: tuple[str, ...]
    reveals: tuple[str, ...]
    reveals_without: tuple[str, ...]
    reveals_when_nonempty: tuple[str, ...]
    refresh_after_write: bool
    created_resource: str | None
    operation_key: str | None


ResolveConnectPortFn = Callable[[MCPBridgeConfig, int | None], int]
PortIsOpenFn = Callable[[int], bool]


def _created_resource_fields(pattern: str) -> tuple[str, ...]:
    """Only plain returned identifiers can certify an unseen new resource."""
    try:
        parts = tuple(Formatter().parse(pattern))
    except ValueError as exc:
        raise GuiRpcError(
            "invalid GUI rpc.catalog created-resource policy",
            reason="incompatible_wire",
        ) from exc
    fields = tuple(field for _, field, _, _ in parts if field is not None)
    if not fields or any(
        not field.isidentifier() or spec or conversion
        for _, field, spec, conversion in parts
        if field is not None
    ):
        raise GuiRpcError(
            "invalid GUI rpc.catalog created-resource policy",
            reason="incompatible_wire",
        )
    return fields


def _catalog_strings(raw: object) -> list[str]:
    """Validate catalog names and policy patterns before storing typed lists."""
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
    """Validate the untrusted GUI reply before installing any policy state."""
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
        refresh_after_write = value.get("refresh_after_write")
        created_resource = value.get("created_resource")
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
            or not isinstance(refresh_after_write, bool)
            or "created_resource" not in value
            or (
                created_resource is not None
                and (
                    not isinstance(created_resource, str)
                    or not created_resource
                    or not refresh_after_write
                )
            )
        ):
            raise GuiRpcError(
                "invalid or duplicate GUI rpc.catalog entry", reason="incompatible_wire"
            )
        if created_resource is not None:
            _created_resource_fields(created_resource)
        deps = _catalog_strings(value.get("guard_deps"))
        reveals = _catalog_strings(value.get("reveals"))
        reveals_without = _catalog_strings(value.get("reveals_without"))
        reveals_when_nonempty = _catalog_strings(value.get("reveals_when_nonempty"))
        if (reveals_without or reveals_when_nonempty) and not reveals:
            raise GuiRpcError(
                "invalid GUI rpc.catalog policy", reason="incompatible_wire"
            )
        methods[method] = CatalogEntry(
            method=method,
            description=value["description"],
            params=schema,
            timeout_seconds=float(timeout),
            exposure=exposure,
            tool_names=tools,
            guard_deps=tuple(deps),
            reveals=tuple(reveals),
            reveals_without=tuple(reveals_without),
            reveals_when_nonempty=tuple(reveals_when_nonempty),
            refresh_after_write=refresh_after_write,
            created_resource=created_resource,
            operation_key=operation_key,
        )
    return methods


class MeasureMcpSession:
    """One MCP session; bridge transport remains independent of method policy."""

    def __init__(
        self,
        config: MCPBridgeConfig,
        *,
        bridge: McpBridge | None = None,
        resolve_connect_port: ResolveConnectPortFn,
        port_is_open: PortIsOpenFn,
    ) -> None:
        self._config = config
        self._bridge = bridge
        self._resolve_connect_port = resolve_connect_port
        self._port_is_open = port_is_open
        self._last_seen: dict[str, int] = {}
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

    @property
    def bridge(self) -> McpBridge:
        if self._bridge is None:
            raise RuntimeError("MeasureMcpSession has no attached McpBridge")
        return self._bridge

    @property
    def catalog(self) -> Mapping[str, CatalogEntry]:
        """Validated live GUI entries, refreshed for each new connection."""
        return self._catalog

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

    def _clear_connection(self) -> None:
        self._catalog = {}
        self._last_seen.clear()
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
        """A lazy attach always reloads catalog/observations after GUI restart."""
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

    def read_internal(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Read a known GUI orientation method without exporting it to rpc_call.

        The caller names only shipped read operations. No MCP method or guard
        registry is maintained for these GUI-owned internal reads.
        """
        self.ensure_connected()
        entry = self._catalog.get(method)
        observed = self._read_revealed_versions(entry, params)
        reply = self.bridge.send_rpc_raw(method, params, 6.0)
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
        # The catalog is the sole owner of read-reveal policy, including reads
        # used internally by status (such as device membership).
        if observed is not None:
            self._last_seen.update(observed)
        return result

    def read_version_table(self) -> dict[str, int] | None:
        """Sample versions before a full read; this never re-snapshots resources."""
        try:
            resp = self.bridge.send_rpc_raw("resources.versions", {}, 5.0)
        except (OSError, RuntimeError):
            return None
        if not resp.get("ok", False):
            return None
        versions = resp.get("result", {}).get("versions")
        if not isinstance(versions, dict) or not all(
            isinstance(key, str)
            and isinstance(version, int)
            and not isinstance(version, bool)
            for key, version in versions.items()
        ):
            return None
        return versions

    def build_expected_versions(
        self, method: str, params: dict[str, Any]
    ) -> dict[str, int]:
        return expand_pattern_keys(
            self._catalog[method]["guard_deps"], params, self._last_seen
        )

    def _read_revealed_versions(
        self, entry: CatalogEntry | None, params: dict[str, Any]
    ) -> dict[str, int] | None:
        """Sample before the read; a later version may describe data not in its reply."""
        if (
            entry is None
            or not entry["reveals"]
            or any(name in params for name in entry["reveals_without"])
            or any(not params.get(name) for name in entry["reveals_when_nonempty"])
        ):
            return None
        versions = self.read_version_table()
        if versions is None:
            return None
        return expand_pattern_keys(entry["reveals"], params, versions)

    def _record_successful_versions(
        self,
        entry: CatalogEntry,
        observed: dict[str, int] | None,
        result: dict[str, Any],
    ) -> None:
        if observed is not None:
            self._last_seen.update(observed)
        if not entry["refresh_after_write"]:
            return
        changes = result.pop("__agent_write_versions", None)
        if not isinstance(changes, dict) or any(
            not isinstance(key, str)
            or not key
            or not isinstance(pair, list)
            or len(pair) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in pair)
            or pair[1] <= pair[0]
            for key, pair in changes.items()
        ):
            raise GuiRpcError(
                "invalid GUI write-version receipt", reason="incompatible_wire"
            )
        created_resource = entry["created_resource"]
        if created_resource is not None:
            fields = _created_resource_fields(created_resource)
            if any(
                not isinstance(result.get(field), str) or not result[field]
                for field in fields
            ):
                raise GuiRpcError(
                    "invalid GUI created-resource identity", reason="incompatible_wire"
                )
            created_key = created_resource.format(**result)
            # A new, never-reused resource is the sole exception to requiring a
            # prior read: its owner-thread receipt certifies both existence and
            # identity. It does not reveal any of the tab's other resources.
            if changes.get(created_key) != [0, 1] or created_key in self._last_seen:
                raise GuiRpcError(
                    "invalid GUI created-resource receipt", reason="incompatible_wire"
                )
            self._last_seen[created_key] = 1
        for key, (before, after) in changes.items():
            # An unguarded write cannot certify an unseen edit made before it.
            if self._last_seen.get(key) == before:
                self._last_seen[key] = after

    def send_gui_rpc(
        self,
        method: str,
        params: dict[str, Any],
        timeout_seconds: float | None = None,
        *,
        rpc_only: bool = False,
    ) -> dict[str, Any]:
        """One guarded send; transport failure never retries an ambiguous mutation."""
        self.ensure_connected()
        entry = self._catalog.get(method)
        if entry is None:
            raise GuiRpcError(f"unknown GUI method {method!r}", reason="unknown_method")
        if rpc_only and entry["exposure"] != "rpc":
            tools = ", ".join(entry["tool_names"])
            raise GuiRpcError(f"use {tools} for {method}", reason="use_tool")
        if timeout_seconds is None:
            timeout_seconds = entry["timeout_seconds"] + 1.0
        send_params = params
        if entry["guard_deps"]:
            send_params = {
                **params,
                "expected_versions": self.build_expected_versions(method, params),
            }
        observed = self._read_revealed_versions(entry, params)
        try:
            resp = self.bridge.send_rpc_raw(method, send_params, timeout_seconds)
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
        result = dict(result)
        self._record_successful_versions(entry, observed, result)
        pattern = entry["operation_key"]
        if pattern is not None and "operation_id" in result:
            handle = self.expose_operation(result.pop("operation_id"))
            keys = expand_pattern_keys((pattern,), params, {})
            key = next(iter(keys))
            self._operation_handles[key] = handle
            result["handle"] = handle
        return result

    def expose_operation(self, gui_id: object) -> int:
        """Issue one stable integer handle for a GUI operation in this connection."""
        if isinstance(gui_id, bool) or not isinstance(gui_id, int) or gui_id <= 0:
            raise GuiRpcError("invalid GUI operation id", reason="incompatible_wire")
        handle = self._gui_operations.get(gui_id)
        if handle is None:
            handle = self._next_operation_handle
            self._next_operation_handle += 1
            self._gui_operations[gui_id] = handle
        return handle

    def gui_operation_id(self, handle: int) -> int:
        """Reject handles from an earlier GUI before sending wait or cancel."""
        self.ensure_connected()
        for gui_id, exposed in self._gui_operations.items():
            if exposed == handle:
                return gui_id
        raise GuiRpcError("unknown or expired operation", reason="unknown_op")

    def operation_handle_for_key(self, key: str) -> int | None:
        return self._operation_handles.get(key)

    def debug_operations(self) -> dict[str, dict[str, dict[str, int]]]:
        return {
            "handles": {
                key: {"operation_id": op_id}
                for key, op_id in self._operation_handles.items()
            }
        }
