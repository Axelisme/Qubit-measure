"""Shared runtime context and helpers for measure MCP tool overrides."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from tempfile import gettempdir
from typing import Any, Protocol

from zcu_tools.gui.remote.method_spec import MethodSpec
from zcu_tools.mcp.core.bridge import (
    McpBridge,
    MCPBridgeConfig,
    generated_rpc_timeout_seconds,
)
from zcu_tools.mcp.measure.session import GuiRpcError, MeasureMcpSession


class GuiRpcSender(Protocol):
    def __call__(
        self,
        method: str,
        params: dict[str, Any],
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class MeasureToolContext:
    config: MCPBridgeConfig
    session: MeasureMcpSession
    method_specs: Mapping[str, MethodSpec]
    resolve_connect_port: Callable[[MCPBridgeConfig, int | None], int]

    @property
    def bridge(self) -> McpBridge:
        return self.session.bridge

    def send_gui_rpc(
        self,
        method: str,
        params: dict[str, Any],
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Send through this session with the wire method's timeout policy."""
        if timeout_seconds is None:
            if method in {"operation.await", "notify.await"}:
                raise ValueError(f"{method!r} requires explicit timeout_seconds")
            timeout_seconds = generated_rpc_timeout_seconds(self.method_specs[method])
        return self.session.send_gui_rpc(method, params, float(timeout_seconds))


_WAIT_TRANSPORT_SLACK_SECONDS = 1.0


def _coerce_pairs(
    value: object, *, field: str, keys: tuple[str, str]
) -> list[dict[str, Any]]:
    """Validate a batch list of {k0, k1} dicts, fail-fast on shape errors.

    Validation happens up front (before any RPC) so a malformed item never lets
    a partial batch fire — keeping the failure boundary at 'nothing applied'
    rather than 'some applied'.
    """
    k0, k1 = keys
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field!r} must be a non-empty list")
    out: list[dict[str, Any]] = []
    for i, item in enumerate(value):
        if not isinstance(item, dict) or k0 not in item or k1 not in item:
            raise ValueError(f"{field}[{i}] must be an object with {k0!r} and {k1!r}")
        out.append(item)
    return out


def _is_timeout_error(exc: Exception) -> bool:
    """True when a send_gui_rpc error is a normal GUI handler timeout.

    ``gui_transport_timeout`` means the control socket stopped replying and the
    bridge has dropped it; that is not an operation-still-running signal.
    """
    if isinstance(exc, GuiRpcError):
        return exc.code == "timeout" and exc.reason != "gui_transport_timeout"
    return "(timeout)" in str(exc)


def _start_op_with_short_wait(
    ctx: MeasureToolContext,
    key: str,
    what: str,
    wait_seconds: float,
    product: Callable[[], dict[str, Any]],
    pending_hint: str,
) -> dict[str, Any]:
    """Wait briefly for a just-started async op, degrading to a handle on timeout.

    The start RPC must already have run (its operation_id captured under ``key`` by
    send_gui_rpc, and also kept in the START reply as ``handle``). Awaits up to
    ``wait_seconds``:
    - settles in time -> ``{status:'finished', handle, **product()}`` so the caller
      sees the op's resulting state immediately (device snapshot / tab snapshot);
    - still running -> ``{status:'pending', handle, message:<hint>}`` so the caller
      can poll/wait the handle via gui_op_poll / gui_op_wait (ADR-0026 §8).
      operation.await still raises on failure/cancel.

    The reply always carries ``handle`` (pending AND finished) when a handle was
    captured, so the agent has one consistent token to drive gui_op_poll /
    gui_op_wait. Shared by device connect/disconnect/setup and tab.run_start.
    (soc.connect is excluded: it is synchronous and returns its product directly.)
    """
    operation_id = ctx.session.operation_handle_for_key(key)
    if operation_id is None:
        # No handle captured (op already settled synchronously) — report product.
        return {"status": "finished", **product()}
    try:
        ctx.send_gui_rpc(
            "operation.await",
            {"operation_id": operation_id, "timeout": wait_seconds},
            wait_seconds + _WAIT_TRANSPORT_SLACK_SECONDS,
        )
    except RuntimeError as exc:
        if _is_timeout_error(exc):
            return {
                "status": "pending",
                "handle": operation_id,
                "message": f"{what} still in progress after {wait_seconds}s; {pending_hint}",
            }
        raise  # genuine failure/cancellation surfaces as an error
    return {"status": "finished", "handle": operation_id, **product()}


def _render_tab_figure(
    ctx: MeasureToolContext, tab_id: str, subtab_id: str, out_path: str | None = None
) -> dict[str, Any]:
    """Render a specific pane's figure to a PNG FILE (never inline base64).

    Drives ``tab.get_figure`` with required (tab_id, subtab_id) in out_path mode;
    synthesises a per-pane temp path under gettempdir() when no path is given.
    Returns the wire reply ({saved_to, bytes}).
    """
    allowed = {"run", "analysis", "post_analysis"}
    if subtab_id not in allowed:
        raise ValueError(
            f"subtab_id must be one of {sorted(allowed)}, got {subtab_id!r}"
        )
    resolved = out_path or str(
        Path(gettempdir()) / f"measure_fig_{tab_id}_{subtab_id}.png"
    )
    return ctx.send_gui_rpc(
        "tab.get_figure",
        {"tab_id": tab_id, "subtab_id": subtab_id, "out_path": resolved},
    )


def _fold_finished_figure(
    ctx: MeasureToolContext, tab_id: str, reply: dict[str, Any], *, subtab_id: str
) -> dict[str, Any]:
    """Fold a pane's figure into a FINISHED run/analyze reply, in place.

    Only acts when ``reply['status'] == 'finished'``. Renders the requested
    pane's figure to the per-pane temp PNG and adds ``figure: <saved_to>``.
    A render failure is swallowed (recorded as ``figure: None``) so a plotting
    hiccup never masks an otherwise-good result.
    """
    if reply.get("status") != "finished":
        return reply
    try:
        reply["figure"] = _render_tab_figure(ctx, tab_id, subtab_id).get("saved_to")
    except Exception:
        reply["figure"] = None
    return reply
