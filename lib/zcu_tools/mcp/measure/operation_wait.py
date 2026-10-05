"""Shared native operation wait; completion policy remains with each caller."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from threading import Condition, Event
from typing import Literal

from zcu_tools.mcp.measure.session import GuiConnection, GuiRpcError


@dataclass(frozen=True)
class OperationCompletion:
    """One confirmed native terminal outcome, not a cancellation intention.

    status is finished, failed or cancelled from the GUI's completed await reply.
    native retains that detached reply, including diagnostic and feedback fields.
    The caller decides how failure/cancellation affects its continuation.
    """

    status: Literal["finished", "failed", "cancelled"]
    native: Mapping[str, object]


def await_operation(
    connection: GuiConnection,
    handle: int,
    *,
    closed: Event,
    condition: Condition,
    before_send: Callable[[], None] | None = None,
) -> OperationCompletion:
    """Await handle on its fixed connection in the current worker; never replay.

    handle is an opaque session operation handle. closed belongs to the owning
    MCP session. Its owner sets it and notifies condition when closing. The
    condition also lets control requests wake the paced wait between GUI calls.
    before_send is a nonblocking admission callback under the RPC lock; it may
    raise but must not send RPCs, as in GuiConnection.send_gui_rpc.

    Use native operation.await with a 0.25-second GUI wait and 2.25-second
    transport deadline. Timeout/user_feedback continue observation; only a
    completed reply with a terminal status returns. Close, connection failures,
    admission errors and malformed replies raise without retry or reconnect.
    """

    def admit() -> None:
        if closed.is_set():
            raise GuiRpcError("MCP session is closed", reason="session_closed")
        if before_send is not None:
            before_send()

    while True:
        began = time.monotonic()
        reply: Mapping[str, object] = connection.send_gui_rpc(
            "operation.await",
            {"timeout": 0.25},
            2.25,
            operation_handle=handle,
            before_send=admit,
        )
        reason = reply.get("reason")
        if reason == "completed":
            status = reply.get("status")
            native = deepcopy(dict(reply))
            if status == "finished":
                return OperationCompletion("finished", native)
            if status == "failed":
                return OperationCompletion("failed", native)
            if status == "cancelled":
                return OperationCompletion("cancelled", native)
            raise GuiRpcError("Invalid operation outcome", reason="incompatible_wire")
        if reason not in ("timeout", "user_feedback"):
            raise GuiRpcError(
                "Invalid operation await reply", reason="incompatible_wire"
            )
        # Pace immediate feedback without holding the RPC lock. Owner control
        # notifications interrupt this wait; close is checked before next send.
        with condition:
            if not closed.is_set():
                condition.wait(max(0.0, 0.25 - (time.monotonic() - began)))
