"""Operation remote method entries."""

from __future__ import annotations

from zcu_tools.gui.remote.method_spec import McpMethodPolicy, MethodSpec

from ._params import (
    _int,
    _num_default,
)
from ._registry import RemoteMethodEntry, method_entry

METHODS: tuple[RemoteMethodEntry, ...] = (
    method_entry(
        "operation.await",
        "operation:_h_operation_await",
        MethodSpec(
            130.0,
            "Block until an async operation settles (by operation_id)",
            (
                _int("operation_id", "Operation handle returned by the start op"),
                _num_default("timeout", 120.0, "Seconds to wait"),
            ),
            off_main_thread=True,
            mcp=McpMethodPolicy.override(
                "gui_op_wait",
                "gui_op_poll",
                reason="manual MCP tools expose blocking wait and nonblocking poll semantics",
            ),
        ),
    ),
    method_entry(
        "operation.progress",
        "operation:_h_operation_progress",
        MethodSpec(
            5.0,
            "Read one operation's live progress bars by operation_id (run or device "
            "setup alike). active=false/bars=[] when idle; each bar has token, format "
            "(human-readable e.g. 'Rounds 23/100 [0:25<1:15]'), maximum/value "
            "(Qt-scaled), percent (0-100, null when total unknown), raw n/total. "
            "Internal: agents read progress folded into the gui_*_poll reply.",
            (_int("operation_id", "Operation handle returned by the start op"),),
            mcp=McpMethodPolicy.internal("folded into gui_op_poll running replies"),
        ),
    ),
)
