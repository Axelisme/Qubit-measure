"""Operation remote method entries."""

from __future__ import annotations

from zcu_tools.gui.remote.method_spec import MethodSpec

from ._params import (
    _int,
    _num_default,
)
from ._registry import AgentMethodPolicy, RemoteMethodEntry, method_entry

METHODS: tuple[RemoteMethodEntry, ...] = (
    method_entry(
        "operation.active",
        "operation:_h_operation_active",
        MethodSpec(5.0, "List all live operations from GUI domain owners."),
        agent=AgentMethodPolicy(exposure="internal"),
    ),
    method_entry(
        "operation.await",
        "operation:_h_operation_await",
        MethodSpec(
            130.0,
            "Wait for a known operation with the fixed wait(op) tool. MCP translates "
            "the exposed op handle to a GUI-local operation_id. Unknown or evicted "
            "ids fail; terminal failure and Stop feedback are returned as data.",
            (
                _int(
                    "operation_id", "GUI-local operation id mapped from an MCP handle"
                ),
                _num_default("timeout", 120.0, "Seconds to wait"),
            ),
            off_main_thread=True,
        ),
        agent=AgentMethodPolicy(exposure="tool", tool_names=("wait",)),
    ),
    method_entry(
        "operation.cancel",
        "operation:_h_operation_cancel",
        MethodSpec(
            5.0,
            "Request cancellation of a GUI-local operation by id (MCP exposes "
            "the fixed cancel(op) tool instead of this internal method).",
            (_int("operation_id", "Known GUI-local operation id"),),
        ),
        agent=AgentMethodPolicy(exposure="internal"),
    ),
    method_entry(
        "operation.progress",
        "operation:_h_operation_progress",
        MethodSpec(
            5.0,
            "Read one operation's live progress bars by operation_id (run or device "
            "setup alike). active=false/bars=[] when idle; each bar has token, format "
            "(human-readable e.g. 'Rounds 23/100 [0:25<1:15]'), maximum/value "
            "(Qt-scaled), percent (0-100, null when total unknown), eta_s "
            "(null when unknown), raw n/total. Top-level elapsed_s is the "
            "operation-wide lifetime from the GUI handle registry, or null for "
            "unknown/evicted ids; it is not derived from individual bars. "
            "Agents read progress through wait.",
            (_int("operation_id", "Known GUI-local operation id"),),
        ),
        agent=AgentMethodPolicy(exposure="internal"),
    ),
)
