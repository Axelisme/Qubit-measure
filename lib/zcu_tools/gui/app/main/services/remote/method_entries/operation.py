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
            "Wait for a known operation. Unknown/evicted ids fail; terminal failure is returned as status/error.",
            (
                _int("operation_id", "Operation handle returned by the start op"),
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
            "Request cancellation of an operation by id",
            (_int("operation_id", "Known operation handle"),),
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
            "(null when unknown), raw n/total. Agents read progress through wait.",
            (_int("operation_id", "Operation handle returned by the start op"),),
        ),
        agent=AgentMethodPolicy(exposure="internal"),
    ),
)
