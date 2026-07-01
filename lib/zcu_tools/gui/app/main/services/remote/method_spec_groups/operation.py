"""Operation remote method specs."""

from __future__ import annotations

from zcu_tools.gui.remote.method_spec import MethodSpec

from ._params import (
    _int,
    _num_default,
)

SPECS: dict[str, MethodSpec] = {
    "operation.await": MethodSpec(
        130.0,
        "Block until an async operation settles (by operation_id)",
        (
            _int("operation_id", "Operation handle returned by the start op"),
            _num_default("timeout", 120.0, "Seconds to wait"),
        ),
        off_main_thread=True,
    ),
    "operation.progress": MethodSpec(
        5.0,
        "Read one operation's live progress bars by operation_id (run or device "
        "setup alike). active=false/bars=[] when idle; each bar has token, format "
        "(human-readable e.g. 'Rounds 23/100 [0:25<1:15]'), maximum/value "
        "(Qt-scaled), percent (0-100, null when total unknown), raw n/total. "
        "Internal: agents read progress folded into the gui_*_poll reply.",
        (_int("operation_id", "Operation handle returned by the start op"),),
    ),
}
