"""RemoteControlAdapter — local socket interface for automation agents.

Bind defaults to ``127.0.0.1``. Newline-delimited JSON request/response.
See ``services/remote/README.md`` for wire schema and threading model.

``RemoteControlAdapter`` / ``ControlOptions`` are imported lazily so that
lightweight, Qt-free consumers (e.g. the standalone ``mcp_server`` bridge) can
import sibling modules such as ``method_specs`` / ``param_spec`` without pulling
in the Qt-bound adapter layer.
"""

from typing import TYPE_CHECKING

from zcu_tools.gui.remote.errors import ErrorCode, ErrorEnvelope, RemoteError

if TYPE_CHECKING:
    from .service import ControlOptions, RemoteControlAdapter

__all__ = [
    "ControlOptions",
    "ErrorCode",
    "ErrorEnvelope",
    "RemoteControlAdapter",
    "RemoteError",
]


def __getattr__(name: str):
    if name in ("ControlOptions", "RemoteControlAdapter"):
        from . import service

        return getattr(service, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
