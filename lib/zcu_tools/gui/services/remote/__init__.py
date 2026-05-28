"""RemoteControlService — local socket interface for automation agents.

Bind defaults to ``127.0.0.1``. Newline-delimited JSON request/response.
See ``AI_NOTE.md`` for wire schema and threading model.

``RemoteControlService`` / ``ControlOptions`` are imported lazily so that
lightweight, Qt-free consumers (e.g. the standalone ``mcp_server`` bridge) can
import sibling modules such as ``method_specs`` / ``param_spec`` without pulling
in the Qt-bound service layer.
"""

from typing import TYPE_CHECKING

from .errors import ErrorCode, ErrorEnvelope, RemoteError

if TYPE_CHECKING:
    from .service import ControlOptions, RemoteControlService

__all__ = [
    "ControlOptions",
    "ErrorCode",
    "ErrorEnvelope",
    "RemoteControlService",
    "RemoteError",
]


def __getattr__(name: str):
    if name in ("ControlOptions", "RemoteControlService"):
        from . import service

        return getattr(service, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
