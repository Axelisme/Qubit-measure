"""RemoteControlService — local socket interface for automation agents.

Bind defaults to ``127.0.0.1``. Newline-delimited JSON request/response.
See ``AI_NOTE.md`` for wire schema and threading model.
"""

from .errors import ErrorCode, ErrorEnvelope, RemoteError
from .service import ControlOptions, RemoteControlService

__all__ = [
    "ControlOptions",
    "ErrorCode",
    "ErrorEnvelope",
    "RemoteControlService",
    "RemoteError",
]
