"""NDJSON framing for RemoteControlAdapter.

One JSON object per line each way. Line length capped to defend against runaway
clients. Parsing rejects non-object roots, control characters, and oversized
input early with typed errors.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

from .errors import ErrorCode, RemoteError

MAX_LINE_BYTES = 1 << 20  # 1 MiB per request line
LINE_TERMINATOR = b"\n"


def encode_line(obj: Mapping[str, object]) -> bytes:
    """Serialise one wire object as a single newline-terminated line.

    `obj` must already be a JSON-friendly mapping (caller responsibility);
    embedded newlines inside string values are escaped by `json.dumps`.

    The byte-length check is performed on the *encoded* bytes (not the str
    character count) so that the outbound limit is symmetric with
    :func:`decode_line`, which measures ``len(line)`` in bytes.  A payload
    containing multi-byte characters (e.g. Chinese text or exception messages)
    can have fewer characters than bytes and must be measured after encoding.
    """
    payload = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    data = payload.encode("utf-8")
    if len(data) > MAX_LINE_BYTES:
        raise RemoteError(
            ErrorCode.INTERNAL,
            f"outgoing line exceeds {MAX_LINE_BYTES} bytes ({len(data)})",
        )
    return data + LINE_TERMINATOR


def decode_line(line: bytes) -> dict[str, object]:
    """Parse one request line.

    Strict:
      - line must be valid UTF-8;
      - JSON root must be an object (not array / scalar);
      - line must be within ``MAX_LINE_BYTES``.
    """
    if len(line) > MAX_LINE_BYTES:
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            f"request line exceeds {MAX_LINE_BYTES} bytes ({len(line)})",
        )
    try:
        text = line.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RemoteError(
            ErrorCode.INVALID_PARAMS, f"request is not valid UTF-8: {exc}"
        ) from exc
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RemoteError(
            ErrorCode.INVALID_PARAMS, f"malformed JSON: {exc.msg}"
        ) from exc
    if not isinstance(obj, dict):
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            f"request root must be an object, got {type(obj).__name__}",
        )
    return obj
