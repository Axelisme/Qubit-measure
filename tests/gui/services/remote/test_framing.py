"""Unit tests for NDJSON framing helpers (encode_line / decode_line).

Pure bytes operations — no Qt required.
"""

from __future__ import annotations

import json

import pytest
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError
from zcu_tools.gui.remote.framing import (
    MAX_LINE_BYTES,
    decode_line,
    encode_line,
)

# ---------------------------------------------------------------------------
# encode_line
# ---------------------------------------------------------------------------


def test_encode_line_produces_newline_terminated_bytes():
    result = encode_line({"id": "1", "method": "foo"})
    assert result.endswith(b"\n")


def test_encode_line_is_valid_json():
    result = encode_line({"id": "1", "method": "foo", "params": {"x": 42}})
    parsed = json.loads(result.decode("utf-8").rstrip("\n"))
    assert parsed == {"id": "1", "method": "foo", "params": {"x": 42}}


def test_encode_decode_roundtrip():
    obj = {"id": "abc", "method": "bar", "params": {"nested": [1, 2, 3]}}
    decoded = decode_line(encode_line(obj))
    assert decoded == obj


def test_encode_line_raises_for_oversized_payload():
    big_value = "x" * (MAX_LINE_BYTES + 1)
    with pytest.raises(RemoteError) as exc_info:
        encode_line({"data": big_value})
    assert exc_info.value.code == ErrorCode.INTERNAL


# ---------------------------------------------------------------------------
# decode_line — success
# ---------------------------------------------------------------------------


def test_decode_line_returns_dict():
    line = b'{"id":"1","method":"ping"}\n'
    result = decode_line(line)
    assert result == {"id": "1", "method": "ping"}


def test_decode_line_accepts_empty_object():
    result = decode_line(b"{}")
    assert result == {}


# ---------------------------------------------------------------------------
# decode_line — oversized line
# ---------------------------------------------------------------------------


def test_decode_line_raises_for_oversized_line():
    oversized = b"x" * (MAX_LINE_BYTES + 1)
    with pytest.raises(RemoteError) as exc_info:
        decode_line(oversized)
    assert exc_info.value.code == ErrorCode.INVALID_PARAMS


# ---------------------------------------------------------------------------
# decode_line — invalid UTF-8
# ---------------------------------------------------------------------------


def test_decode_line_raises_for_invalid_utf8():
    invalid_utf8 = b"\xff\xfe"  # not valid UTF-8
    with pytest.raises(RemoteError) as exc_info:
        decode_line(invalid_utf8)
    assert exc_info.value.code == ErrorCode.INVALID_PARAMS
    assert "UTF-8" in str(exc_info.value)


# ---------------------------------------------------------------------------
# decode_line — invalid JSON
# ---------------------------------------------------------------------------


def test_decode_line_raises_for_malformed_json():
    with pytest.raises(RemoteError) as exc_info:
        decode_line(b"{not valid json}")
    assert exc_info.value.code == ErrorCode.INVALID_PARAMS


# ---------------------------------------------------------------------------
# decode_line — JSON root is not an object
# ---------------------------------------------------------------------------


def test_decode_line_raises_for_array_root():
    with pytest.raises(RemoteError) as exc_info:
        decode_line(b"[1,2,3]")
    assert exc_info.value.code == ErrorCode.INVALID_PARAMS
    assert "object" in str(exc_info.value)


def test_decode_line_raises_for_scalar_root():
    with pytest.raises(RemoteError) as exc_info:
        decode_line(b"42")
    assert exc_info.value.code == ErrorCode.INVALID_PARAMS
