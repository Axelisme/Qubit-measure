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


# ---------------------------------------------------------------------------
# encode_line — multi-byte character byte-length symmetry (Fix 1 regression)
# ---------------------------------------------------------------------------


def test_encode_line_multibyte_chars_exceed_byte_limit_but_not_char_limit():
    """A payload whose JSON *character* count is under MAX_LINE_BYTES but whose
    UTF-8 *byte* count is over must be rejected by encode_line.

    Each CJK character (e.g. "測") encodes to 3 UTF-8 bytes, so a string of
    roughly MAX_LINE_BYTES // 3 + 1 such characters will exceed the byte limit
    while being well under the character limit.
    """
    # Build a value string where:
    #   char count ≈ MAX_LINE_BYTES // 3 + 1  →  under MAX_LINE_BYTES chars
    #   byte count ≈ (MAX_LINE_BYTES // 3 + 1) * 3  →  over MAX_LINE_BYTES bytes
    cjk_count = MAX_LINE_BYTES // 3 + 1  # each char → 3 bytes → total > 1 MiB
    big_value = "測" * cjk_count
    # Sanity check: char count is under the limit, byte count is over.
    import json

    payload_chars = json.dumps(
        {"data": big_value}, ensure_ascii=False, separators=(",", ":")
    )
    assert len(payload_chars) < MAX_LINE_BYTES, (
        "test setup: character count should be under the limit"
    )
    assert len(payload_chars.encode("utf-8")) > MAX_LINE_BYTES, (
        "test setup: byte count should exceed the limit"
    )

    with pytest.raises(RemoteError) as exc_info:
        encode_line({"data": big_value})
    assert exc_info.value.code == ErrorCode.INTERNAL


def test_encode_line_pure_ascii_near_limit_still_encodes():
    """A pure-ASCII payload just under MAX_LINE_BYTES must not be rejected.

    For ASCII the byte count equals the character count, so the old (broken)
    check and the new (correct) check agree — this test confirms no regression
    for the common case.
    """
    # Build a key + value whose JSON representation is just under the limit.
    # json.dumps({"data": "x" * N}) => '{"data":"' + 'x'*N + '"}' = N + 10 chars.
    n = MAX_LINE_BYTES - 10 - 1  # one byte under the limit after serialisation
    result = encode_line({"data": "x" * n})
    assert result.endswith(b"\n")
    # The encoded bytes (without the newline) must be within the limit.
    assert len(result) - 1 <= MAX_LINE_BYTES
