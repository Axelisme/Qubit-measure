"""Editing representation contracts, independent of persistence and transport."""

import math

import pytest
from zcu_tools.gui.cfg.edit_codec import (
    decode_edits,
    decode_input,
    decode_path,
    decode_ref,
    decode_revision,
    encode_input,
    encode_ref,
)
from zcu_tools.gui.cfg.model import DirectValue, EvalValue
from zcu_tools.gui.cfg.resource import CfgInput, CfgInputError, CfgInputReason
from zcu_tools.gui.expected_error import ExpectedErrorCategory


@pytest.mark.parametrize(
    "value",
    [
        None,
        True,
        12,
        1.25,
        "12",
        "md.frequency",
        2 - 3j,
        DirectValue(raw="1e-"),
        EvalValue("md.frequency +"),
        {"__ref": "readout", "frequency": 42},
        [1, "2", 3j],
    ],
)
def test_input_roundtrip_preserves_intent(value: CfgInput) -> None:
    assert decode_input(encode_input(value)) == value


def test_plain_strings_do_not_request_parsing() -> None:
    assert decode_input("1e-") == "1e-"
    assert decode_input("$md.frequency") == "$md.frequency"
    assert decode_input({"__text": "1e-"}) == DirectValue(raw="1e-")
    assert decode_input({"__expr": "$md.frequency"}) == EvalValue("$md.frequency")


def test_carrier_runtime_metadata_is_not_trusted_input() -> None:
    direct = DirectValue(3.0, raw="2.0", validation_error="stale")
    expr = EvalValue("md.f", resolved=900, error="stale", validation_error="stale")
    assert decode_input(encode_input(direct)) == DirectValue(raw="2.0")
    assert decode_input(encode_input(expr)) == EvalValue("md.f")


def test_decoding_detaches_nested_aliases() -> None:
    children = [1, 2]
    wire = {"section": {"values": children}}
    decoded = decode_input(wire)
    children.append(3)
    assert decoded == {"section": {"values": [1, 2]}}
    assert isinstance(decoded, dict)
    decoded["section"] = None
    assert wire == {"section": {"values": [1, 2, 3]}}


@pytest.mark.parametrize(
    "wire",
    [
        {"__once": "md.f"},
        {"__shape": "module"},
        {"__complex__": [1, 2]},
        {"__unknown": 1},
        {"__text": 1},
        {"__expr": None},
        {"__text": "x", "other": 1},
        {"__expr": "x", "__text": "x"},
        {"__complex": [True, 1]},
        {"__complex": [1, False]},
        {"__complex": [1]},
        {"__complex": "1j"},
        {"__ref": 1},
        {1: "x"},
        object(),
        (1, 2),
        1j,
    ],
)
def test_malformed_wire_is_typed_invalid_input(wire: object) -> None:
    with pytest.raises(CfgInputError, match=".") as caught:
        decode_input(wire)
    assert caught.value.reason is CfgInputReason.MALFORMED_INPUT
    assert caught.value.category is ExpectedErrorCategory.INVALID_INPUT
    assert caught.value.reason_code == "malformed_input"


@pytest.mark.parametrize(
    "wire",
    [
        math.nan,
        math.inf,
        -math.inf,
        {"__complex": [0, math.inf]},
        {"__complex": [10**400, 0]},
    ],
)
def test_nonfinite_wire_is_invalid_value(wire: object) -> None:
    with pytest.raises(CfgInputError, match="finite") as caught:
        decode_input(wire)
    assert caught.value.reason is CfgInputReason.INVALID_VALUE


@pytest.mark.parametrize("value", [complex(math.nan, 1), complex(1, math.inf)])
def test_nonfinite_complex_cannot_be_encoded(value: complex) -> None:
    with pytest.raises(CfgInputError, match="finite"):
        encode_input(value)


@pytest.mark.parametrize(
    "wire", ["", "00", "01", "+1", "-1", " 1", "1 ", "1.0", "١", 1, True, None]
)
def test_revision_requires_canonical_decimal_string(wire: object) -> None:
    with pytest.raises(CfgInputError, match="canonical decimal"):
        decode_revision(wire)


@pytest.mark.parametrize("wire", ["0", "1", str(2**100)])
def test_revision_roundtrip_is_lossless(wire: str) -> None:
    expected = {"cfg_id": "resource-a", "revision": wire}
    assert encode_ref(decode_ref(expected)) == expected


@pytest.mark.parametrize(
    "wire",
    [
        None,
        {},
        {"cfg_id": "a"},
        {"cfg_id": "", "revision": "0"},
        {"cfg_id": 1, "revision": "0"},
    ],
)
def test_ref_rejects_missing_or_invalid_identity(wire: object) -> None:
    with pytest.raises(CfgInputError, match="cfg_id"):
        decode_ref(wire)


def test_path_preserves_literal_segments_and_root() -> None:
    assert decode_path([]) == ()
    assert decode_path(["literal.dot", "0", "__ref"]) == ("literal.dot", "0", "__ref")


@pytest.mark.parametrize("wire", ["a.b", [""], [1], [None], ("a",)])
def test_path_rejects_non_array_or_empty_segments(wire: object) -> None:
    with pytest.raises(CfgInputError, match="path must"):
        decode_path(wire)


def test_batch_preserves_order_and_reports_failure_location() -> None:
    edits = decode_edits(
        [
            {"path": ["a"], "value": "1"},
            {"path": ["a"], "value": {"__text": "2"}},
        ]
    )
    assert [(edit.path, edit.value) for edit in edits] == [
        (("a",), "1"),
        (("a",), DirectValue(raw="2")),
    ]
    with pytest.raises(CfgInputError, match="unknown reserved") as caught:
        decode_edits(
            [
                {"path": ["a"], "value": 1},
                {"path": ["b"], "value": {"__once": "md.f"}},
            ]
        )
    assert caught.value.path == ("b",)
    assert caught.value.edit_index == 1


@pytest.mark.parametrize(
    "wire", [None, {}, [{}], [{"path": []}], [{"path": [], "value": 1, "extra": 2}]]
)
def test_batch_rejects_malformed_envelope(wire: object) -> None:
    with pytest.raises(CfgInputError, match="edits must|each edit must"):
        decode_edits(wire)


def test_empty_batch_is_decodable() -> None:
    assert decode_edits([]) == ()
