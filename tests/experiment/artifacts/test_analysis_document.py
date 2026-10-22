"""Analysis JSON format through its public model and encoder."""

import json

import numpy as np
import pytest
from zcu_tools.experiment.artifacts import AnalysisDocument, encode_analysis_document


@pytest.mark.parametrize("pane", ["analysis", "post_analysis"])
def test_encoder_preserves_json_safe_parameters_and_summary(pane: str) -> None:
    document = AnalysisDocument.model_validate(
        {
            "pane": pane,
            "parameters": {"skip": 2, "label": "測試", "weights": [1.0, None]},
            "summary": {"frequency": 5000.0, "accepted": True},
        }
    )
    encoded = encode_analysis_document(document)
    assert isinstance(encoded, bytes)
    assert json.loads(encoded.decode("utf-8")) == {
        "format": "zcu.analysis-result",
        "format_version": "1.0",
        "pane": pane,
        "parameters": document.parameters,
        "summary": document.summary,
    }
    assert "測試".encode() in encoded


@pytest.mark.parametrize(
    "values",
    [
        {"pane": "unknown", "parameters": {}, "summary": {}},
        {"pane": "analysis", "parameters": {"bad": {1: 2}}, "summary": {}},
        {"pane": "analysis", "parameters": {}, "summary": {"bad": float("nan")}},
        {"pane": "analysis", "parameters": {}, "summary": {"bad": float("inf")}},
        {"pane": "analysis", "parameters": {}, "summary": {"bad": (1, 2)}},
    ],
)
def test_format_entry_rejects_non_json_or_non_finite_input(values: object) -> None:
    with pytest.raises(ValueError, match="pane|parameters|summary|JSON|finite"):
        AnalysisDocument.model_validate(values)


@pytest.mark.parametrize(
    "value,expected",
    [
        pytest.param(np.float16(-0.125), -0.125, id="float16"),
        pytest.param(np.float32(0.93), 0.9300000071525574, id="float32"),
        pytest.param(np.float64(1.2345678901234567), 1.2345678901234567, id="float64"),
        pytest.param(np.longdouble("1.0000000000000000001"), 1.0, id="longdouble"),
        pytest.param(np.int8(-128), -128, id="int8"),
        pytest.param(np.int16(-32768), -32768, id="int16"),
        pytest.param(np.int32(-2147483648), -2147483648, id="int32"),
        pytest.param(np.int64(-9223372036854775808), -9223372036854775808, id="int64"),
        pytest.param(np.uint8(255), 255, id="uint8"),
        pytest.param(np.uint16(65535), 65535, id="uint16"),
        pytest.param(np.uint32(4294967295), 4294967295, id="uint32"),
        pytest.param(
            np.uint64(18446744073709551615), 18446744073709551615, id="uint64"
        ),
    ],
)
def test_summary_numpy_scalar_normalizes_and_encodes(
    value: object, expected: int | float
) -> None:
    summary = {"value": value, "details": {"weights": [1.0, None], "accepted": True}}
    try:
        document = AnalysisDocument.model_validate(
            {"pane": "analysis", "parameters": {}, "summary": summary}
        )
    except ValueError as error:
        raise AssertionError(
            "AnalysisDocument must accept finite numpy scalars as direct summary values"
        ) from error
    assert type(document.summary["value"]) is type(expected)
    assert document.summary["value"] == expected
    stored = json.loads(encode_analysis_document(document))["summary"]
    assert type(stored["value"]) is type(expected)
    assert stored == {
        "value": expected,
        "details": {"weights": [1.0, None], "accepted": True},
    }
    assert summary["value"] is value
    summary["details"] = {"changed": True}
    assert document.summary["details"] == {"weights": [1.0, None], "accepted": True}


@pytest.mark.parametrize(
    "scalar_type", [np.float16, np.float32, np.float64, np.longdouble]
)
@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_summary_numpy_scalar_rejects_nonfinite(
    scalar_type: type[np.floating], value: str
) -> None:
    with pytest.raises(ValueError, match="strict JSON|finite"):
        AnalysisDocument.model_validate(
            {
                "pane": "analysis",
                "parameters": {},
                "summary": {"value": scalar_type(value)},
            }
        )


@pytest.mark.skipif(
    bool(np.finfo(np.longdouble).max <= np.finfo(np.float64).max),
    reason="longdouble has no extended range on this platform",
)
def test_summary_numpy_scalar_rejects_finite_float_overflow() -> None:
    value = np.longdouble(np.finfo(np.longdouble).max)
    assert np.isfinite(value)
    with pytest.raises(ValueError, match="strict JSON|finite"):
        AnalysisDocument.model_validate(
            {"pane": "analysis", "parameters": {}, "summary": {"value": value}}
        )


@pytest.mark.parametrize(
    "value",
    [
        pytest.param({"nested": np.float64(1.25)}, id="nested-float64"),
        pytest.param([np.float32(1.25)], id="nested-float32"),
        pytest.param({"count": np.int64(2)}, id="nested-integer"),
        pytest.param(np.array(1.25), id="zero-dimensional-array"),
        pytest.param(np.array([1.25]), id="array"),
        pytest.param(np.bool_(True), id="numpy-bool"),
        pytest.param(1 + 2j, id="complex"),
        pytest.param((1, 2), id="tuple"),
        pytest.param(b"bytes", id="bytes"),
        pytest.param({1: "value"}, id="non-string-key"),
        pytest.param(object(), id="object"),
    ],
)
def test_summary_numpy_scalar_does_not_relax_nested_json(value: object) -> None:
    with pytest.raises(ValueError, match="strict JSON|finite"):
        AnalysisDocument.model_validate(
            {"pane": "analysis", "parameters": {}, "summary": {"value": value}}
        )


@pytest.mark.parametrize("value", [np.float32(1.25), np.float64(1.25), np.int64(2)])
def test_numpy_scalar_in_parameters_remains_invalid(value: object) -> None:
    with pytest.raises(ValueError, match="strict JSON|finite"):
        AnalysisDocument.model_validate(
            {"pane": "analysis", "parameters": {"value": value}, "summary": {}}
        )


def test_document_isolates_input_containers_and_encoder_rechecks_mutation() -> None:
    parameters = {"nested": [1.0]}
    document = AnalysisDocument.model_validate(
        {"pane": "analysis", "parameters": parameters, "summary": {"value": 2.0}}
    )
    parameters["nested"].append(3.0)
    assert json.loads(encode_analysis_document(document))["parameters"] == {
        "nested": [1.0]
    }
    document.summary["value"] = float("nan")
    with pytest.raises(ValueError, match="JSON|finite|range|NaN"):
        encode_analysis_document(document)
