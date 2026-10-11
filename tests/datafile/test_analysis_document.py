"""Analysis JSON format through its public model and encoder."""

import json

import pytest
from zcu_tools.datafile import AnalysisDocument, encode_analysis_document


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
