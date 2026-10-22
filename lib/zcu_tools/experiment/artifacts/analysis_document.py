"""UTF-8 analysis-result documents, independent of storage destinations."""

import json
import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

from zcu_tools.core.numeric_scalar import normalize_numpy_scalar
from zcu_tools.datafile import JsonObject


class AnalysisDocument(BaseModel):
    """One completed pane's JSON-safe inputs and numeric summary.

    pane identifies analysis or post_analysis. parameters captures the inputs
    used for that analysis; summary contains its JSON-safe result, not arrays,
    figures or a typed fit object. Direct summary values that are numpy
    floating/integer scalars become Python float/int with normal float rounding.
    Non-finite results, including finite extended floats that overflow Python
    float, raise ValueError. No conversion applies to parameters or nested
    summary values. Otherwise values must be finite strict JSON primitives,
    string-keyed objects and lists. Construction rejects invalid values with
    ValueError and isolates the supplied containers.
    """

    model_config = ConfigDict(frozen=True, strict=True, allow_inf_nan=False)

    pane: Literal["analysis", "post_analysis"]
    parameters: JsonObject
    summary: JsonObject

    def to_doc(self) -> JsonObject:
        """Return a detached zcu.analysis-result 1.0 JSON envelope for docs/analysis.

        Revalidate mutable nested values; non-finite or non-JSON edits raise
        ValueError. The container does not interpret this producer convention.
        """
        _check_json(self.parameters)
        _check_json(self.summary)
        validated = AnalysisDocument.model_validate(self.model_dump())
        return {
            "format": "zcu.analysis-result",
            "format_version": "1.0",
            "pane": validated.pane,
            "parameters": validated.parameters,
            "summary": validated.summary,
        }

    @field_validator("summary", mode="before")
    @classmethod
    def _normalize_summary(cls, value: object) -> object:
        if isinstance(value, dict):
            value = {key: normalize_numpy_scalar(item) for key, item in value.items()}
        _check_json(value)
        return value

    @field_validator("parameters", mode="before")
    @classmethod
    def _validate_json(cls, value: object) -> object:
        _check_json(value)
        return value


def _check_json(value: object) -> None:
    if value is None or type(value) in (bool, int, str):
        return
    if type(value) is float and math.isfinite(value):
        return
    if isinstance(value, list):
        for item in value:
            _check_json(item)
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for item in value.values():
            _check_json(item)
        return
    raise ValueError("Analysis values must be strict JSON with finite numbers")


def encode_analysis_document(document: AnalysisDocument) -> bytes:
    """Encode a validated pane as zcu.analysis-result 1.0 UTF-8 JSON.

    Include format, format_version, pane, parameters and summary. Perform no
    I/O and choose no destination. Invalid or subsequently mutated non-JSON
    or non-finite values raise ValueError.
    """
    return json.dumps(
        document.to_doc(),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
