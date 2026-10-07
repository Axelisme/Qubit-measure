"""UTF-8 analysis-result documents, independent of storage destinations."""

import json
import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

from .native_models import JsonObject


class AnalysisDocument(BaseModel):
    """One completed pane's JSON-safe inputs and numeric summary.

    pane identifies analysis or post_analysis. parameters captures the inputs
    used for that analysis; summary contains its JSON-safe result, not arrays,
    figures or a typed fit object. Values must be finite JSON primitives,
    string-keyed objects and lists. Construction rejects invalid values with
    ValueError and isolates the supplied containers.
    """

    model_config = ConfigDict(frozen=True, strict=True, allow_inf_nan=False)

    pane: Literal["analysis", "post_analysis"]
    parameters: JsonObject
    summary: JsonObject

    @field_validator("parameters", "summary", mode="before")
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
    _check_json(document.parameters)
    _check_json(document.summary)
    validated = AnalysisDocument.model_validate(document.model_dump())
    return json.dumps(
        {
            "format": "zcu.analysis-result",
            "format_version": "1.0",
            **validated.model_dump(),
        },
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
