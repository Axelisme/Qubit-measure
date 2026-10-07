"""UTF-8 analysis-result documents, independent of storage destinations."""

from typing import Literal

from pydantic import BaseModel, ConfigDict

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


def encode_analysis_document(document: AnalysisDocument) -> bytes:
    """Encode a validated pane as zcu.analysis-result 1.0 UTF-8 JSON.

    Include format, format_version, pane, parameters and summary. Perform no
    I/O and choose no destination. Invalid or subsequently mutated non-JSON
    or non-finite values raise ValueError.
    """
    raise NotImplementedError
