"""JSON comparison and historical cfg/identity validation pending producer move."""

import json
import re

from pydantic import JsonValue, TypeAdapter, ValidationError

from .container_models import JsonObject
from .native_models import CfgSnapshot

_JSON = TypeAdapter(JsonObject)


def json_values_equal(left: JsonValue, right: JsonValue) -> bool:
    """Compare finite JSON recursively, distinguishing boolean from number.

    Numbers compare by numeric value (1 equals 1.0). Object key order does not
    matter, but key sets and list order do. Compare unknown fields as well.
    Inputs must already be valid finite JSON; neither input is changed.
    """
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            json_values_equal(value, right[key]) for key, value in left.items()
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            json_values_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def validate_cfg_snapshot(cfg: CfgSnapshot) -> None:
    """Check finite cfg JSON, nonempty cfg_type and an ASCII major.minor version.

    Unknown document keys are valid. Do not inspect concrete cfg model fields or
    change the supplied snapshot. Invalid values raise ValueError naming /cfg.
    """
    try:
        _JSON.validate_json(json.dumps(cfg.values, allow_nan=False), strict=True)
    except (ValidationError, ValueError) as error:
        raise ValueError(f"/cfg: {error}") from error
    if re.fullmatch(r"[0-9]+\.[0-9]+", cfg.schema_version) is None:
        raise ValueError("/cfg: cfg_schema_version must be major.minor")
    if not cfg.cfg_type:
        raise ValueError("/cfg: cfg_type must not be empty")


def validate_run_id(run_id: str) -> None:
    """Check YYYYMMDDTHHMMSSZ-six_lowercase_alphanumerics without deriving time.

    Invalid strings raise ValueError naming run_id; nonstrings raise TypeError.
    Calendar semantics and acquisition evidence are not inferred from this token.
    """
    if re.fullmatch(r"[0-9]{8}T[0-9]{6}Z-[a-z0-9]{6}", run_id) is None:
        raise ValueError(
            "/: run_id must be a UTC timestamp plus six-character identity"
        )
