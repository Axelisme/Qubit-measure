"""Closed editing-wire codec; deliberately separate from persisted cfg format.

Decoding establishes representation validity, not target writability or domain
validity. The resource applies those checks to each candidate in batch order.
Plain strings never request text parsing or expression evaluation.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence

from .model import DirectValue, EvalValue
from .resource import (
    CfgEdit,
    CfgId,
    CfgInput,
    CfgInputError,
    CfgInputReason,
    CfgPath,
    CfgRef,
    CfgRevision,
)


def decode_revision(value: object) -> CfgRevision:
    if not isinstance(value, str) or re.fullmatch(r"0|[1-9][0-9]*", value) is None:
        raise CfgInputError(
            CfgInputReason.MALFORMED_INPUT,
            "revision must be a canonical decimal string",
        )
    return CfgRevision(int(value))


def decode_path(value: object) -> CfgPath:
    if not isinstance(value, list) or any(
        not isinstance(part, str) or not part for part in value
    ):
        raise CfgInputError(
            CfgInputReason.MALFORMED_INPUT,
            "path must be an array of nonempty string segments",
        )
    return tuple(value)


def decode_ref(value: object) -> CfgRef:
    if not isinstance(value, dict) or set(value) != {"cfg_id", "revision"}:
        raise CfgInputError(
            CfgInputReason.MALFORMED_INPUT, "expected must contain cfg_id and revision"
        )
    return CfgRef(CfgId(value["cfg_id"]), decode_revision(value["revision"]))


def encode_ref(ref: CfgRef) -> dict[str, str]:
    return {"cfg_id": str(ref.cfg_id), "revision": str(ref.revision)}


def decode_edits(value: object) -> tuple[CfgEdit, ...]:
    if not isinstance(value, list):
        raise CfgInputError(CfgInputReason.MALFORMED_INPUT, "edits must be an array")
    result: list[CfgEdit] = []
    for index, item in enumerate(value):
        path: CfgPath | None = None
        try:
            if not isinstance(item, dict) or set(item) != {"path", "value"}:
                raise CfgInputError(
                    CfgInputReason.MALFORMED_INPUT,
                    "each edit must contain path and value",
                )
            path = decode_path(item["path"])
            result.append(CfgEdit(path, decode_input(item["value"])))
        except CfgInputError as exc:
            raise CfgInputError(
                exc.reason, str(exc), path=path, edit_index=index
            ) from exc
    return tuple(result)


def decode_input(value: object) -> CfgInput:
    """Decode JSON-compatible input, allocating all mutable containers anew."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        _require_finite(value)
        return value
    if isinstance(value, list):
        return [decode_input(item) for item in value]
    if isinstance(value, dict):
        return _decode_object(value)
    raise CfgInputError(
        CfgInputReason.MALFORMED_INPUT, "input must use the editing wire representation"
    )


def _decode_object(value: dict[object, object]) -> CfgInput:
    if any(not isinstance(key, str) for key in value):
        raise CfgInputError(
            CfgInputReason.MALFORMED_INPUT, "object keys must be strings"
        )
    tags = {key for key in value if isinstance(key, str) and key.startswith("__")}
    if tags & {"__complex", "__text", "__expr"}:
        if len(value) != 1:
            raise CfgInputError(
                CfgInputReason.MALFORMED_INPUT, "scalar tag must be the only object key"
            )
        return _decode_scalar(next(iter(tags)), next(iter(value.values())))
    if tags - {"__ref"}:
        raise CfgInputError(
            CfgInputReason.MALFORMED_INPUT, "unknown reserved input tag"
        )
    if (
        "__ref" in value
        and value["__ref"] is not None
        and not isinstance(value["__ref"], str)
    ):
        raise CfgInputError(
            CfgInputReason.MALFORMED_INPUT, "__ref must be a source key or null"
        )
    return {str(key): decode_input(item) for key, item in value.items()}


def _decode_scalar(tag: str, value: object) -> DirectValue | EvalValue | complex:
    if tag == "__complex":
        if not isinstance(value, list) or len(value) != 2:
            raise CfgInputError(
                CfgInputReason.MALFORMED_INPUT, "__complex must contain two numbers"
            )
        return complex(_component(value[0]), _component(value[1]))
    if not isinstance(value, str):
        raise CfgInputError(
            CfgInputReason.MALFORMED_INPUT, f"{tag} must contain a string"
        )
    if tag == "__text":
        return DirectValue(raw=value)
    return EvalValue(value)


def _component(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CfgInputError(
            CfgInputReason.MALFORMED_INPUT,
            "complex components must be numbers, not bool",
        )
    try:
        result = float(value)
    except OverflowError as exc:
        raise CfgInputError(
            CfgInputReason.INVALID_VALUE,
            "complex component is outside finite float range",
        ) from exc
    _require_finite(result)
    return result


def _require_finite(value: float) -> None:
    if not math.isfinite(value):
        raise CfgInputError(CfgInputReason.INVALID_VALUE, "input must be finite")


def encode_input(value: CfgInput) -> object:
    """Encode input intent only; resolved values and diagnostics are not inputs."""
    if isinstance(value, DirectValue):
        value = {"__text": value.raw} if value.raw is not None else value.value
    if isinstance(value, EvalValue):
        return {"__expr": value.expr}
    if isinstance(value, complex):
        _require_finite(value.real)
        _require_finite(value.imag)
        return {"__complex": [value.real, value.imag]}
    if isinstance(value, Mapping):
        # Reuse the decoder's reserved-tag rules instead of maintaining two grammars.
        encoded = {key: encode_input(item) for key, item in value.items()}
        decode_input(encoded)
        return encoded
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [encode_input(item) for item in value]
    return decode_input(value)
