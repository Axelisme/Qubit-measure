"""JSON projection of workflow point records, with diagnostic-only degradation."""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import fields, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

import numpy as np
from pydantic import BaseModel, JsonValue, TypeAdapter

from .models import EncodedRecord

_logger = logging.getLogger(__name__)


def encode_record(record: object) -> EncodedRecord:
    """Project a record to detached JSON, degrading unsupported data to repr.

    Dataclasses, Pydantic models, numpy scalars/arrays, complex numbers,
    datetimes, Paths, and Enums are supported. Complex values become real/imag
    pairs; nonfinite floats become null. Arrays warn because records should
    contain summaries, not curves. Unsupported data or cycles warn and return
    mode='repr' with a diagnostic error; a broken repr gets a type-labelled
    placeholder. Encoded UTF-8 envelopes over 256 KiB warn but are not truncated.
    No state copying or file I/O is performed or degraded by this function.
    """
    try:
        serialized = json.dumps(record, default=_record_default, ensure_ascii=False)
        serialized.encode("utf-8")
        value = TypeAdapter(JsonValue).validate_python(
            json.loads(serialized, parse_constant=_nonfinite)
        )
        encoded = EncodedRecord("json", value, None)
    except Exception as error:
        # Serialization is the sole recoverable failure at this boundary.
        _logger.exception("Workflow record serialization failed")
        encoded = _degraded(record, error)
    size = len(
        json.dumps(
            {
                "mode": encoded.mode,
                "value": encoded.value,
                "error": encoded.error,
            },
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    )
    if size > 256 * 1024:
        warnings.warn(
            f"Encoded workflow record is {size} bytes, exceeding 256 KiB",
            UserWarning,
            stacklevel=2,
        )
    return encoded


def _record_default(value: object) -> object:
    # JSON's circular-reference guard owns traversal, including converted objects.
    if isinstance(value, type):
        raise TypeError(f"Record contains a class: {value.__qualname__}")
    if is_dataclass(value):
        return {field.name: getattr(value, field.name) for field in fields(value)}
    if isinstance(value, BaseModel):
        return value.model_dump(mode="python")
    if isinstance(value, (np.ndarray, np.generic)):
        return _numpy_value(value)
    if isinstance(value, complex):
        return [value.real, value.imag]
    if isinstance(value, (datetime, Path)):
        return str(value) if isinstance(value, Path) else value.isoformat()
    if isinstance(value, Enum):
        return value.value
    raise TypeError(
        f"Unsupported record value: {type(value).__module__}.{type(value).__qualname__}"
    )


def _numpy_value(value: np.ndarray | np.generic) -> object:
    if isinstance(value, np.ndarray):
        warnings.warn(
            "Workflow record contains an ndarray; store curves in iteration files",
            UserWarning,
            stacklevel=4,
        )
        return value.tolist()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.complexfloating):
        return complex(value)
    return value.item()


def _nonfinite(_value: str) -> None:
    return None


def _degraded(record: object, error: Exception) -> EncodedRecord:
    diagnostic = _error_description(error)
    try:
        value = repr(record).encode("utf-8", errors="backslashreplace").decode("utf-8")
    except Exception as repr_error:
        # A record's own diagnostic hook must not defeat record degradation.
        _logger.exception("Workflow record repr failed")
        name = f"{type(record).__module__}.{type(record).__qualname__}"
        value = f"<{name}: repr failed>"
        diagnostic += f"; repr failed: {_error_description(repr_error)}"
    warnings.warn(
        f"Workflow record degraded to repr: {diagnostic}",
        UserWarning,
        stacklevel=3,
    )
    return EncodedRecord("repr", value, diagnostic)


def _error_description(error: Exception) -> str:
    try:
        message = str(error).encode("utf-8", errors="backslashreplace").decode("utf-8")
    except Exception:
        # Exception formatting can itself be user code; retain its type instead.
        _logger.exception("Workflow serialization error formatting failed")
        message = "<message unavailable>"
    return f"{type(error).__module__}.{type(error).__qualname__}: {message}"
