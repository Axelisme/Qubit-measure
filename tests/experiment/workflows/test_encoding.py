"""Observable workflow record projection and diagnostic degradation contracts."""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import numpy as np
import pytest
from pydantic import BaseModel
from zcu_tools.experiment.workflows.encoding import encode_record


class Quality(Enum):
    ACCEPTED = "accepted"


class Fit(BaseModel):
    value_us: float
    quality: Quality


@dataclass
class Point:
    fit: Fit
    time: datetime
    destination: Path
    signal: complex
    count: np.int64


def test_record_projects_supported_nested_values_to_detached_json(
    tmp_path: Path,
) -> None:
    point = Point(
        Fit(value_us=20.0, quality=Quality.ACCEPTED),
        datetime(2026, 10, 6, tzinfo=timezone.utc),
        tmp_path / "fit.json",
        2 + 3j,
        np.int64(4),
    )
    encoded = encode_record(point)
    point.fit.value_us = 99.0
    assert encoded.mode == "json"
    assert encoded.error is None
    assert encoded.value == {
        "fit": {"value_us": 20.0, "quality": "accepted"},
        "time": "2026-10-06T00:00:00+00:00",
        "destination": str(tmp_path / "fit.json"),
        "signal": [2.0, 3.0],
        "count": 4,
    }


def test_nonfinite_numbers_become_null_in_nested_records() -> None:
    encoded = encode_record(
        {
            "missing": float("nan"),
            "bounds": (float("inf"), float("-inf")),
            "signal": complex(float("nan"), 2),
            "valid": np.float64(1.25),
        }
    )
    assert encoded.mode == "json"
    assert encoded.value == {
        "missing": None,
        "bounds": [None, None],
        "signal": [None, 2.0],
        "valid": 1.25,
    }


def test_numpy_arrays_warn_but_keep_complete_shape_and_samples() -> None:
    with pytest.warns(UserWarning, match="contains an ndarray"):
        encoded = encode_record(np.array([[1 + 2j, complex(float("nan"), 3)]]))
    assert encoded.mode == "json"
    assert encoded.value == [[[1.0, 2.0], [None, 3.0]]]


@pytest.mark.parametrize(
    "value",
    [
        np.bool_(True),
        np.int32(-2),
        np.longdouble(1.5),
        np.complex128(2 + 3j),
        np.str_("ok"),
    ],
)
def test_numpy_scalar_records_are_json_values(value: np.generic) -> None:
    encoded = encode_record(value)
    assert encoded.mode == "json"
    assert encoded.error is None
    if isinstance(value, np.complexfloating):
        assert encoded.value == [2.0, 3.0]
    else:
        assert encoded.value == value.item()


class Unsupported:
    def __repr__(self) -> str:
        return "unsupported-point"


def test_unsupported_record_degrades_to_repr_with_error() -> None:
    with pytest.warns(UserWarning, match="degraded to repr"):
        encoded = encode_record(Unsupported())
    assert encoded.mode == "repr"
    assert encoded.value == "unsupported-point"
    assert encoded.error is not None
    assert "Unsupported record value" in encoded.error


def test_circular_record_degrades_without_stopping_producer() -> None:
    points: list[object] = []
    points.append(points)
    with pytest.warns(UserWarning, match="degraded to repr"):
        encoded = encode_record(points)
    assert encoded.mode == "repr"
    assert encoded.value == "[[...]]"
    assert encoded.error is not None
    assert "Circular reference" in encoded.error


@dataclass
class LinkedPoint:
    next: object | None = None


def test_circular_dataclass_degrades_without_recursive_asdict() -> None:
    point = LinkedPoint()
    point.next = point
    with pytest.warns(UserWarning, match="degraded to repr"):
        encoded = encode_record(point)
    assert encoded.mode == "repr"
    assert encoded.error is not None
    assert "Circular reference" in encoded.error


class BrokenRepr:
    def __repr__(self) -> str:
        raise RuntimeError("repr unavailable")


def test_repr_failure_keeps_type_and_both_error_diagnostics() -> None:
    with pytest.warns(UserWarning, match="degraded to repr"):
        encoded = encode_record(BrokenRepr())
    assert encoded.mode == "repr"
    assert isinstance(encoded.value, str)
    assert "BrokenRepr" in encoded.value
    assert "repr failed" in encoded.value
    assert encoded.error is not None
    assert "Unsupported record value" in encoded.error
    assert "repr unavailable" in encoded.error


def test_invalid_utf8_record_degrades_to_printable_repr() -> None:
    record = "\ud800"
    with pytest.warns(UserWarning, match="degraded to repr"):
        encoded = encode_record(record)
    assert encoded.mode == "repr"
    assert encoded.value == repr(record)
    assert encoded.error is not None
    assert "UnicodeEncodeError" in encoded.error


def test_large_multibyte_record_warns_without_truncating() -> None:
    record = "量" * 90_000
    with pytest.warns(UserWarning, match="exceeding 256 KiB"):
        encoded = encode_record(record)
    assert encoded.mode == "json"
    assert encoded.value == record
