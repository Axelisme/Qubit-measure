"""Public JSON projection of analysis values, independent of result serializers."""

import re
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import pytest
from pydantic import BaseModel, field_serializer
from zcu_tools.gui.app.measure import summary


class Fit(BaseModel):
    frequency: float

    @field_serializer("frequency")
    def serialize_frequency(self, value: float) -> str:
        return f"{value} MHz"


@dataclass
class Result:
    fit: Fit
    values: object

    def to_summary_dict(self) -> dict[str, str]:
        raise AssertionError("Custom serializers must not run")


def test_summary_projects_fields_arrays_and_complex_with_nonfinite_paths() -> None:
    result = Result(
        Fit(frequency=1250.0),
        {
            "grid": np.array([[1.0, np.nan], [np.inf, -np.inf]]),
            "scalar": np.int64(3),
            "flag": np.bool_(True),
            "pair": (np.float32(2.5), complex(4.0, np.nan)),
        },
    )

    projected = summary.summarize_analysis(result)

    assert projected.summary == {
        "fit": {"frequency": 1250.0},
        "values": {
            "grid": [[1.0, None], [None, None]],
            "scalar": 3,
            "flag": True,
            "pair": [2.5, {"real": 4.0, "imag": None}],
        },
    }
    assert [(issue.path, issue.reason) for issue in projected.invalid] == [
        ("summary.values.grid[0][1]", "non_finite"),
        ("summary.values.grid[1][0]", "non_finite"),
        ("summary.values.grid[1][1]", "non_finite"),
        ("summary.values.pair[1].imag", "non_finite"),
    ]
    assert isinstance(projected.invalid, tuple)


@pytest.mark.parametrize(
    "value",
    [
        np.arange(65),
        np.arange(65).reshape(5, 13),
        list(range(65)),
        tuple(range(65)),
        [list(range(13)) for _ in range(5)],
        (list(range(32)), np.arange(33)),
    ],
)
def test_summary_rejects_oversized_sequences_with_field_path(value: object) -> None:
    with pytest.raises(ValueError, match=r"summary\.gg\.excitation") as error:
        summary.summarize_analysis({"gg": {"excitation": value}})

    message = str(error.value)
    assert "65" in message
    assert "64" in message
    assert "analysis return values must not grow with scan size" in message


@pytest.mark.parametrize(
    "value",
    [
        np.arange(64).reshape(8, 8),
        list(range(64)),
        tuple(range(64)),
        [list(range(8)) for _ in range(8)],
        (list(range(32)), np.arange(32)),
        [],
        np.empty((0, 65)),
    ],
)
def test_summary_accepts_sequences_up_to_64_flattened_elements(value: object) -> None:
    projected = summary.summarize_analysis({"value": value})
    assert projected.invalid == ()
    assert projected.summary["value"] is not None


@pytest.mark.parametrize(
    "value",
    [Fit(frequency=1250.0), MappingProxyType({"frequency": 1250.0})],
)
def test_summary_projects_model_and_mapping_roots(value: object) -> None:
    projected = summary.summarize_analysis(value)

    assert projected.summary == {"frequency": 1250.0}
    assert projected.invalid == ()


@pytest.mark.parametrize(
    ("value", "expected", "paths"),
    [
        (None, None, []),
        (np.array(7), 7, []),
        (np.longdouble(2.5), 2.5, []),
        (np.clongdouble(2 + 3j), {"real": 2.0, "imag": 3.0}, []),
        (float("inf"), None, ["summary.value"]),
        ([np.float64("nan")], [None], ["summary.value[0]"]),
        ("ready", "ready", []),
    ],
)
def test_summary_applies_projection_to_nested_values(value, expected, paths) -> None:
    projected = summary.summarize_analysis({"value": value})
    assert projected.summary == {"value": expected}
    assert [issue.path for issue in projected.invalid] == paths


@pytest.mark.parametrize(
    "value",
    [
        None,
        True,
        7,
        2.5,
        float("nan"),
        float("inf"),
        "ready",
        b"unsupported",
        [],
        [1],
        (1,),
        np.array(7),
        np.array([1]),
        np.int64(3),
        np.longdouble(2.5),
        2 + 3j,
        np.clongdouble(2 + 3j),
        Result,
        Fit,
        object(),
    ],
)
def test_summary_rejects_non_object_roots_with_type_and_path(value: object) -> None:
    with pytest.raises(TypeError) as error:
        summary.summarize_analysis(value)

    assert "summary" in str(error.value)
    assert type(value).__name__ in str(error.value)


@pytest.mark.parametrize(
    ("value", "path"),
    [
        ({"fit": {1: "not a string key"}}, "summary.fit"),
        ({"fit": [object()]}, "summary.fit[0]"),
        ({1: "not a string key"}, "summary"),
    ],
)
def test_summary_rejects_unsupported_values_with_their_path(value, path) -> None:
    with pytest.raises(TypeError, match=re.escape(path)):
        summary.summarize_analysis(value)
