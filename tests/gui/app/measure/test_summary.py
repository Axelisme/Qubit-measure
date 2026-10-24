"""Public JSON projection of analysis values, independent of result serializers."""

import re
from dataclasses import dataclass

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
    ("value", "expected", "paths"),
    [
        (None, None, []),
        (np.array(7), 7, []),
        (np.longdouble(2.5), 2.5, []),
        (np.clongdouble(2 + 3j), {"real": 2.0, "imag": 3.0}, []),
        (float("inf"), None, ["summary"]),
        ([np.float64("nan")], [None], ["summary[0]"]),
        ("ready", "ready", []),
    ],
)
def test_summary_applies_projection_to_root_values(value, expected, paths) -> None:
    projected = summary.summarize_analysis(value)
    assert projected.summary == expected
    assert [issue.path for issue in projected.invalid] == paths


@pytest.mark.parametrize(
    ("value", "path"),
    [
        ({"fit": {1: "not a string key"}}, "summary.fit"),
        ({"fit": [object()]}, "summary.fit[0]"),
        (b"unsupported", "summary"),
    ],
)
def test_summary_rejects_unsupported_values_with_their_path(value, path) -> None:
    with pytest.raises(TypeError, match=re.escape(path)):
        summary.summarize_analysis(value)
