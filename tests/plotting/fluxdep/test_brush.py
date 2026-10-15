from __future__ import annotations

import math

import pytest
from matplotlib.figure import Figure
from zcu_tools.plotting.fluxdep.brush import BrushBand


@pytest.mark.parametrize(
    ("x_span", "y_span"),
    [
        (0.0, 1.0),
        (1.0, 0.0),
        (-1.0, 1.0),
        (1.0, -2.0),
        (math.nan, 1.0),
        (1.0, math.nan),
    ],
)
def test_brush_band_rejects_a_span_that_is_not_positive(x_span, y_span):
    axes = Figure().add_subplot()
    with pytest.raises(ValueError, match="span"):
        BrushBand(axes, x_span=x_span, y_span=y_span, zorder=6)


def test_brush_band_accepts_positive_spans():
    axes = Figure().add_subplot()
    BrushBand(axes, x_span=2.0, y_span=0.5, zorder=6)
    assert len(axes.patches) == 1 and len(axes.lines) == 1
