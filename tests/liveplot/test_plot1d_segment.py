from __future__ import annotations

from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pytest
from zcu_tools.liveplot.segments import Plot1DSegment


def test_plot1d_segment_rejects_complex_signals() -> None:
    fig, ax = plt.subplots()
    try:
        segment = Plot1DSegment("x", "y")
        segment.init_ax(ax)

        xs = np.array([0.0, 1.0], dtype=np.float64)
        signals = np.array([1.0 + 2.0j, 3.0 + 4.0j], dtype=np.complex128)

        with pytest.raises(ValueError, match="complex"):
            segment.update(ax, xs, cast(Any, signals))
    finally:
        plt.close(fig)
