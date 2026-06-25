from __future__ import annotations

import numpy as np
import pytest
from zcu_tools.analysis.fluxdep import (
    points_in_normalized_brush,
    toggle_near_mask,
)


def test_toggle_near_mask_select_then_erase() -> None:
    devs = np.linspace(-5.0, 5.0, 40).astype(np.float64)
    freqs = np.linspace(4.0, 5.0, 30).astype(np.float64)
    mask = np.zeros((len(devs), len(freqs)), dtype=bool)
    toggle_near_mask(devs, freqs, mask, 0.0, 4.5, 0.2, select=True)
    assert mask.any()
    selected_count = int(mask.sum())
    toggle_near_mask(devs, freqs, mask, 0.0, 4.5, 0.2, select=False)
    assert int(mask.sum()) < selected_count


def test_points_in_normalized_brush_matches_expected_region() -> None:
    xs = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    ys = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    mask = points_in_normalized_brush(
        xs,
        ys,
        x=0.5,
        y=0.5,
        width=0.1,
        x_bound=(0.0, 1.0),
        y_bound=(0.0, 1.0),
    )
    np.testing.assert_array_equal(mask, [False, True, False])


def test_points_in_normalized_brush_rejects_zero_span() -> None:
    xs = np.array([0.0], dtype=np.float64)
    ys = np.array([0.0], dtype=np.float64)
    with pytest.raises(ValueError, match="x_bound span"):
        points_in_normalized_brush(
            xs,
            ys,
            x=0.0,
            y=0.0,
            width=1.0,
            x_bound=(1.0, 1.0),
            y_bound=(0.0, 1.0),
        )
