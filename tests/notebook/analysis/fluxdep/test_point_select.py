from __future__ import annotations

import numpy as np
import pytest
from zcu_tools.notebook.analysis.fluxdep.interactive import point_select
from zcu_tools.notebook.analysis.fluxdep.interactive.point_select import (
    InteractiveSelector,
)
from zcu_tools.notebook.persistance import SpectrumResult


def _spectrum_result() -> SpectrumResult:
    fluxs = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    freqs = np.array([4.0, 4.5, 5.0], dtype=np.float64)
    return {
        "flux_half": 0.0,
        "flux_int": 1.0,
        "flux_period": 2.0,
        "spectrum": {
            "dev_values": fluxs,
            "fluxs": fluxs,
            "freqs": freqs,
            "signals": np.ones((3, 3), dtype=np.complex128),
        },
        "points": {
            "dev_values": fluxs,
            "fluxs": fluxs,
            "freqs": freqs,
        },
    }


def test_selector_initial_filter_mask_matches_selected_points(monkeypatch) -> None:
    monkeypatch.setattr(point_select, "display", lambda _widget: None)
    selected = np.array([True, False, True], dtype=bool)

    selector = InteractiveSelector({"sample": _spectrum_result()}, selected=selected)

    assert selector.filter_mask.shape == (2,)
    np.testing.assert_array_equal(selector.get_cur_selected(), selected)
    selector.finish_interactive()


def test_selector_rejects_misaligned_selected_mask(monkeypatch) -> None:
    monkeypatch.setattr(point_select, "display", lambda _widget: None)

    with pytest.raises(ValueError, match="selected"):
        InteractiveSelector(
            {"sample": _spectrum_result()},
            selected=np.array([True, False], dtype=bool),
        )
