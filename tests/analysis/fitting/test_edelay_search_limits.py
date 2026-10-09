"""Resource and adaptive-radius limits of electrical-delay branch search."""

from __future__ import annotations

import numpy as np
import pytest
import zcu_tools.analysis.fitting.resonance.base as resonance_base
from zcu_tools.analysis.fitting.resonance import find_edelay_branch


def test_find_edelay_branch_fast_fails_before_oversized_radius_overflow() -> None:
    freqs = np.asarray([5000.0, 5000.2, 5000.55, 5001.0])
    signals = np.exp(-1j * 2.0 * np.pi * freqs * 0.2)

    with np.errstate(over="raise"):
        with pytest.raises(ValueError, match="candidate resource limit"):
            find_edelay_branch(
                freqs,
                signals,
                search_radius=np.finfo(np.float64).max,
            )


def test_find_edelay_branch_rejects_optimum_at_search_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    freqs = np.asarray([0.0, 0.2, 0.55, 0.8])
    candidate_step = 1.0 / (8.0 * np.ptp(freqs))
    boundary_delay = 3.0 * candidate_step
    signals = np.exp(-1j * 2.0 * np.pi * freqs * boundary_delay)
    monkeypatch.setattr(resonance_base, "get_rough_edelay", lambda *_args: 0.0)

    with pytest.raises(ValueError, match="search boundary"):
        find_edelay_branch(freqs, signals, search_radius=boundary_delay)


def test_find_edelay_branch_expands_boundary_limited_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    freqs = np.asarray([0.0, 0.2, 0.55, 0.8])
    candidate_step = 1.0 / (8.0 * np.ptp(freqs))
    boundary_delay = 3.0 * candidate_step
    signals = np.exp(-1j * 2.0 * np.pi * freqs * boundary_delay)
    monkeypatch.setattr(resonance_base, "get_rough_edelay", lambda *_args: 0.0)

    estimated = find_edelay_branch(
        freqs,
        signals,
        search_radius=boundary_delay,
        max_search_radius=2.0 * boundary_delay,
    )

    assert estimated == pytest.approx(boundary_delay)


def test_find_edelay_branch_still_fails_at_adaptive_search_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    freqs = np.asarray([0.0, 0.2, 0.55, 0.8])
    candidate_step = 1.0 / (8.0 * np.ptp(freqs))
    boundary_delay = 6.0 * candidate_step
    signals = np.exp(-1j * 2.0 * np.pi * freqs * boundary_delay)
    monkeypatch.setattr(resonance_base, "get_rough_edelay", lambda *_args: 0.0)

    with pytest.raises(ValueError, match="max_search_radius"):
        find_edelay_branch(
            freqs,
            signals,
            search_radius=3.0 * candidate_step,
            max_search_radius=boundary_delay,
        )
