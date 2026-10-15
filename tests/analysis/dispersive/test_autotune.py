"""Public scoring and tuning with real dispersive predictions."""

from __future__ import annotations

import numpy as np
import pytest
from zcu_tools.analysis.dispersive.autotune import auto_tune, sample_score
from zcu_tools.analysis.dispersive.predict import predict_dispersive_at


def _grid():
    return np.linspace(0.0, 0.5, 26), np.linspace(5.0, 6.0, 40)


@pytest.mark.parametrize("phase_sign", [1.0, -1.0])
def test_sample_score_picks_max_of_ground_excited(phase_sign):
    sp_fluxs, sp_freqs = _grid()
    norm = np.tile(0.5 + phase_sign * (sp_freqs - 5.5), (len(sp_fluxs), 1))
    samples = np.array([0.1, 0.3])
    ground, excited = predict_dispersive_at((4.0, 1.0, 0.5), samples, 0.06, 5.3)
    assert np.all(np.abs(ground - excited) > 0.001)

    score = sample_score((4.0, 1.0, 0.5), sp_fluxs, sp_freqs, norm, samples, 0.06, 5.3)

    ground_phase = 0.5 + phase_sign * (ground - 5.5)
    excited_phase = 0.5 + phase_sign * (excited - 5.5)
    assert score == pytest.approx(np.maximum(ground_phase, excited_phase).mean())


def test_auto_tune_finds_the_phase_peak():
    sp_fluxs, sp_freqs = _grid()
    norm = np.tile(np.exp(-((sp_freqs - 5.5) ** 2) / (2 * 0.2**2)), (len(sp_fluxs), 1))

    # With zero coupling both real dressed lines equal the bare resonator frequency.
    g, bare_rf = auto_tune(
        (4.0, 1.0, 0.5),
        sp_fluxs,
        sp_freqs,
        norm,
        np.array([0.1, 0.25, 0.4]),
        g0=0.0,
        bare_rf0=5.35,
        g_bounds=(0.0, 0.0),
        rf_bounds=(5.0, 6.0),
    )

    assert g == 0.0
    assert abs(bare_rf - 5.5) < 0.05


def test_auto_tune_respects_bounds():
    sp_fluxs, sp_freqs = _grid()
    norm = np.tile(np.exp(-((sp_freqs - 5.9) ** 2) / (2 * 0.03**2)), (len(sp_fluxs), 1))

    g, bare_rf = auto_tune(
        (4.0, 1.0, 0.5),
        sp_fluxs,
        sp_freqs,
        norm,
        np.array([0.2]),
        g0=0.0,
        bare_rf0=5.3,
        g_bounds=(0.0, 0.0),
        rf_bounds=(5.0, 5.6),
    )

    assert g == 0.0
    assert 5.0 <= bare_rf <= 5.6


def test_auto_tune_grid_scan_escapes_a_decoy():
    sp_fluxs, sp_freqs = _grid()
    true_band = np.exp(-((sp_freqs - 5.5) ** 2) / (2 * 0.05**2))
    decoy_band = 0.6 * np.exp(-((sp_freqs - 5.15) ** 2) / (2 * 0.03**2))
    norm = np.tile(true_band + decoy_band, (len(sp_fluxs), 1))

    _, bare_rf = auto_tune(
        (4.0, 1.0, 0.5),
        sp_fluxs,
        sp_freqs,
        norm,
        np.array([0.1, 0.3]),
        g0=0.0,
        bare_rf0=5.16,
        g_bounds=(0.0, 0.0),
        rf_bounds=(5.0, 6.0),
    )

    assert abs(bare_rf - 5.5) < 0.05


def test_auto_tune_keeps_a_good_current_seed():
    sp_fluxs, sp_freqs = _grid()
    norm = np.tile(np.exp(-((sp_freqs - 5.5) ** 2) / (2 * 0.08**2)), (len(sp_fluxs), 1))

    _, bare_rf = auto_tune(
        (4.0, 1.0, 0.5),
        sp_fluxs,
        sp_freqs,
        norm,
        np.array([0.2]),
        g0=0.0,
        bare_rf0=5.5,
        g_bounds=(0.0, 0.0),
        rf_bounds=(5.0, 6.0),
    )

    assert abs(bare_rf - 5.5) < 0.02


def test_auto_tune_fast_fails_without_samples():
    sp_fluxs, sp_freqs = _grid()
    norm = np.zeros((len(sp_fluxs), len(sp_freqs)))
    with pytest.raises(ValueError, match="sample"):
        auto_tune(
            (4.0, 1.0, 0.5),
            sp_fluxs,
            sp_freqs,
            norm,
            np.array([]),
            0.06,
            5.3,
            (0.0, 0.2),
            (5.0, 6.0),
        )
