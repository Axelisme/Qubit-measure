"""Tests for the dispersive auto-tune service.

The objective is mean over sample fluxes of max(norm_phase@ground, norm_phase@excited),
read by bilinear interpolation; auto_tune maximises it over (g, bare_rf). The dispersive
prediction is stubbed so the test is fast and the optimum is known.
"""

from __future__ import annotations

import numpy as np
import zcu_tools.gui.app.dispersive.services.autotune as autotune_mod
from zcu_tools.gui.app.dispersive.services.autotune import (
    _interp_norm_phase,
    auto_tune,
    sample_score,
)


def _grid():
    sp_fluxs = np.linspace(0.0, 0.5, 26).astype(np.float64)
    sp_freqs = np.linspace(5.0, 6.0, 40).astype(np.float64)
    return sp_fluxs, sp_freqs


def test_interp_norm_phase_bilinear_and_clamped():
    sp_fluxs, sp_freqs = _grid()
    # a ramp in freq so interpolation is exactly predictable
    norm = np.tile(sp_freqs[None, :], (len(sp_fluxs), 1))  # value == freq
    # midway between two freq grid points -> mean of the two (bilinear)
    f = 0.5 * (sp_freqs[10] + sp_freqs[11])
    v = _interp_norm_phase(sp_fluxs, sp_freqs, norm, np.array([0.2]), np.array([f]))
    np.testing.assert_allclose(v[0], f, atol=1e-9)
    # out-of-range freq clamps/extrapolates, never NaN
    v_hi = _interp_norm_phase(
        sp_fluxs, sp_freqs, norm, np.array([0.2]), np.array([99.0])
    )
    assert np.isfinite(v_hi[0])


def test_sample_score_picks_max_of_ground_excited(monkeypatch):
    sp_fluxs, sp_freqs = _grid()
    # norm_phase = 1 on a band around 5.5 GHz, else 0
    norm = np.zeros((len(sp_fluxs), len(sp_freqs)))
    band = np.abs(sp_freqs - 5.5) < 0.05
    norm[:, band] = 1.0

    # stub: ground far off (0 phase), excited on the band (phase 1) -> max picks 1
    def stub(params, fluxs, g, bare_rf, *, return_dim=2):
        return (np.full(len(fluxs), 5.0), np.full(len(fluxs), 5.5))

    monkeypatch.setattr(autotune_mod, "predict_dispersive_at", stub)
    s = sample_score(
        (4.0, 1.0, 0.5), sp_fluxs, sp_freqs, norm, np.array([0.1, 0.3]), 0.06, 5.3
    )
    assert s == np.float64(1.0)


def test_auto_tune_finds_the_phase_peak(monkeypatch):
    sp_fluxs, sp_freqs = _grid()
    # the phase image has a broad peak at 5.5 GHz (a realistic, smooth feature whose
    # basin reaches the seed); the stubbed excited frequency equals bare_rf, so the
    # optimal bare_rf is 5.5. A local optimiser needs a seed in the basin — that is
    # the real contract (the user drags roughly close first).
    norm = np.exp(-((sp_freqs[None, :] - 5.5) ** 2) / (2 * 0.2**2)) + np.zeros(
        (len(sp_fluxs), 1)
    )

    def stub(params, fluxs, g, bare_rf, *, return_dim=2):
        # ground = bare_rf - g (off the peak), excited = bare_rf (drives the score)
        return (np.full(len(fluxs), bare_rf - g), np.full(len(fluxs), bare_rf))

    monkeypatch.setattr(autotune_mod, "predict_dispersive_at", stub)
    g, bare_rf = auto_tune(
        (4.0, 1.0, 0.5),
        sp_fluxs,
        sp_freqs,
        norm,
        np.array([0.1, 0.25, 0.4]),
        g0=0.06,
        bare_rf0=5.35,  # seed within the broad peak's basin
        g_bounds=(0.0, 0.2),
        rf_bounds=(5.0, 6.0),
    )
    assert 0.0 <= g <= 0.2
    assert abs(bare_rf - 5.5) < 0.05  # converged toward the phase peak


def test_auto_tune_respects_bounds(monkeypatch):
    sp_fluxs, sp_freqs = _grid()
    # phase peaks at 5.9 but the bound caps r_f at 5.6 -> result clamped <= 5.6
    norm = np.exp(-((sp_freqs[None, :] - 5.9) ** 2) / (2 * 0.03**2)) + np.zeros(
        (len(sp_fluxs), 1)
    )

    def stub(params, fluxs, g, bare_rf, *, return_dim=2):
        return (np.full(len(fluxs), bare_rf), np.full(len(fluxs), bare_rf))

    monkeypatch.setattr(autotune_mod, "predict_dispersive_at", stub)
    g, bare_rf = auto_tune(
        (4.0, 1.0, 0.5),
        sp_fluxs,
        sp_freqs,
        norm,
        np.array([0.2]),
        g0=0.06,
        bare_rf0=5.3,
        g_bounds=(0.0, 0.2),
        rf_bounds=(5.0, 5.6),
    )
    assert bare_rf <= 5.6 + 1e-9


def test_auto_tune_grid_scan_escapes_a_decoy(monkeypatch):
    # The coarse grid scan must escape a spurious (decoy) phase band that a purely
    # local search, seeded right next to it, would stick in. The TRUE peak is at
    # 5.5 GHz; a brighter-but-wrong decoy sits at 5.15, and the seed starts on it.
    sp_fluxs, sp_freqs = _grid()
    true_band = np.exp(-((sp_freqs - 5.5) ** 2) / (2 * 0.05**2))
    decoy_band = 0.6 * np.exp(-((sp_freqs - 5.15) ** 2) / (2 * 0.03**2))
    norm = (true_band + decoy_band)[None, :] + np.zeros((len(sp_fluxs), 1))

    def stub(params, fluxs, g, bare_rf, *, return_dim=2):
        # both lines sit at bare_rf so the score is driven purely by where r_f lands
        return (np.full(len(fluxs), bare_rf), np.full(len(fluxs), bare_rf))

    monkeypatch.setattr(autotune_mod, "predict_dispersive_at", stub)
    g, bare_rf = auto_tune(
        (4.0, 1.0, 0.5),
        sp_fluxs,
        sp_freqs,
        norm,
        np.array([0.1, 0.3]),
        g0=0.06,
        bare_rf0=5.16,  # seed sitting on the decoy
        g_bounds=(0.0, 0.2),
        rf_bounds=(5.0, 6.0),
    )
    # the grid scan finds the (taller) true peak at 5.5, not the decoy at 5.15
    assert abs(bare_rf - 5.5) < 0.05


def test_auto_tune_keeps_a_good_current_seed(monkeypatch):
    # If the current slider is already at the optimum and no grid point beats it,
    # the result stays there (the current g0/bare_rf0 is always a candidate seed).
    sp_fluxs, sp_freqs = _grid()
    norm = np.exp(-((sp_freqs[None, :] - 5.5) ** 2) / (2 * 0.08**2)) + np.zeros(
        (len(sp_fluxs), 1)
    )

    def stub(params, fluxs, g, bare_rf, *, return_dim=2):
        return (np.full(len(fluxs), bare_rf), np.full(len(fluxs), bare_rf))

    monkeypatch.setattr(autotune_mod, "predict_dispersive_at", stub)
    g, bare_rf = auto_tune(
        (4.0, 1.0, 0.5),
        sp_fluxs,
        sp_freqs,
        norm,
        np.array([0.2]),
        g0=0.06,
        bare_rf0=5.5,  # already at the peak
        g_bounds=(0.0, 0.2),
        rf_bounds=(5.0, 6.0),
    )
    assert abs(bare_rf - 5.5) < 0.02


def test_auto_tune_fast_fails_without_samples():
    sp_fluxs, sp_freqs = _grid()
    norm = np.zeros((len(sp_fluxs), len(sp_freqs)))
    try:
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
    except ValueError as exc:
        assert "sample" in str(exc)
    else:
        raise AssertionError("expected ValueError for empty sample fluxes")
