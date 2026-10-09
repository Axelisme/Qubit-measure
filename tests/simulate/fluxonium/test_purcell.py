from __future__ import annotations

import numpy as np
import zcu_tools.simulate.fluxonium as fluxonium
import zcu_tools.simulate.fluxonium.coherence as coherence


def test_purcell_export_uses_correct_spelling() -> None:
    assert (
        fluxonium.calculate_purcell_t1_vs_flux is coherence.calculate_purcell_t1_vs_flux
    )
    assert "calculate_purcell_t1_vs_flux" in fluxonium.__all__
    assert "calculate_percell_t1_vs_flux" not in fluxonium.__all__
    assert not hasattr(fluxonium, "calculate_percell_t1_vs_flux")


def test_purcell_t1_scales_inversely_with_resonator_linewidth() -> None:
    fluxs = np.array([0.2, 0.35], dtype=np.float64)
    first = fluxonium.calculate_purcell_t1_vs_flux(
        fluxs,
        bare_rf=5.0,
        kappa=0.001,
        g=0.05,
        Temp=0.04,
        params=(4.0, 1.0, 0.5),
        progress=False,
    )
    second = fluxonium.calculate_purcell_t1_vs_flux(
        fluxs,
        bare_rf=5.0,
        kappa=0.002,
        g=0.05,
        Temp=0.04,
        params=(4.0, 1.0, 0.5),
        progress=False,
    )

    assert first.shape == second.shape == (2,)
    assert first.dtype == second.dtype == np.float64
    for result in (first, second):
        assert np.all(np.isfinite(result))
        assert np.all(result > 0)
    np.testing.assert_allclose(
        2 * np.asarray(second), np.asarray(first), rtol=1e-12, atol=0.0
    )
