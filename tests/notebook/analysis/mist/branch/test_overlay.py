from __future__ import annotations

import numpy as np
from zcu_tools.notebook.analysis.mist.branch.overlay import calc_overlay


def test_calc_overlay_retains_per_photon_state_overlaps() -> None:
    overlays = calc_overlay(
        params=(5.0, 1.0, 0.5),
        photons=np.asarray([0.0, 0.01], dtype=np.float64),
        r_f=6.5,
        g=0.01,
        flux=0.5,
        qub_dim=3,
        qub_cutoff=8,
    )

    assert overlays.shape == (2, 2)
    assert np.all(np.isfinite(overlays))
    assert np.all(overlays >= -1e-8)
    assert np.all(overlays <= 1.0 + 1e-8)
    np.testing.assert_allclose(overlays[0], [1.0, 1.0], rtol=0.0, atol=1e-8)
