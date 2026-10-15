"""Single-point prediction contracts at arbitrary flux coordinates."""

from __future__ import annotations

import numpy as np
from zcu_tools.analysis.dispersive.predict import predict_dispersive_at
from zcu_tools.simulate.fluxonium import calculate_dispersive_vs_flux_fast


def test_predict_dispersive_at_arbitrary_fluxs():
    fluxs = np.array([0.47, 0.12, 0.31, 0.12])
    lines = predict_dispersive_at((4.0, 1.0, 0.5), fluxs, 0.06, 5.3)
    expected = calculate_dispersive_vs_flux_fast(
        (4.0, 1.0, 0.5), fluxs, 5.3, 0.06, qub_cutoff=30, qub_dim=15, res_dim=4
    )
    assert len(lines) == 2
    np.testing.assert_allclose(lines, expected)
