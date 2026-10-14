from __future__ import annotations

import numpy as np
from zcu_tools.notebook.analysis.fit_tools import predict_f01_mhz
from zcu_tools.notebook.analysis.t1_curve import correct_flux_from_f01


def test_correct_flux_from_f01_facade_corrects_reachable_frequency() -> None:
    params = (3.4, 0.9, 0.6)
    target_fluxs = np.array([0.31], dtype=np.float64)
    freqs_ghz = 1e-3 * predict_f01_mhz(params, target_fluxs)

    result = correct_flux_from_f01(np.array([0.3], dtype=np.float64), freqs_ghz, params)

    np.testing.assert_allclose(result.corrected_fluxs, target_fluxs, atol=1e-5)
    np.testing.assert_array_equal(result.accepted, [True])
