from __future__ import annotations

import numpy as np
import pytest
from zcu_tools.experiment.v2.onetone.freq import (
    HomophasalSamplingCfg,
    homophasal_freqs_from_sweep,
)
from zcu_tools.program.v2 import SweepCfg


def _theta_from_freq(freqs: np.ndarray, params: HomophasalSamplingCfg) -> np.ndarray:
    return params.theta0 + 2.0 * np.arctan(
        2.0 * params.q_l * (1.0 - freqs / params.r_f)
    )


@pytest.mark.parametrize(
    "sweep",
    [
        SweepCfg(start=5960.0, stop=6040.0, expts=9, step=10.0),
        SweepCfg(start=6040.0, stop=5960.0, expts=9, step=-10.0),
    ],
)
def test_homophasal_freqs_preserve_endpoints_and_equal_theta_spacing(
    sweep: SweepCfg,
) -> None:
    params = HomophasalSamplingCfg(r_f=6000.0, rf_w=20.0, theta0=0.35)

    freqs = homophasal_freqs_from_sweep(sweep, params)

    assert freqs[0] == pytest.approx(sweep.start)
    assert freqs[-1] == pytest.approx(sweep.stop)
    thetas = _theta_from_freq(freqs, params)
    theta_steps = np.diff(thetas)
    np.testing.assert_allclose(theta_steps, np.full_like(theta_steps, theta_steps[0]))


def test_homophasal_sampling_params_require_positive_fit_scale() -> None:
    with pytest.raises(ValueError, match="r_f must be positive"):
        HomophasalSamplingCfg(r_f=0.0, rf_w=20.0, theta0=0.0)

    with pytest.raises(ValueError, match="rf_w must be positive"):
        HomophasalSamplingCfg(r_f=6000.0, rf_w=0.0, theta0=0.0)
