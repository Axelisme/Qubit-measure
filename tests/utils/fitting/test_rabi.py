import numpy as np

from zcu_tools.utils.fitting.base import cosfunc, decaycos
from zcu_tools.utils.fitting.rabi import fit_rabi


def test_fit_rabi_no_decay():
    xs = np.linspace(0, 4, 400)
    y0, yscale, freq, phase = 0.0, 1.0, 0.5, 0.0
    ys = cosfunc(xs, y0, yscale, freq, phase)
    pi_x, pi2_x, fit_freq, _, _ = fit_rabi(xs, ys, decay=False)
    assert abs(fit_freq - freq) < 1e-3
    assert abs(pi_x - 1.0) < 1e-2
    assert abs(pi2_x - 0.5) < 1e-2


def test_fit_rabi_with_decay():
    xs = np.linspace(0, 6, 400)
    y0, yscale, freq, phase, tau = 0.0, 1.0, 0.4, 0.0, 10.0
    ys = decaycos(xs, y0, yscale, freq, phase, tau)
    pi_x, pi2_x, fit_freq, _, _ = fit_rabi(xs, ys, decay=True)
    assert abs(fit_freq - freq) / freq < 5e-2
    assert abs(pi_x - 1.25) < 5e-2
