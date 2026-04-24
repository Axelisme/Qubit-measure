import numpy as np
from zcu_tools.utils.fitting.base import lorfunc, sincfunc
from zcu_tools.utils.fitting.qubfreq import fit_qubit_freq


def test_fit_qubit_freq_lor():
    freqs = np.linspace(-10, 10, 400)
    true = (0.0, 0.0, 1.0, 1.5, 0.3)
    signals = lorfunc(freqs, *true)
    freq, _, kappa, _, _, _ = fit_qubit_freq(freqs, signals, type="lor")
    assert abs(freq - 1.5) < 1e-3
    assert abs(kappa - 2 * 0.3) < 1e-2


def test_fit_qubit_freq_sinc():
    freqs = np.linspace(-10, 10, 400)
    true = (0.0, 0.0, 1.0, -0.5, 0.8)
    signals = sincfunc(freqs, *true)
    freq, _, kappa, _, _, _ = fit_qubit_freq(freqs, signals, type="sinc")
    assert abs(freq - (-0.5)) < 1e-2
    assert abs(kappa - 1.2067 * 0.8) / (1.2067 * 0.8) < 5e-2
