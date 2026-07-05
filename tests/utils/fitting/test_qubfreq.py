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


def test_fit_qubit_freq_lor_left_edge_peak_with_tilted_background():
    freqs = np.linspace(-20, 50, 141)
    true = (0.2, 0.005, 1.0, -19.0, 3.0)
    signals = lorfunc(freqs, *true)
    signals = (signals - signals.min()) / np.ptp(signals)

    freq, _, _, _, fit_signals, _ = fit_qubit_freq(freqs, signals, type="lor")

    residual = np.mean(np.abs(signals - fit_signals))
    assert abs(freq - true[3]) < 1.0
    assert residual < 0.2 * np.ptp(fit_signals)


def test_fit_qubit_freq_lor_left_edge_dip():
    freqs = np.linspace(-20, 50, 141)
    true = (0.2, 0.0, -1.0, -20.0, 3.0)
    signals = lorfunc(freqs, *true)
    signals = (signals - signals.min()) / np.ptp(signals)

    freq, _, _, _, fit_signals, _ = fit_qubit_freq(freqs, signals, type="lor")

    residual = np.mean(np.abs(signals - fit_signals))
    assert abs(freq - true[3]) < 1.0
    assert residual < 0.2 * np.ptp(fit_signals)


def test_fit_qubit_freq_sinc():
    freqs = np.linspace(-10, 10, 400)
    true = (0.0, 0.0, 1.0, -0.5, 0.8)
    signals = sincfunc(freqs, *true)
    freq, _, kappa, _, _, _ = fit_qubit_freq(freqs, signals, type="sinc")
    assert abs(freq - (-0.5)) < 1e-2
    assert abs(kappa - 1.2067 * 0.8) / (1.2067 * 0.8) < 5e-2
