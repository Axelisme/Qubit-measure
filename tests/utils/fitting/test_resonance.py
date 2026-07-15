import numpy as np
from zcu_tools.utils.fitting.resonance import (
    HangerModel,
    TransmissionModel,
    fit_circle_params,
    get_proper_model,
)


def test_transmission_fit_recovers_freq_and_Ql():
    f0, Ql = 5000.0, 500.0
    a0 = 1.0 + 0j
    edelay = 0.0
    freqs = np.linspace(f0 - 50, f0 + 50, 801)
    signals = TransmissionModel.calc_signals(freqs, f0, Ql, a0, edelay)
    params = TransmissionModel.fit(freqs, signals, edelay=edelay)
    assert abs(params["freq"] - f0) / f0 < 1e-3
    assert abs(params["Ql"] - Ql) / Ql < 1e-1


def test_hanger_fit_recovers_freq_and_Ql():
    f0, Ql, Qc = 5000.0, 4000.0, 6000.0
    phi = 0.0
    a0 = 1.0 + 0j
    edelay = 0.0
    freqs = np.linspace(f0 - 20, f0 + 20, 801)
    signals = HangerModel.calc_signals(freqs, f0, Ql, Qc, phi, a0, edelay)
    params = HangerModel.fit(freqs, signals, edelay=edelay)
    assert abs(params["freq"] - f0) / f0 < 1e-3
    assert abs(params["Ql"] - Ql) / Ql < 2e-1


def test_circle_fit_recovers_nonuniform_exact_circle():
    center = 2.3 - 0.7j
    radius = 1.9
    angles = np.linspace(-2.8, 2.4, 301) ** 3 / 8.0
    signals = center + radius * np.exp(1j * angles)

    xc, yc, fitted_radius = fit_circle_params(signals.real, signals.imag)

    np.testing.assert_allclose(xc, center.real, atol=1e-9)
    np.testing.assert_allclose(yc, center.imag, atol=1e-9)
    np.testing.assert_allclose(fitted_radius, radius, atol=1e-9)


def test_hanger_fit_recovers_nonuniform_grid_with_edelay():
    f0, Ql, Qc = 6050.0, 4000.0, 6000.0
    phi = 0.1
    a0 = 35.0 * np.exp(0.2j)
    edelay = -1.117
    theta0 = -0.32

    def theta(freq: float) -> float:
        return theta0 + 2.0 * np.arctan(2.0 * Ql * (1.0 - freq / f0))

    thetas = np.linspace(theta(6000.0), theta(6100.0), 301)
    freqs = f0 * (1.0 - np.tan((thetas - theta0) / 2.0) / (2.0 * Ql))
    signals = HangerModel.calc_signals(freqs, f0, Ql, Qc, phi, a0, edelay)

    params = HangerModel.fit(freqs, signals)

    np.testing.assert_allclose(params["freq"], f0, atol=0.1)
    np.testing.assert_allclose(params["Ql"], Ql, rtol=0.1)
    np.testing.assert_allclose(params["edelay"], edelay, atol=1e-3)


def test_get_proper_model_hanger_vs_transmission():
    freqs = np.linspace(4980, 5020, 801)
    hanger = HangerModel.calc_signals(freqs, 5000.0, 4000.0, 6000.0, 0.0, 1.0 + 0j, 0.0)
    tm = TransmissionModel.calc_signals(freqs, 5000.0, 4000.0, 1.0 + 0j, 0.0)
    assert isinstance(get_proper_model(freqs, hanger), HangerModel)
    assert isinstance(get_proper_model(freqs, tm), TransmissionModel)
