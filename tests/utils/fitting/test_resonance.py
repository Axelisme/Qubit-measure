import numpy as np
from zcu_tools.utils.fitting.resonance import (
    HangerModel,
    TransmissionModel,
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


def test_get_proper_model_hanger_vs_transmission():
    freqs = np.linspace(4980, 5020, 801)
    hanger = HangerModel.calc_signals(freqs, 5000.0, 4000.0, 6000.0, 0.0, 1.0 + 0j, 0.0)
    tm = TransmissionModel.calc_signals(freqs, 5000.0, 4000.0, 1.0 + 0j, 0.0)
    assert isinstance(get_proper_model(freqs, hanger), HangerModel)
    assert isinstance(get_proper_model(freqs, tm), TransmissionModel)
