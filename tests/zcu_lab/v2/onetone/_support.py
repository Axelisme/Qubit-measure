"""Typed inputs shared by OneTone public contract tests."""

from __future__ import annotations

import numpy as np
from zcu_tools.analysis.fitting import HangerModel, TransmissionModel
from zcu_tools.program.v2 import PulseReadoutCfg

from zcu_lab.v2.onetone.freq.core import FreqCfg, FreqResult, HomophasalSamplingCfg
from zcu_lab.v2.onetone.power_dep.core import PowerDepCfg
from zcu_lab.v2.onetone.sa.core import SA_FreqCfg


def make_readout() -> PulseReadoutCfg:
    return PulseReadoutCfg.model_validate(
        {
            "pulse_cfg": {
                "ch": 0,
                "nqz": 1,
                "gain": 0.2,
                "freq": 6000.0,
                "waveform": {"style": "const", "length": 1.0},
            },
            "ro_cfg": {
                "type": "readout/direct",
                "ro_ch": 0,
                "gen_ch": 0,
                "ro_length": 0.4,
                "ro_freq": 6000.0,
                "trig_offset": 0.1,
            },
        }
    )


def make_freq_cfg(*, homophasal: bool = False) -> FreqCfg:
    return FreqCfg.model_validate(
        {
            "reps": 2,
            "rounds": 1,
            "modules": {"readout": make_readout()},
            "sweep": {
                "freq": {"start": 5970.0, "stop": 6030.0, "step": 7.5, "expts": 9}
            },
            "sampling_mode": "homophasal" if homophasal else "linear",
            "homophasal": (
                HomophasalSamplingCfg(r_f=6000.0, rf_w=6000.0 / 700.0, theta0=0.12)
                if homophasal
                else None
            ),
        }
    )


def make_power_cfg(*, earlystop_snr: float | None = None) -> PowerDepCfg:
    return PowerDepCfg.model_validate(
        {
            "reps": 2,
            "rounds": 1,
            "earlystop_snr": earlystop_snr,
            "modules": {"readout": make_readout()},
            "sweep": {
                "freq": {"start": 5970.0, "stop": 6030.0, "step": 7.5, "expts": 9},
                "gain": {"start": 0.1, "stop": 0.3, "step": 0.1, "expts": 3},
            },
        }
    )


def make_sa_cfg() -> SA_FreqCfg:
    return SA_FreqCfg.model_validate(
        {
            "reps": 2,
            "rounds": 1,
            "modules": {"readout": make_readout()},
            "sweep": {
                "freq": {"start": 5970.0, "stop": 6030.0, "step": 7.5, "expts": 9}
            },
        }
    )


def make_freq_result(
    *, freq: float = 6000.0, hanger: bool = True, background: bool = False
) -> FreqResult:
    freqs = np.linspace(freq - 35.0, freq + 35.0, 301)
    common = dict(
        freq=freq,
        Ql=700.0,
        a0=1.2 * np.exp(0.3j),
        edelay=0.021,
        bg_amp_slope=0.008 if background else 0.0,
        bg_phase_curvature=7e-4 if background else 0.0,
    )
    if hanger:
        signals = HangerModel.calc_signals(freqs, Qc=980.0, phi=0.12, **common)
    else:
        signals = TransmissionModel.calc_signals(freqs, **common)
    return FreqResult(freqs, signals)
