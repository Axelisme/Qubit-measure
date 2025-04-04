from copy import deepcopy

import numpy as np

from zcu_tools.analysis import (
    minus_background,
    calculate_noise,
    rotate2real,
    peak_n_avg,
)
from zcu_tools.program.v1 import RGainTwoToneProgram, TwoToneProgram
from zcu_tools.schedule.tools import sweep2array
from zcu_tools.schedule.v1.template import sweep1D_hard_template, sweep1D_soft_template


def signal2real(signals: np.ndarray) -> np.ndarray:
    return np.abs(minus_background(signals))


def qub_signal2snr(signals):
    noise, m_signals = calculate_noise(signals)

    m_real = rotate2real(m_signals).real
    contrast = peak_n_avg(m_real, n=3, mode="max") - peak_n_avg(m_real, n=3, mode="min")

    return contrast / noise


def measure_lenrabi(soc, soccfg, cfg, *, force_align=False, earlystop_snr=None):
    cfg = deepcopy(cfg)

    if force_align:
        raise NotImplementedError("force_align is not implemented for v1 yet")
    if earlystop_snr is not None:
        raise NotImplementedError("earlystop_snr is not implemented for v1 yet")

    lens = sweep2array(cfg["sweep"], allow_array=True)

    del cfg["sweep"]  # remove sweep for program use

    def updateCfg(cfg, _, length):
        cfg["dac"]["qub_pulse"]["length"] = length

    lens, signals = sweep1D_soft_template(
        soc,
        soccfg,
        cfg,
        TwoToneProgram,
        xs=lens,
        updateCfg=updateCfg,
        xlabel="Length (us)",
        ylabel="Amplitude",
        signal2real=signal2real,
    )

    return lens, signals


def measure_amprabi(soc, soccfg, cfg, *, earlystop_snr=None):
    cfg = deepcopy(cfg)

    pdrs = sweep2array(cfg["sweep"])

    if earlystop_snr is not None:

        def checker(signals):
            snr = qub_signal2snr(signals)
            return snr >= earlystop_snr, f"SNR: {snr:.2f}"

    else:
        checker = None

    pdrs, signals = sweep1D_hard_template(
        soc,
        soccfg,
        cfg,
        RGainTwoToneProgram,
        xs=pdrs,
        xlabel="Pulse Power (a.u.)",
        ylabel="Amplitude",
        signal2real=signal2real,
        early_stop_checker=checker,
    )

    return pdrs, signals
