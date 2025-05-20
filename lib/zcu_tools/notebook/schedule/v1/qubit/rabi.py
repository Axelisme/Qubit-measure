from copy import deepcopy
from typing import Literal

import numpy as np
from zcu_tools.notebook.single_qubit.process import (
    calculate_noise,
    minus_background,
    peak_n_avg,
    rotate2real,
)
from zcu_tools.program.v1 import RGainTwoToneProgram, TwoToneProgram

from ...tools import format_sweep1D, sweep2array
from ..template import sweep1D_hard_template, sweep1D_soft_template


def signal2real(signals: np.ndarray) -> np.ndarray:
    return np.abs(minus_background(signals))


def qub_signal2snr(signals):
    noise, m_signals = calculate_noise(signals)

    m_real = rotate2real(m_signals).real
    contrast = peak_n_avg(m_real, n=3, mode="max") - peak_n_avg(m_real, n=3, mode="min")

    return contrast / noise


def measure_lenrabi(
    soc,
    soccfg,
    cfg,
    *,
    force_align=False,
    align_type: Literal["pre_delay", "post_delay"] = "post_delay",
    earlystop_snr=None,
):
    cfg = deepcopy(cfg)

    if earlystop_snr is not None:
        raise NotImplementedError("earlystop_snr is not implemented for v1 yet")
    if force_align:
        align_type = align_type  # make ruff happy
        raise NotImplementedError("force_align is not implemented for v1 yet")

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")
    len_sweep = cfg["sweep"]["length"]

    lens = sweep2array(len_sweep, allow_array=True)

    del cfg["sweep"]  # remove sweep for program use

    if force_align:
        max_length = max(lens.max(), cfg["dac"]["qub_pulse"].get(align_type, 0.0))

    def updateCfg(cfg, _, length):
        cfg["dac"]["qub_pulse"]["length"] = length
        cfg["dac"]["qub_pulse"][align_type] = max_length - length

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
