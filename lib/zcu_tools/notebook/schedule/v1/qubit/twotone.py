from copy import deepcopy

from zcu_tools.notebook.single_qubit.process import (
    calculate_noise,
    peak_n_avg,
    rotate2real,
)
from zcu_tools.program.v1 import RFreqTwoToneProgram

from ...tools import sweep2array
from ..template import sweep1D_hard_template


def signal2real(signals):
    return rotate2real(signals).real


def qub_signal2snr(signals):
    noise, m_signals = calculate_noise(signals)

    m_amps = signal2real(m_signals)
    contrast = peak_n_avg(m_amps, n=3, mode="max")

    return contrast / noise


def measure_qub_freq(soc, soccfg, cfg, remove_bg=False, earlystop_snr=None):
    cfg = deepcopy(cfg)  # prevent in-place modification

    fpts = sweep2array(cfg["sweep"])

    kwargs = {"xlabel": "Frequency (MHz)", "ylabel": "Amplitude"}
    if remove_bg:
        kwargs["signal2real"] = signal2real

    if earlystop_snr is not None:

        def checker(signals):
            snr = qub_signal2snr(signals)
            return snr >= earlystop_snr, f"Current SNR: {snr:.2g}"

        kwargs["early_stop_checker"] = checker

    fpts, signals = sweep1D_hard_template(
        soc,
        soccfg,
        cfg,
        RFreqTwoToneProgram,
        xs=fpts,
        **kwargs,
    )

    return fpts, signals
