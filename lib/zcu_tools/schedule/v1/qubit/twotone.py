from copy import deepcopy

import numpy as np
from zcu_tools.analysis import minus_background, calculate_noise
from zcu_tools.program.v1 import RFreqTwoToneProgram, RFreqTwoToneProgramWithRedReset
from zcu_tools.schedule.tools import sweep2array
from zcu_tools.schedule.v1.template import sweep1D_hard_template


def signal2real(signals):
    return np.abs(minus_background(signals))


def qub_signal2snr(signals):
    noise, m_signals = calculate_noise(signals)

    amps = np.abs(m_signals)

    # use avg of highest three point as signal contrast
    max1_idx = np.argmax(amps)
    max1, amps[max1_idx] = amps[max1_idx], 0
    max2_idx = np.argmax(amps)
    max2, amps[max2_idx] = amps[max2_idx], 0
    max3_idx = np.argmax(amps)
    max3 = amps[max3_idx]

    contrast = (max1 + max2 + max3) / 3

    return contrast / noise


def measure_qub_freq(
    soc, soccfg, cfg, reset_rf=None, remove_bg=False, earlystop_snr=None
):
    cfg = deepcopy(cfg)  # prevent in-place modification

    if reset_rf is not None:
        assert cfg["dac"]["reset"] == "pulse", "Need reset=pulse for conjugate reset"
        assert "reset_pulse" in cfg["dac"], "Need reset_pulse for conjugate reset"
        cfg["r_f"] = reset_rf

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
        RFreqTwoToneProgram if reset_rf is None else RFreqTwoToneProgramWithRedReset,
        xs=fpts,
        **kwargs,
    )

    return fpts, signals
