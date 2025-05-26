from copy import deepcopy
from typing import Tuple

import numpy as np
from zcu_tools.program.v2 import OneToneProgram
from zcu_tools.notebook.single_qubit.process import rotate2real

from ...tools import format_sweep1D, sweep2array, sweep2param
from ..template import sweep_hard_template

def qub_signal2real(signals: np.ndarray) -> np.ndarray:
    return np.abs(rotate2real(signals - np.mean(signals)).real)

def measure_reset_freq(
    soc, soccfg, cfg, ro_freq: float
) -> Tuple[np.ndarray, np.ndarray]:
    cfg = deepcopy(cfg)  # prevent in-place modification
    reset_pulse = cfg["dac"]["reset_pulse"]

    if cfg["dac"]["reset"] != "pulse":
        raise ValueError("Reset pulse must be one pulse")
    if cfg["dac"]["readout"] != "passive":
        raise ValueError("Readout must be passive")
    cfg["adc"]["ro_freq"] = ro_freq
    cfg["adc"]["gen_ch"] = reset_pulse["ch"] # TODO: I don't known why need this

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "freq")

    sweep_cfg = cfg["sweep"]["freq"]
    reset_pulse["freq"] = sweep2param("freq", sweep_cfg)

    fpts = sweep2array(sweep_cfg)  # predicted frequency points

    prog, signals = sweep_hard_template(
        soc,
        soccfg,
        cfg,
        OneToneProgram,
        ticks=(fpts,),
        progress=True,
        xlabel="Frequency (MHz)",
        ylabel="Amplitude",
        signal2real=qub_signal2real
    )

    # get the actual frequency points
    fpts = prog.get_pulse_param("reset", "freq", as_array=True)

    return fpts, signals


def measure_mux_reset_freq(
    soc, soccfg, cfg
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cfg = deepcopy(cfg)  # prevent in-place modification

    reset_pulse1 = cfg["dac"]["reset_pulse1"]
    reset_pulse2 = cfg["dac"]["reset_pulse2"]

    if cfg["dac"]["reset"] != "two_pulse":
        raise ValueError("Reset pulse must be two pulse")

    # force freq1 to be the outer loop
    cfg["sweep"] = {"freq1": cfg["sweep"]["freq1"], "freq2": cfg["sweep"]["freq2"]}

    reset_pulse1["freq"] = sweep2param("freq1", cfg["sweep"]["freq1"])
    reset_pulse2["freq"] = sweep2param("freq2", cfg["sweep"]["freq2"])

    fpts1 = sweep2array(cfg["sweep"]["freq1"])  # predicted frequency points
    fpts2 = sweep2array(cfg["sweep"]["freq2"])  # predicted frequency points

    prog, signals = sweep_hard_template(
        soc,
        soccfg,
        cfg,
        OneToneProgram,
        ticks=(fpts1, fpts2),
        progress=True,
        xlabel="Frequency1 (MHz)",
        ylabel="Frequency2 (MHz)",
        signal2real=qub_signal2real
    )

    # get the actual frequency points
    fpts1 = prog.get_pulse_param("reset1", "freq", as_array=True)
    fpts2 = prog.get_pulse_param("reset2", "freq", as_array=True)

    return fpts1, fpts2, signals
