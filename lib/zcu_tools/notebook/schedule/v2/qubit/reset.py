from copy import deepcopy
from typing import Tuple

import numpy as np
from zcu_tools.program.v2 import OneToneProgram

from ...tools import format_sweep1D, sweep2array, sweep2param
from ..template import sweep_hard_template


def measure_reset_freq(
    soc, soccfg, cfg, ro_freq: float
) -> Tuple[np.ndarray, np.ndarray]:
    cfg = deepcopy(cfg)  # prevent in-place modification

    if cfg["dac"]["reset"] != "pulse":
        raise ValueError("Reset pulse must be one pulse")
    if cfg["dac"]["readout"] != "passive":
        raise ValueError("Readout must be passive")
    cfg["adc"]["ro_freq"] = ro_freq

    reset_pulse = cfg["dac"]["reset_pulse"]

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
    )

    # get the actual frequency points
    fpts = prog.get_pulse_param("reset_pulse", "freq", as_array=True)

    return fpts, signals


def measure_mux_reset_freq(
    soc, soccfg, cfg, ro_freq: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cfg = deepcopy(cfg)  # prevent in-place modification

    if cfg["dac"]["reset"] != "two_pulse":
        raise ValueError("Reset pulse must be two pulse")
    if cfg["dac"]["readout"] != "passive":
        raise ValueError("Readout must be passive")
    cfg["adc"]["ro_freq"] = ro_freq

    reset_pulse1 = cfg["dac"]["reset_pulse1"]
    reset_pulse2 = cfg["dac"]["reset_pulse2"]

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
    )

    # get the actual frequency points
    fpts1 = prog.get_pulse_param("reset_pulse1", "freq", as_array=True)
    fpts2 = prog.get_pulse_param("reset_pulse2", "freq", as_array=True)

    return fpts1, fpts2, signals
