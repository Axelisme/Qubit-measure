from copy import deepcopy
from typing import Tuple

import numpy as np
from zcu_tools.liveplot.jupyter import LivePlotter1D
from zcu_tools.notebook.single_qubit.process import rotate2real
from zcu_tools.program.v2 import TwoToneProgram
from zcu_tools.program.v2.base.simulate import SimulateProgramV2

from ...tools import format_sweep1D, sweep2array, sweep2param
from ..template import sweep_hard_template


def qub_signal2real(signals: np.ndarray) -> np.ndarray:
    return rotate2real(signals).real


def measure_qub_freq(soc, soccfg, cfg) -> Tuple[np.ndarray, np.ndarray]:
    cfg = deepcopy(cfg)  # prevent in-place modification

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "freq")

    sweep_cfg = cfg["sweep"]["freq"]
    params = sweep2param("freq", sweep_cfg)
    cfg["qub_pulse"]["freq"] = params

    fpts = sweep2array(sweep_cfg)  # predicted frequency points

    prog = TwoToneProgram(soccfg, cfg)

    signals = sweep_hard_template(
        cfg,
        lambda _, cb: prog.acquire(soc, progress=True, callback=cb),
        LivePlotter1D("Frequency (MHz)", "Amplitude"),
        ticks=(fpts,),
        signal2real=qub_signal2real,
    )

    # get the actual frequency points
    fpts = prog.get_pulse_param("qubit_pulse", "freq", as_array=True)

    return fpts, signals


def visualize_qub_freq(soccfg, cfg, time_fly=0.0) -> None:
    cfg = deepcopy(cfg)  # prevent in-place modification

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "freq")

    sweep_cfg = cfg["sweep"]["freq"]
    params = sweep2param("freq", sweep_cfg)
    cfg["qub_pulse"]["freq"] = params

    visualizer = SimulateProgramV2(TwoToneProgram, soccfg, cfg)
    visualizer.visualize(time_fly=time_fly)
