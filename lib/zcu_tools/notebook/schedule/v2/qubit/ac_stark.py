from copy import deepcopy
from typing import Optional, Tuple

import numpy as np
from zcu_tools.liveplot.jupyter import LivePlotter2D
from zcu_tools.notebook.single_qubit.process import minus_background, rotate2real
from zcu_tools.program.v2 import ACStarkProgram
from zcu_tools.program.v2.base.simulate import SimulateProgramV2

from ...tools import sweep2array, sweep2param
from ..template import sweep_hard_template


def qub_signal2real(signals: np.ndarray) -> np.ndarray:
    return rotate2real(minus_background(signals, axis=1)).real


def measure_ac_stark(soc, soccfg, cfg) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cfg = deepcopy(cfg)  # prevent in-place modification

    # force the order of sweep
    gain_sweep = cfg["sweep"]["gain"]
    freq_sweep = cfg["sweep"]["freq"]
    cfg["sweep"] = {"gain": gain_sweep, "freq": freq_sweep}

    cfg["dac"]["stark_res_pulse"]["gain"] = sweep2param("gain", gain_sweep)
    cfg["dac"]["stark_qub_pulse"]["freq"] = sweep2param("freq", freq_sweep)

    amps = sweep2array(gain_sweep)
    freqs = sweep2array(freq_sweep)

    prog: Optional[ACStarkProgram] = None

    def measure_fn(cfg, callback) -> Tuple[list, list]:
        nonlocal prog
        prog = ACStarkProgram(soccfg, cfg)
        return prog.acquire(soc, progress=True, callback=callback)

    signals = sweep_hard_template(
        cfg,
        measure_fn,
        LivePlotter2D("Pulse gain", "Frequency (MHz)"),
        ticks=(amps, freqs),
        signal2real=qub_signal2real,
    )

    # get the actual amplitudes
    amps: np.ndarray = prog.get_pulse_param("stark_res_pulse", "gain", as_array=True)
    freqs: np.ndarray = prog.get_pulse_param("stark_qub_pulse", "freq", as_array=True)

    return amps, freqs, signals


def visualize_ac_stark(soccfg, cfg, *, time_fly=0.0) -> None:
    cfg = deepcopy(cfg)  # prevent in-place modification

    gain_sweep = cfg["sweep"]["gain"]
    freq_sweep = cfg["sweep"]["freq"]
    cfg["sweep"] = {"gain": gain_sweep, "freq": freq_sweep}

    cfg["dac"]["stark_res_pulse"]["gain"] = sweep2param("gain", gain_sweep)
    cfg["dac"]["stark_qub_pulse"]["freq"] = sweep2param("freq", freq_sweep)

    visualizer = SimulateProgramV2(ACStarkProgram, soccfg, cfg)
    visualizer.visualize(time_fly=time_fly)
