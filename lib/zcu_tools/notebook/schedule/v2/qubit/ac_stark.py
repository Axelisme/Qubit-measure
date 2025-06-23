from copy import deepcopy
from typing import Tuple

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

    cfg["stark_pulse1"]["gain"] = sweep2param("gain", gain_sweep)
    cfg["stark_pulse2"]["freq"] = sweep2param("freq", freq_sweep)

    amps = sweep2array(gain_sweep)
    freqs = sweep2array(freq_sweep)

    prog = ACStarkProgram(soccfg, cfg)

    signals = sweep_hard_template(
        cfg,
        lambda _, cb: prog.acquire(soc, progress=True, callback=cb),
        LivePlotter2D("Pulse gain", "Frequency (MHz)"),
        ticks=(amps, freqs),
        signal2real=qub_signal2real,
    )

    # get the actual amplitudes
    amps = prog.get_pulse_param("stark_pulse1", "gain", as_array=True)
    freqs = prog.get_pulse_param("stark_pulse2", "freq", as_array=True)

    return amps, freqs, signals


def visualize_ac_stark(soccfg, cfg, *, time_fly=0.0) -> None:
    cfg = deepcopy(cfg)  # prevent in-place modification

    gain_sweep = cfg["sweep"]["gain"]
    freq_sweep = cfg["sweep"]["freq"]
    cfg["sweep"] = {"gain": gain_sweep, "freq": freq_sweep}

    cfg["stark_pulse1"]["gain"] = sweep2param("gain", gain_sweep)
    cfg["stark_pulse2"]["freq"] = sweep2param("freq", freq_sweep)

    visualizer = SimulateProgramV2(ACStarkProgram, soccfg, cfg)
    visualizer.visualize(time_fly=time_fly)
