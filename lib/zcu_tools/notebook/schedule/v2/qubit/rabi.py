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


def measure_lenrabi(soc, soccfg, cfg) -> Tuple[np.ndarray, np.ndarray]:
    cfg = deepcopy(cfg)  # prevent in-place modification

    qub_pulse = cfg["qub_pulse"]

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")
    len_sweep = cfg["sweep"]["length"]

    qub_pulse["length"] = sweep2param("length", len_sweep)

    lens = sweep2array(len_sweep)  # predicted lengths

    prog = TwoToneProgram(soccfg, cfg)

    signals = sweep_hard_template(
        cfg,
        lambda _, cb: prog.acquire(soc, progress=True, callback=cb),
        LivePlotter1D("Length (us)", "Amplitude"),
        ticks=(lens,),
        signal2real=qub_signal2real,
    )

    # get the actual lengths
    real_lens = prog.get_pulse_param("qubit_pulse", "length", as_array=True)

    # TODO: better way to do this?
    real_lens += lens[0] - real_lens[0]  # add back the side length of the pulse

    return real_lens, signals


def visualize_lenrabi(soccfg, cfg, *, time_fly=0.0) -> None:
    cfg = deepcopy(cfg)  # prevent in-place modification

    qub_pulse = cfg["qub_pulse"]

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")
    len_sweep = cfg["sweep"]["length"]

    qub_pulse["length"] = sweep2param("length", len_sweep)

    visualizer = SimulateProgramV2(TwoToneProgram, soccfg, cfg)
    visualizer.visualize(time_fly=time_fly)


def measure_amprabi(soc, soccfg, cfg) -> Tuple[np.ndarray, np.ndarray]:
    cfg = deepcopy(cfg)  # prevent in-place modification

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "gain")

    sweep_cfg = cfg["sweep"]["gain"]
    cfg["qub_pulse"]["gain"] = sweep2param("gain", sweep_cfg)

    amps = sweep2array(sweep_cfg)  # predicted amplitudes

    prog = TwoToneProgram(soccfg, cfg)

    signals = sweep_hard_template(
        cfg,
        lambda _, cb: prog.acquire(soc, progress=True, callback=cb),
        LivePlotter1D("Pulse gain", "Amplitude"),
        ticks=(amps,),
        signal2real=qub_signal2real,
    )

    # get the actual amplitudes
    amps = prog.get_pulse_param("qubit_pulse", "gain", as_array=True)

    return amps, signals


def visualize_amprabi(soccfg, cfg, *, time_fly=0.0) -> None:
    cfg = deepcopy(cfg)  # prevent in-place modification

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "gain")

    sweep_cfg = cfg["sweep"]["gain"]
    cfg["qub_pulse"]["gain"] = sweep2param("gain", sweep_cfg)

    visualizer = SimulateProgramV2(TwoToneProgram, soccfg, cfg)
    visualizer.visualize(time_fly=time_fly)
