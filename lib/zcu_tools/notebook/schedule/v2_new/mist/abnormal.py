from copy import deepcopy
from typing import Optional, Tuple

import numpy as np
from zcu_tools.liveplot.jupyter import LivePlotter1D
from zcu_tools.notebook.single_qubit.process import rotate2real
from zcu_tools.program.v2 import TwoPulseResetProgram, TwoToneProgram
from zcu_tools.program.v2.base.simulate import SimulateProgramV2

from ...tools import format_sweep1D, sweep2array, sweep2param
from ..template import sweep_hard_template


def mist_signal2real(signal: np.ndarray) -> np.ndarray:
    return rotate2real(signal).real


def measure_abnormal_pdr_dep(soc, soccfg, cfg) -> Tuple[np.ndarray, np.ndarray]:
    cfg = deepcopy(cfg)  # prevent in-place modification

    qub_pulse = cfg["dac"]["qub_pulse"]
    pre_res_pulse = cfg["dac"]["pre_res_pulse"]

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "gain")
    gain_sweep = cfg["sweep"]["gain"]

    cfg["sweep"] = {
        "ge": {"start": 0.0, "stop": pre_res_pulse["gain"], "expts": 2},
        "gain": gain_sweep,
    }

    # set with / without pi gain for qubit pulse
    qub_pulse["gain"] = sweep2param("gain", gain_sweep)
    pre_res_pulse["gain"] = sweep2param("ge", cfg["sweep"]["ge"])

    pdrs = sweep2array(gain_sweep)  # predicted pulse gains

    prog: Optional[TwoToneProgram] = None

    def measure_fn(cfg, callback) -> Tuple[list, list]:
        nonlocal prog
        prog = TwoToneProgram(soccfg, cfg)
        return prog.acquire(soc, progress=True, callback=callback)

    signals = sweep_hard_template(
        cfg,
        measure_fn,
        LivePlotter1D("Probe gain", "Signal", num_lines=2),
        ticks=(pdrs,),
        signal2real=mist_signal2real,
    )

    # get the actual pulse gains
    pdrs = prog.get_pulse_param("qub_pulse", "gain", as_array=True)

    return pdrs, signals  # pdrs


def measure_abnormal_pdr_mux_reset(soc, soccfg, cfg) -> Tuple[np.ndarray, np.ndarray]:
    cfg = deepcopy(cfg)  # prevent in-place modification

    qub_pulse = cfg["dac"]["qub_pulse"]
    pre_res_pulse = cfg["dac"]["pre_res_pulse"]

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "gain")
    pdr_sweep = cfg["sweep"]["gain"]

    cfg["sweep"] = {
        "ge": {"start": 0.0, "stop": pre_res_pulse["gain"], "expts": 2},
        "gain": pdr_sweep,
    }

    qub_pulse["gain"] = sweep2param("gain", pdr_sweep)
    pre_res_pulse["gain"] = sweep2param("ge", cfg["sweep"]["ge"])

    pdrs = sweep2array(pdr_sweep)  # predicted amplitudes

    prog: Optional[TwoPulseResetProgram] = None

    def measure_fn(cfg, callback) -> Tuple[list, list]:
        nonlocal prog
        prog = TwoPulseResetProgram(soccfg, cfg)
        return prog.acquire(soc, progress=True, callback=callback)

    signals = sweep_hard_template(
        cfg,
        measure_fn,
        LivePlotter1D("Pulse gain", "MIST", num_lines=2),
        ticks=(pdrs,),
        signal2real=mist_signal2real,
    )

    # get the actual amplitudes
    pdrs: np.ndarray = prog.get_pulse_param("qub_pulse", "gain", as_array=True)  # type: ignore

    return pdrs, signals


def visualize_abnormal_pdr_dep(soccfg, cfg, *, time_fly=0.0) -> None:
    cfg = deepcopy(cfg)  # prevent in-place modification

    qub_pulse = cfg["dac"]["qub_pulse"]
    pre_res_pulse = cfg["dac"]["pre_res_pulse"]

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "gain")
    gain_sweep = cfg["sweep"]["gain"]

    cfg["sweep"] = {
        "ge": {"start": 0.0, "stop": pre_res_pulse["gain"], "expts": 2},
        "gain": gain_sweep,
    }

    # set with / without pi gain for qubit pulse
    qub_pulse["gain"] = sweep2param("gain", gain_sweep)
    pre_res_pulse["gain"] = sweep2param("ge", cfg["sweep"]["ge"])

    visualizer = SimulateProgramV2(TwoToneProgram, soccfg, cfg)
    visualizer.visualize(time_fly=time_fly)


def visualize_abnormal_pdr_mux_reset(soccfg, cfg, *, time_fly=0.0) -> None:
    cfg = deepcopy(cfg)  # prevent in-place modification

    qub_pulse = cfg["dac"]["qub_pulse"]
    pre_res_pulse = cfg["dac"]["pre_res_pulse"]

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "gain")
    pdr_sweep = cfg["sweep"]["gain"]

    cfg["sweep"] = {
        "ge": {"start": 0.0, "stop": pre_res_pulse["gain"], "expts": 2},
        "gain": pdr_sweep,
    }

    qub_pulse["gain"] = sweep2param("gain", pdr_sweep)
    pre_res_pulse["gain"] = sweep2param("ge", cfg["sweep"]["ge"])

    visualizer = SimulateProgramV2(TwoPulseResetProgram, soccfg, cfg)
    visualizer.visualize(time_fly=time_fly)
