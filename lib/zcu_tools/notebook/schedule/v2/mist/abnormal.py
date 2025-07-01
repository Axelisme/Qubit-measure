from copy import deepcopy
from typing import Tuple

import numpy as np
from zcu_tools.liveplot.jupyter import LivePlotter1D
from zcu_tools.notebook.single_qubit.process import rotate2real
from zcu_tools.program.v2 import TwoToneProgram
from zcu_tools.program.v2.base.simulate import SimulateProgramV2

from ...tools import format_sweep1D, sweep2array, sweep2param
from ..template import sweep_hard_template


def mist_signal2real(signal: np.ndarray) -> np.ndarray:
    return rotate2real(signal).real


def measure_abnormal_pdr_dep(soc, soccfg, cfg) -> Tuple[np.ndarray, np.ndarray]:
    cfg = deepcopy(cfg)  # prevent in-place modification

    if cfg["readout"]["type"] != "two_pulse":
        raise ValueError("Only two-pulse readout is supported")

    qub_pulse = cfg["qub_pulse"]
    res_pulse1 = cfg["readout"]["pulse1_cfg"]

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "gain")
    gain_sweep = cfg["sweep"]["gain"]

    cfg["sweep"] = {
        "ge": {"start": 0.0, "stop": res_pulse1["gain"], "expts": 2},
        "gain": gain_sweep,
    }

    # set with / without pi gain for qubit pulse
    qub_pulse["gain"] = sweep2param("gain", gain_sweep)
    res_pulse1["gain"] = sweep2param("ge", cfg["sweep"]["ge"])

    pdrs = sweep2array(gain_sweep)  # predicted pulse gains

    prog = TwoToneProgram(soccfg, cfg)

    signals = sweep_hard_template(
        cfg,
        lambda _, cb: prog.acquire(soc, progress=True, callback=cb),
        LivePlotter1D("Probe gain", "Signal", num_lines=2),
        ticks=(pdrs,),
        signal2real=mist_signal2real,
    )

    # get the actual pulse gains
    pdrs = prog.get_pulse_param("qubit_pulse", "gain", as_array=True)

    return pdrs, signals  # pdrs


def measure_abnormal_pdr_mux_reset(soc, soccfg, cfg) -> Tuple[np.ndarray, np.ndarray]:
    cfg = deepcopy(cfg)  # prevent in-place modification

    if cfg["readout"]["type"] != "two_pulse":
        raise ValueError("Only two-pulse readout is supported")

    qub_pulse = cfg["qub_pulse"]
    res_pulse1 = cfg["readout"]["pulse1_cfg"]

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "gain")
    pdr_sweep = cfg["sweep"]["gain"]

    cfg["sweep"] = {
        "ge": {"start": 0.0, "stop": res_pulse1["gain"], "expts": 2},
        "gain": pdr_sweep,
    }

    qub_pulse["gain"] = sweep2param("gain", pdr_sweep)
    res_pulse1["gain"] = sweep2param("ge", cfg["sweep"]["ge"])

    pdrs = sweep2array(pdr_sweep)  # predicted amplitudes

    prog = TwoToneProgram(soccfg, cfg)

    signals = sweep_hard_template(
        cfg,
        lambda _, cb: prog.acquire(soc, progress=True, callback=cb),
        LivePlotter1D("Pulse gain", "MIST", num_lines=2),
        ticks=(pdrs,),
        signal2real=mist_signal2real,
    )

    # get the actual amplitudes
    pdrs = prog.get_pulse_param("qubit_pulse", "gain", as_array=True)  # type: ignore

    return pdrs, signals


def visualize_abnormal_pdr_dep(soccfg, cfg, *, time_fly=0.0) -> None:
    cfg = deepcopy(cfg)  # prevent in-place modification

    if cfg["readout"]["type"] != "two_pulse":
        raise ValueError("Only two-pulse readout is supported")

    qub_pulse = cfg["qub_pulse"]
    res_pulse1 = cfg["readout"]["pulse1_cfg"]

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "gain")
    gain_sweep = cfg["sweep"]["gain"]

    cfg["sweep"] = {
        "ge": {"start": 0.0, "stop": res_pulse1["gain"], "expts": 2},
        "gain": gain_sweep,
    }

    # set with / without pi gain for qubit pulse
    qub_pulse["gain"] = sweep2param("gain", gain_sweep)
    res_pulse1["gain"] = sweep2param("ge", cfg["sweep"]["ge"])

    visualizer = SimulateProgramV2(TwoToneProgram, soccfg, cfg)
    visualizer.visualize(time_fly=time_fly)


def visualize_abnormal_pdr_mux_reset(soccfg, cfg, *, time_fly=0.0) -> None:
    cfg = deepcopy(cfg)  # prevent in-place modification

    if cfg["readout"]["type"] != "two_pulse":
        raise ValueError("Only two-pulse readout is supported")

    qub_pulse = cfg["qub_pulse"]
    res_pulse1 = cfg["readout"]["pulse1_cfg"]

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "gain")
    pdr_sweep = cfg["sweep"]["gain"]

    cfg["sweep"] = {
        "ge": {"start": 0.0, "stop": res_pulse1["gain"], "expts": 2},
        "gain": pdr_sweep,
    }

    qub_pulse["gain"] = sweep2param("gain", pdr_sweep)
    res_pulse1["gain"] = sweep2param("ge", cfg["sweep"]["ge"])

    visualizer = SimulateProgramV2(TwoToneProgram, soccfg, cfg)
    visualizer.visualize(time_fly=time_fly)
