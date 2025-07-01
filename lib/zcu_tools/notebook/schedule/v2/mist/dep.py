from copy import deepcopy
from typing import Tuple

import numpy as np
from numpy import ndarray
from zcu_tools.liveplot.jupyter import LivePlotter1D, LivePlotter2DwithLine
from zcu_tools.notebook.single_qubit.process import rotate2real
from zcu_tools.program.v2 import ResetProbeProgram, TwoToneProgram
from zcu_tools.program.v2.base.simulate import SimulateProgramV2

from ...tools import format_sweep1D, sweep2array, sweep2param
from ..template import sweep2D_soft_hard_template, sweep_hard_template


def mist_signal2real(signal: ndarray) -> ndarray:
    return rotate2real(signal).real


def measure_mist_pdr_dep(soc, soccfg, cfg, liveplot=False) -> Tuple[ndarray, ndarray]:
    cfg = deepcopy(cfg)  # prevent in-place modification

    qub_pulse = cfg["qub_pulse"]

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "gain")
    pdr_sweep = cfg["sweep"]["gain"]

    qub_pulse["gain"] = sweep2param("gain", pdr_sweep)

    pdrs = sweep2array(pdr_sweep)  # predicted amplitudes

    prog = TwoToneProgram(soccfg, cfg)

    signals = sweep_hard_template(
        cfg,
        lambda _, cb: prog.acquire(soc, progress=liveplot, callback=cb),
        LivePlotter1D("Pulse gain", "MIST", disable=not liveplot),
        ticks=(pdrs,),
        signal2real=mist_signal2real,
        catch_interrupt=liveplot,
    )

    # get the actual amplitudes
    pdrs = prog.get_pulse_param("qubit_pulse", "gain", as_array=True)  # type: ignore

    return pdrs, signals


def measure_mist_pdr_mux_reset(soc, soccfg, cfg) -> Tuple[ndarray, ndarray]:
    cfg = deepcopy(cfg)  # prevent in-place modification

    if cfg["readout"]["type"] != "two_pulse":
        raise ValueError("Only two-pulse readout is supported")

    qub_pulse = cfg["qub_pulse"]
    res_pulse1 = cfg["readout"]["pulse1_cfg"]
    res_pulse2 = cfg["readout"]["pulse2_cfg"]

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "gain")
    pdr_sweep = cfg["sweep"]["gain"]

    cfg["sweep"] = {
        "w/o_reset": {"start": 0.0, "stop": 1.0, "expts": 2},
        "gain": pdr_sweep,
    }

    qub_pulse["gain"] = sweep2param("gain", pdr_sweep)

    # TODO: better way to handle this
    reset_factor = sweep2param("w/o_reset", cfg["sweep"]["w/o_reset"])
    res_pulse1["gain"] = reset_factor * res_pulse1["gain"]
    res_pulse2["gain"] = reset_factor * res_pulse2["gain"]
    res_pulse1["length"] = reset_factor * res_pulse1["length"] + 0.005
    res_pulse2["length"] = reset_factor * res_pulse2["length"] + 0.005
    if res_pulse1["style"] == "flat_top":
        res_pulse1["length"] += res_pulse1["raise_pulse"]["length"]
    if res_pulse2["style"] == "flat_top":
        res_pulse2["length"] += res_pulse2["raise_pulse"]["length"]

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


def measure_mist_flx_pdr_dep2D(soc, soccfg, cfg) -> Tuple[ndarray, ndarray, ndarray]:
    cfg = deepcopy(cfg)  # prevent in-place modification

    pdr_sweep = cfg["sweep"]["gain"]
    flx_sweep = cfg["sweep"]["flux"]

    del cfg["sweep"]["flux"]  # use soft loop
    cfg["qub_pulse"]["gain"] = sweep2param("gain", pdr_sweep)

    As = sweep2array(flx_sweep, allow_array=True)  # predicted currents
    pdrs = sweep2array(pdr_sweep)  # predicted gains

    cfg["dev"]["flux"] = As[0]  # set initial flux

    def updateCfg(cfg, _, mA):
        cfg["dev"]["flux"] = mA * 1e-3  # convert to A

    def signal2real(signal: ndarray) -> ndarray:
        return np.abs(signal - signal[:, 0][:, None])

    signals2D = sweep2D_soft_hard_template(
        cfg,
        lambda _, cb: TwoToneProgram(soccfg, cfg).acquire(
            soc, progress=False, callback=cb
        ),
        LivePlotter2DwithLine(
            "Flux (mA)", "Drive power (a.u.)", line_axis=1, num_lines=2
        ),
        xs=1e3 * As,
        ys=pdrs,
        updateCfg=updateCfg,
        signal2real=signal2real,
    )

    # get the actual lengths
    prog = TwoToneProgram(soccfg, cfg)
    pdrs = prog.get_pulse_param("qubit_pulse", "gain", as_array=True)  # type: ignore

    return As, pdrs, signals2D


def visualize_mist_pdr_dep(soccfg, cfg, *, time_fly=0.0) -> None:
    cfg = deepcopy(cfg)  # prevent in-place modification

    qub_pulse = cfg["qub_pulse"]

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "gain")
    pdr_sweep = cfg["sweep"]["gain"]

    qub_pulse["gain"] = sweep2param("gain", pdr_sweep)

    visualizer = SimulateProgramV2(TwoToneProgram, soccfg, cfg)
    visualizer.visualize(time_fly=time_fly)


def visualize_mist_pdr_mux_reset(soccfg, cfg, *, time_fly=0.0) -> None:
    cfg = deepcopy(cfg)  # prevent in-place modification

    if cfg["readout"]["type"] != "two_pulse":
        raise ValueError("Only two-pulse readout is supported")

    qub_pulse = cfg["qub_pulse"]
    res_pulse1 = cfg["readout"]["pulse1_cfg"]
    res_pulse2 = cfg["readout"]["pulse2_cfg"]

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "gain")
    pdr_sweep = cfg["sweep"]["gain"]

    cfg["sweep"] = {
        "w/o_reset": {"start": 0.0, "stop": 1.0, "expts": 2},
        "gain": pdr_sweep,
    }

    qub_pulse["gain"] = sweep2param("gain", pdr_sweep)

    # TODO: better way to handle this
    reset_factor = sweep2param("w/o_reset", cfg["sweep"]["w/o_reset"])
    res_pulse1["gain"] = reset_factor * res_pulse1["gain"]
    res_pulse2["gain"] = reset_factor * res_pulse2["gain"]
    res_pulse1["length"] = reset_factor * res_pulse1["length"] + 0.005
    res_pulse2["length"] = reset_factor * res_pulse2["length"] + 0.005
    if res_pulse1["style"] == "flat_top":
        res_pulse1["length"] += res_pulse1["raise_pulse"]["length"]
    if res_pulse2["style"] == "flat_top":
        res_pulse2["length"] += res_pulse2["raise_pulse"]["length"]

    visualizer = SimulateProgramV2(TwoToneProgram, soccfg, cfg)
    visualizer.visualize(time_fly=time_fly)
