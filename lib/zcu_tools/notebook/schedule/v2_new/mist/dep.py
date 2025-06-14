from typing import Optional, Tuple

import numpy as np
from numpy import ndarray
from zcu_tools import make_cfg
from zcu_tools.liveplot.jupyter import LivePlotter1D, LivePlotter2DwithLine
from zcu_tools.notebook.single_qubit.process import rotate2real
from zcu_tools.program.v2 import TwoPulseResetProgram, TwoToneProgram
from zcu_tools.program.v2.base.simulate import SimulateProgramV2

from ...tools import format_sweep1D, sweep2array, sweep2param
from ..template import sweep2D_soft_hard_template, sweep_hard_template


def mist_signal2real(signal: ndarray) -> ndarray:
    return rotate2real(signal).real


def measure_mist_pdr_dep(
    soc, soccfg, cfg, backend_mode=False
) -> Tuple[ndarray, ndarray]:
    cfg = make_cfg(cfg)  # prevent in-place modification

    qub_pulse = cfg["dac"]["qub_pulse"]

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "gain")
    pdr_sweep = cfg["sweep"]["gain"]

    qub_pulse["gain"] = sweep2param("gain", pdr_sweep)

    pdrs = sweep2array(pdr_sweep)  # predicted amplitudes

    prog: Optional[TwoToneProgram] = None

    def measure_fn(cfg, callback) -> Tuple[list, list]:
        nonlocal prog
        prog = TwoToneProgram(soccfg, cfg)
        return prog.acquire(soc, progress=not backend_mode, callback=callback)

    signals = sweep_hard_template(
        cfg,
        measure_fn,
        LivePlotter1D("Pulse gain", "MIST", disable=backend_mode),
        ticks=(pdrs,),
        signal2real=mist_signal2real,
        catch_interrupt=not backend_mode,
    )

    # get the actual amplitudes
    pdrs: ndarray = prog.get_pulse_param("qub_pulse", "gain", as_array=True)  # type: ignore

    return pdrs, signals


def measure_mist_pdr_mux_reset(soc, soccfg, cfg) -> Tuple[ndarray, ndarray]:
    cfg = make_cfg(cfg)  # prevent in-place modification

    qub_pulse = cfg["dac"]["qub_pulse"]
    reset_test_pulse1 = cfg["dac"]["reset_test_pulse1"]
    reset_test_pulse2 = cfg["dac"]["reset_test_pulse2"]

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "gain")
    pdr_sweep = cfg["sweep"]["gain"]

    cfg["sweep"] = {
        "w/o_reset": {"start": 0.0, "stop": 1.0, "expts": 2},
        "gain": pdr_sweep,
    }

    qub_pulse["gain"] = sweep2param("gain", pdr_sweep)
    reset_factor = sweep2param("w/o_reset", cfg["sweep"]["w/o_reset"])
    reset_test_pulse1["gain"] = reset_factor * reset_test_pulse1["gain"]
    reset_test_pulse2["gain"] = reset_factor * reset_test_pulse2["gain"]
    reset_test_pulse1["length"] = reset_factor * reset_test_pulse1["length"] + 0.01
    reset_test_pulse2["length"] = reset_factor * reset_test_pulse2["length"] + 0.01
    if reset_test_pulse1["style"] == "flat_top":
        reset_test_pulse1["length"] += reset_test_pulse1["raise_pulse"]["length"]
        reset_test_pulse2["length"] += reset_test_pulse2["raise_pulse"]["length"]

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
    pdrs: ndarray = prog.get_pulse_param("qub_pulse", "gain", as_array=True)  # type: ignore

    return pdrs, signals


def measure_mist_flx_pdr_dep2D(soc, soccfg, cfg) -> Tuple[ndarray, ndarray, ndarray]:
    cfg = make_cfg(cfg)  # prevent in-place modification

    pdr_sweep = cfg["sweep"]["gain"]
    flx_sweep = cfg["sweep"]["flux"]

    del cfg["sweep"]["flux"]  # use soft loop
    cfg["dac"]["qub_pulse"]["gain"] = sweep2param("gain", pdr_sweep)

    As = sweep2array(flx_sweep, allow_array=True)  # predicted currents
    pdrs = sweep2array(pdr_sweep)  # predicted gains

    cfg["dev"]["flux"] = As[0]  # set initial flux

    def updateCfg(cfg, _, mA):
        cfg["dev"]["flux"] = mA * 1e-3  # convert to A

    prog: Optional[TwoToneProgram] = None

    def measure_fn(cfg, callback) -> Tuple[list, list]:
        nonlocal prog
        prog = TwoToneProgram(soccfg, cfg)
        return prog.acquire(soc, progress=False, callback=callback)

    def signal2real(signal: ndarray) -> ndarray:
        return np.abs(signal - signal[:, 0][:, None])

    signals2D = sweep2D_soft_hard_template(
        cfg,
        measure_fn,
        LivePlotter2DwithLine(
            "Flux (mA)", "Drive power (a.u.)", line_axis=1, num_lines=2
        ),
        xs=1e3 * As,
        ys=pdrs,
        updateCfg=updateCfg,
        signal2real=signal2real,
    )

    # get the actual lengths
    pdrs: ndarray = prog.get_pulse_param("qub_pulse", "gain", as_array=True)  # type: ignore

    return As, pdrs, signals2D


def visualize_mist_pdr_dep(soccfg, cfg, *, time_fly=0.0) -> None:
    cfg = make_cfg(cfg)  # prevent in-place modification

    qub_pulse = cfg["dac"]["qub_pulse"]

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "gain")
    pdr_sweep = cfg["sweep"]["gain"]

    qub_pulse["gain"] = sweep2param("gain", pdr_sweep)

    visualizer = SimulateProgramV2(TwoToneProgram, soccfg, cfg)
    visualizer.visualize(time_fly=time_fly)


def visualize_mist_pdr_mux_reset(soccfg, cfg, *, time_fly=0.0) -> None:
    cfg = make_cfg(cfg)  # prevent in-place modification

    qub_pulse = cfg["dac"]["qub_pulse"]
    reset_test_pulse1 = cfg["dac"]["reset_test_pulse1"]
    reset_test_pulse2 = cfg["dac"]["reset_test_pulse2"]

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "gain")
    pdr_sweep = cfg["sweep"]["gain"]

    cfg["sweep"] = {
        "w/o_reset": {"start": 0.0, "stop": 1.0, "expts": 2},
        "gain": pdr_sweep,
    }

    qub_pulse["gain"] = sweep2param("gain", pdr_sweep)
    reset_factor = sweep2param("w/o_reset", cfg["sweep"]["w/o_reset"])
    reset_test_pulse1["gain"] = reset_factor * reset_test_pulse1["gain"]
    reset_test_pulse2["gain"] = reset_factor * reset_test_pulse2["gain"]

    visualizer = SimulateProgramV2(TwoPulseResetProgram, soccfg, cfg)
    visualizer.visualize(time_fly=time_fly)
