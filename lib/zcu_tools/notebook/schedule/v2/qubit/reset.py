from copy import deepcopy
from typing import Optional, Tuple

import numpy as np
from zcu_tools.liveplot.jupyter import LivePlotter1D, LivePlotter2D
from zcu_tools.notebook.single_qubit.process import rotate2real
from zcu_tools.program.v2 import (
    MyProgramV2,
    OneToneProgram,
    ResetProgram,
    TwoPulseResetProgram,
    TwoToneProgram,
)
from zcu_tools.program.v2.base.simulate import SimulateProgramV2

from ...tools import format_sweep1D, sweep2array, sweep2param
from ..template import sweep_hard_template


def qub_signal2real(signals: np.ndarray) -> np.ndarray:
    return rotate2real(signals).real


def measure_reset_freq(soc, soccfg, cfg) -> Tuple[np.ndarray, np.ndarray]:
    cfg = deepcopy(cfg)  # prevent in-place modification
    reset_pulse = cfg["dac"]["reset_pulse"]

    if cfg["dac"]["reset"] != "pulse":
        raise ValueError("Reset pulse must be one pulse")

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "freq")

    sweep_cfg = cfg["sweep"]["freq"]
    reset_pulse["freq"] = sweep2param("freq", sweep_cfg)

    fpts = sweep2array(sweep_cfg)  # predicted frequency points

    prog: Optional[OneToneProgram] = None

    def measure_fn(cfg, callback) -> list:
        nonlocal prog
        prog = OneToneProgram(soccfg, cfg)
        return prog.acquire(soc, progress=True, callback=callback)

    signals = sweep_hard_template(
        cfg,
        measure_fn,
        LivePlotter1D("Frequency (MHz)", "Amplitude"),
        ticks=(fpts,),
        signal2real=qub_signal2real,
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

    prog: Optional[OneToneProgram] = None

    def measure_fn(cfg, callback) -> list:
        nonlocal prog
        prog = OneToneProgram(soccfg, cfg)
        return prog.acquire(soc, progress=True, callback=callback)

    signals = sweep_hard_template(
        cfg,
        measure_fn,
        LivePlotter2D("Frequency1 (MHz)", "Frequency2 (MHz)"),
        ticks=(fpts1, fpts2),
        signal2real=qub_signal2real,
    )

    # get the actual frequency points
    fpts1 = prog.get_pulse_param("reset1", "freq", as_array=True)
    fpts2 = prog.get_pulse_param("reset2", "freq", as_array=True)

    return fpts1, fpts2, signals


def measure_mux_reset_pdr(
    soc, soccfg, cfg
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cfg = deepcopy(cfg)  # prevent in-place modification

    reset_test_pulse1 = cfg["dac"]["reset_test_pulse1"]
    reset_test_pulse2 = cfg["dac"]["reset_test_pulse2"]

    # force gain1 to be the outer loop
    cfg["sweep"] = {
        "gain1": cfg["sweep"]["gain1"],
        "gain2": cfg["sweep"]["gain2"],
    }

    reset_test_pulse1["gain"] = sweep2param("gain1", cfg["sweep"]["gain1"])
    reset_test_pulse2["gain"] = sweep2param("gain2", cfg["sweep"]["gain2"])

    pdrs1 = sweep2array(cfg["sweep"]["gain1"])  # predicted amplitudes
    pdrs2 = sweep2array(cfg["sweep"]["gain2"])  # predicted amplitudes

    prog: Optional[TwoPulseResetProgram] = None

    def measure_fn(cfg, callback) -> list:
        nonlocal prog
        prog = TwoPulseResetProgram(soccfg, cfg)
        return prog.acquire(soc, progress=True, callback=callback)

    signals = sweep_hard_template(
        cfg,
        measure_fn,
        LivePlotter2D("Gain1", "Gain2"),
        ticks=(pdrs1, pdrs2),
        signal2real=qub_signal2real,
    )

    # get the actual frequency points
    pdrs1 = prog.get_pulse_param("reset_test1", "gain", as_array=True)
    pdrs2 = prog.get_pulse_param("reset_test2", "gain", as_array=True)

    return pdrs1, pdrs2, signals


def measure_reset_time(soc, soccfg, cfg) -> Tuple[np.ndarray, np.ndarray]:
    cfg = deepcopy(cfg)  # prevent in-place modification

    qub_pulse = cfg["dac"]["qub_pulse"]

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")

    # prepend ge sweep to inner loop
    cfg["sweep"] = {
        "ge": {"start": 0, "stop": qub_pulse["gain"], "expts": 2},
        "length": cfg["sweep"]["length"],
    }

    # set with / without pi gain for qubit pulse
    qub_pulse["gain"] = sweep2param("ge", cfg["sweep"]["ge"])

    len_params = sweep2param("length", cfg["sweep"]["length"])
    if cfg["dac"]["reset"] == "pulse":
        cfg["dac"]["reset_pulse"]["length"] = len_params
    elif cfg["dac"]["reset"] == "two_pulse":
        cfg["dac"]["reset_pulse1"]["length"] = len_params
        cfg["dac"]["reset_pulse2"]["length"] = len_params
    else:
        raise ValueError(f"Reset type {cfg['dac']['reset']} not supported")

    lens = sweep2array(cfg["sweep"]["length"])  # predicted pulse gains

    def result2signals(avg_d: list, std_d: list) -> Tuple[np.ndarray, np.ndarray]:
        avg_d = avg_d[0][0].dot([1, 1j])  # (ge, *sweep)
        std_d = std_d[0][0].dot([1, 1j])  # (ge, *sweep)
        avg_d = avg_d[1, ...] - avg_d[0, ...]  # (*sweep)
        std_d = np.sqrt(std_d[1, ...] ** 2 + std_d[0, ...] ** 2)  # (*sweep)

        return avg_d, std_d

    prog: Optional[TwoToneProgram] = None

    def measure_fn(cfg, callback) -> list:
        nonlocal prog
        prog = TwoToneProgram(soccfg, cfg)
        return prog.acquire(soc, progress=True, callback=callback)

    signals = sweep_hard_template(
        cfg,
        measure_fn,
        LivePlotter1D("Length (us)", "Amplitude"),
        ticks=(lens,),
        result2signals=result2signals,
    )

    # get the actual pulse gains and frequency points
    pulse_name = "reset" if cfg["dac"]["reset"] == "pulse" else "reset1"
    real_lens = prog.get_pulse_param(pulse_name, "length", as_array=True)
    real_lens += lens[0] - real_lens[0]  # adjust to the first length

    return real_lens, signals  # lens


def measure_reset_amprabi(soc, soccfg, cfg) -> Tuple[np.ndarray, np.ndarray]:
    cfg = deepcopy(cfg)  # prevent in-place modification

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "gain")
    gain_sweep = cfg["sweep"]["gain"]

    cfg["sweep"] = {
        "w/o_reset": {"start": 0, "stop": 1.0, "expts": 2},
        "gain": gain_sweep,
    }

    cfg["dac"]["qub_pulse"]["gain"] = sweep2param("gain", gain_sweep)

    reset_factor = sweep2param("w/o_reset", cfg["sweep"]["w/o_reset"])
    if cfg["dac"]["reset_test"] == "pulse":
        reset_test_pulse = cfg["dac"]["reset_test_pulse"]
        reset_test_pulse["gain"] = reset_factor * reset_test_pulse["gain"]
        reset_test_pulse["length"] = reset_factor * reset_test_pulse["length"]
        if reset_test_pulse["style"] == "flat_top":
            reset_test_pulse["length"] += (
                reset_test_pulse["raise_pulse"]["length"] + 0.005
            )
    elif cfg["dac"]["reset_test"] == "two_pulse":
        reset_test_pulse1 = cfg["dac"]["reset_test_pulse1"]
        reset_test_pulse2 = cfg["dac"]["reset_test_pulse2"]
        reset_test_pulse1["gain"] = reset_factor * reset_test_pulse1["gain"]
        reset_test_pulse2["gain"] = reset_factor * reset_test_pulse2["gain"]
        reset_test_pulse1["length"] = reset_factor * reset_test_pulse1["length"] + 0.01
        reset_test_pulse2["length"] = reset_factor * reset_test_pulse2["length"] + 0.01
        if reset_test_pulse1["style"] == "flat_top":
            reset_test_pulse1["length"] += reset_test_pulse1["raise_pulse"]["length"]
            reset_test_pulse2["length"] += reset_test_pulse2["raise_pulse"]["length"]
    else:
        raise ValueError(f"Reset type {cfg['dac']['reset_test']} not supported")

    pdrs = sweep2array(gain_sweep)  # predicted amplitudes

    prog: Optional[MyProgramV2] = None

    def measure_fn(cfg, callback) -> Tuple[np.ndarray, np.ndarray]:
        nonlocal prog
        if cfg["dac"]["reset_test"] == "pulse":
            prog = ResetProgram(soccfg, cfg)
        elif cfg["dac"]["reset_test"] == "two_pulse":
            prog = TwoPulseResetProgram(soccfg, cfg)
        else:
            raise ValueError(f"Reset type {cfg['dac']['reset']} not supported")

        return prog.acquire(soc, progress=False, callback=callback)

    signals = sweep_hard_template(
        cfg,
        measure_fn,
        LivePlotter1D("Pulse gain", "Amplitude", num_lines=2),
        ticks=(pdrs,),
        signal2real=qub_signal2real,
    )

    pdrs = prog.get_pulse_param("qub_pulse", "gain", as_array=True)

    return pdrs, signals


def visualize_reset_time(soccfg, cfg, *, time_fly=0.0) -> None:
    cfg = deepcopy(cfg)  # prevent in-place modification

    qub_pulse = cfg["dac"]["qub_pulse"]

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")

    # prepend ge sweep to inner loop
    cfg["sweep"] = {
        "ge": {"start": 0, "stop": qub_pulse["gain"], "expts": 2},
        "length": cfg["sweep"]["length"],
    }

    # set with / without pi gain for qubit pulse
    qub_pulse["gain"] = sweep2param("ge", cfg["sweep"]["ge"])

    len_params = sweep2param("length", cfg["sweep"]["length"])
    if cfg["dac"]["reset"] == "pulse":
        cfg["dac"]["reset_pulse"]["length"] = len_params
    elif cfg["dac"]["reset"] == "two_pulse":
        cfg["dac"]["reset_pulse1"]["length"] = len_params
        cfg["dac"]["reset_pulse2"]["length"] = len_params
    else:
        raise ValueError(f"Reset type {cfg['dac']['reset']} not supported")

    visualizer = SimulateProgramV2(TwoToneProgram, soccfg, cfg)
    visualizer.visualize(time_fly=time_fly)


def visualize_reset_amprabi(soccfg, cfg, *, time_fly=0.0) -> None:
    cfg = deepcopy(cfg)  # prevent in-place modification

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "gain")
    gain_sweep = cfg["sweep"]["gain"]

    cfg["sweep"] = {
        "w/o_reset": {"start": 0, "stop": 1.0, "expts": 2},
        "gain": gain_sweep,
    }

    cfg["dac"]["qub_pulse"]["gain"] = sweep2param("gain", gain_sweep)

    reset_factor = sweep2param("w/o_reset", cfg["sweep"]["w/o_reset"])
    if cfg["dac"]["reset_test"] == "pulse":
        reset_test_pulse = cfg["dac"]["reset_test_pulse"]
        reset_test_pulse["gain"] = reset_factor * reset_test_pulse["gain"]
    elif cfg["dac"]["reset_test"] == "two_pulse":
        reset_test_pulse1 = cfg["dac"]["reset_test_pulse1"]
        reset_test_pulse2 = cfg["dac"]["reset_test_pulse2"]
        reset_test_pulse1["gain"] = reset_factor * reset_test_pulse1["gain"]
        reset_test_pulse2["gain"] = reset_factor * reset_test_pulse2["gain"]
    else:
        raise ValueError(f"Reset type {cfg['dac']['reset_test']} not supported")

    if cfg["dac"]["reset_test"] == "pulse":
        progCls = ResetProgram
    elif cfg["dac"]["reset_test"] == "two_pulse":
        progCls = TwoPulseResetProgram
    else:
        raise ValueError(f"Reset type {cfg['dac']['reset']} not supported")

    visualizer = SimulateProgramV2(progCls, soccfg, cfg)
    visualizer.visualize(time_fly=time_fly)
