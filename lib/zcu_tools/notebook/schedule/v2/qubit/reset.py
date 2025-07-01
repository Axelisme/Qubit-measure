from copy import deepcopy
from typing import Optional, Tuple

import numpy as np
from zcu_tools.liveplot.jupyter import LivePlotter1D, LivePlotter2D
from zcu_tools.notebook.single_qubit.process import rotate2real
from zcu_tools.program.v2 import ResetProbeProgram, TwoToneProgram

from ...tools import format_sweep1D, sweep2array, sweep2param
from ..template import sweep_hard_template


def reset_result2signal(avg_d: list, std_d: list) -> Tuple[np.ndarray, np.ndarray]:
    avg_d = avg_d[0][0].dot([1, 1j])  # (ge, *sweep)
    std_d = std_d[0][0].dot([1, 1j])  # (ge, *sweep)
    avg_d = avg_d[1, ...] - avg_d[0, ...]  # (*sweep)
    std_d = np.sqrt(std_d[1, ...] ** 2 + std_d[0, ...] ** 2)  # (*sweep)

    return avg_d, std_d


def reset_signal2real(signals: np.ndarray) -> np.ndarray:
    return rotate2real(signals).real


def measure_reset_freq(soc, soccfg, cfg) -> Tuple[np.ndarray, np.ndarray]:
    cfg = deepcopy(cfg)  # prevent in-place modification

    test_reset_cfg = cfg["tested_reset"]
    reset_pulse = test_reset_cfg["pulse_cfg"]

    if test_reset_cfg["type"] != "pulse":
        raise ValueError("Reset pulse must be pulse")

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "freq")

    sweep_cfg = cfg["sweep"]["freq"]
    reset_pulse["freq"] = sweep2param("freq", sweep_cfg)

    fpts = sweep2array(sweep_cfg)  # predicted frequency points

    prog = ResetProbeProgram(soccfg, cfg)

    signals = sweep_hard_template(
        cfg,
        lambda _, cb: prog.acquire(soc, progress=True, callback=cb),
        LivePlotter1D("Frequency (MHz)", "Amplitude"),
        ticks=(fpts,),
        signal2real=reset_signal2real,
    )

    # get the actual frequency points
    fpts = prog.get_pulse_param("tested_reset_pulse", "freq", as_array=True)

    return fpts, signals


def measure_reset_time(soc, soccfg, cfg) -> Tuple[np.ndarray, np.ndarray]:
    cfg = deepcopy(cfg)  # prevent in-place modification

    reset_cfg = cfg["reset"]
    if reset_cfg["type"] != "pulse":
        raise ValueError("Reset pulse must be pulse")

    qub_pulse = cfg["qub_pulse"]
    reset_pulse = reset_cfg["pulse_cfg"]

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")

    # prepend ge sweep to inner loop
    cfg["sweep"] = {
        "ge": {"start": 0, "stop": qub_pulse["gain"], "expts": 2},
        "length": cfg["sweep"]["length"],
    }

    # set with / without pi gain for qubit pulse
    qub_pulse["gain"] = sweep2param("ge", cfg["sweep"]["ge"])
    reset_pulse["length"] = sweep2param("length", cfg["sweep"]["length"])

    lens = sweep2array(cfg["sweep"]["length"])  # predicted pulse gains

    prog = TwoToneProgram(soccfg, cfg)

    signals = sweep_hard_template(
        cfg,
        lambda _, cb: prog.acquire(soc, progress=True, callback=cb),
        LivePlotter1D("Length (us)", "Amplitude"),
        ticks=(lens,),
        result2signals=reset_result2signal,
    )

    # get the actual pulse length
    real_lens = prog.get_pulse_param("reset_pulse", "length", as_array=True)
    # TODO: better way to do this?
    real_lens += lens[0] - real_lens[0]  # add back the side length of the pulse

    return real_lens, signals  # lens


def measure_reset_amprabi(soc, soccfg, cfg) -> Tuple[np.ndarray, np.ndarray]:
    cfg = deepcopy(cfg)  # prevent in-place modification

    reset_cfg = cfg["tested_reset"]
    if reset_cfg["type"] != "pulse":
        raise ValueError("Reset pulse must be pulse")

    reset_pulse = reset_cfg["pulse_cfg"]
    init_pulse = cfg["init_pulse"]

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "gain")
    gain_sweep = cfg["sweep"]["gain"]

    cfg["sweep"] = {
        "w/o_reset": {"start": 0, "stop": 1.0, "expts": 2},
        "gain": gain_sweep,
    }

    init_pulse["gain"] = sweep2param("gain", gain_sweep)

    # TODO: better way to do this?
    reset_factor = sweep2param("w/o_reset", cfg["sweep"]["w/o_reset"])
    reset_pulse["gain"] = reset_factor * reset_pulse["gain"]
    reset_pulse["length"] = reset_factor * reset_pulse["length"] + 0.005
    if reset_pulse["style"] == "flat_top":  # prevent negative length in flat part
        reset_pulse["length"] += reset_pulse["raise_pulse"]["length"]

    pdrs = sweep2array(gain_sweep)  # predicted amplitudes

    prog = ResetProbeProgram(soccfg, cfg)

    signals = sweep_hard_template(
        cfg,
        lambda _, cb: prog.acquire(soc, progress=True, callback=cb),
        LivePlotter1D("Pulse gain", "Amplitude", num_lines=2),
        ticks=(pdrs,),
        signal2real=reset_signal2real,
    )

    pdrs = prog.get_pulse_param("init_pulse", "gain", as_array=True)

    return pdrs, signals


def measure_mux_reset_freq(
    soc, soccfg, cfg
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cfg = deepcopy(cfg)  # prevent in-place modification

    reset_cfg = cfg["tested_reset"]

    if reset_cfg["type"] != "two_pulse":
        raise ValueError("Reset pulse must be two pulse")

    reset_pulse1 = reset_cfg["pulse1_cfg"]
    reset_pulse2 = reset_cfg["pulse2_cfg"]

    if reset_cfg["type"] != "two_pulse":
        raise ValueError("Reset pulse must be two pulse")

    # force freq1 to be the outer loop
    cfg["sweep"] = {"freq1": cfg["sweep"]["freq1"], "freq2": cfg["sweep"]["freq2"]}

    reset_pulse1["freq"] = sweep2param("freq1", cfg["sweep"]["freq1"])
    reset_pulse2["freq"] = sweep2param("freq2", cfg["sweep"]["freq2"])

    fpts1 = sweep2array(cfg["sweep"]["freq1"])  # predicted frequency points
    fpts2 = sweep2array(cfg["sweep"]["freq2"])  # predicted frequency points

    prog = ResetProbeProgram(soccfg, cfg)

    signals = sweep_hard_template(
        cfg,
        lambda _, cb: prog.acquire(soc, progress=True, callback=cb),
        LivePlotter2D("Frequency1 (MHz)", "Frequency2 (MHz)"),
        ticks=(fpts1, fpts2),
        signal2real=reset_signal2real,
    )

    # get the actual frequency points
    fpts1 = prog.get_pulse_param("tested_reset_pulse1", "freq", as_array=True)
    fpts2 = prog.get_pulse_param("tested_reset_pulse2", "freq", as_array=True)

    return fpts1, fpts2, signals


def measure_mux_reset_pdr(
    soc, soccfg, cfg
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cfg = deepcopy(cfg)  # prevent in-place modification

    reset_cfg = cfg["tested_reset"]

    if reset_cfg["type"] != "two_pulse":
        raise ValueError("Reset pulse must be two pulse")

    reset_test_pulse1 = reset_cfg["pulse_cfg1"]
    reset_test_pulse2 = reset_cfg["pulse_cfg2"]

    # force gain1 to be the outer loop
    cfg["sweep"] = {
        "gain1": cfg["sweep"]["gain1"],
        "gain2": cfg["sweep"]["gain2"],
    }

    reset_test_pulse1["gain"] = sweep2param("gain1", cfg["sweep"]["gain1"])
    reset_test_pulse2["gain"] = sweep2param("gain2", cfg["sweep"]["gain2"])

    pdrs1 = sweep2array(cfg["sweep"]["gain1"])  # predicted amplitudes
    pdrs2 = sweep2array(cfg["sweep"]["gain2"])  # predicted amplitudes

    prog = ResetProbeProgram(soccfg, cfg)

    signals = sweep_hard_template(
        cfg,
        lambda _, cb: prog.acquire(soc, progress=True, callback=cb),
        LivePlotter2D("Gain1", "Gain2"),
        ticks=(pdrs1, pdrs2),
        signal2real=reset_signal2real,
    )

    # get the actual frequency points
    pdrs1 = prog.get_pulse_param("tested_reset_pulse1", "gain", as_array=True)
    pdrs2 = prog.get_pulse_param("tested_reset_pulse2", "gain", as_array=True)

    return pdrs1, pdrs2, signals


def measure_mux_reset_time(soc, soccfg, cfg) -> Tuple[np.ndarray, np.ndarray]:
    cfg = deepcopy(cfg)  # prevent in-place modification

    reset_cfg = cfg["reset"]
    if reset_cfg["type"] != "two_pulse":
        raise ValueError("Reset pulse must be two pulse")

    qub_pulse = cfg["qub_pulse"]
    reset_pulse1 = reset_cfg["pulse_cfg1"]
    reset_pulse2 = reset_cfg["pulse_cfg2"]

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")

    # prepend ge sweep to inner loop
    cfg["sweep"] = {
        "ge": {"start": 0, "stop": qub_pulse["gain"], "expts": 2},
        "length": cfg["sweep"]["length"],
    }

    # set with / without pi gain for qubit pulse
    qub_pulse["gain"] = sweep2param("ge", cfg["sweep"]["ge"])

    len_params = sweep2param("length", cfg["sweep"]["length"])
    reset_pulse1["length"] = len_params
    reset_pulse2["length"] = len_params  # TODO: support not equal length pulses

    lens = sweep2array(cfg["sweep"]["length"])  # predicted pulse gains

    prog = TwoToneProgram(soccfg, cfg)

    signals = sweep_hard_template(
        cfg,
        lambda _, cb: prog.acquire(soc, progress=True, callback=cb),
        LivePlotter1D("Length (us)", "Amplitude"),
        ticks=(lens,),
        result2signals=reset_result2signal,
    )

    # get the actual pulse length
    real_lens = prog.get_pulse_param("reset_pulse1", "length", as_array=True)
    # TODO: better way to do this?
    real_lens += lens[0] - real_lens[0]  # add back the side length of the pulse

    return real_lens, signals  # lens


def measure_mux_reset_amprabi(soc, soccfg, cfg) -> Tuple[np.ndarray, np.ndarray]:
    cfg = deepcopy(cfg)  # prevent in-place modification

    reset_cfg = cfg["tested_reset"]

    if reset_cfg["type"] != "two_pulse":
        raise ValueError("Reset pulse must be two pulse")

    reset_pulse1 = reset_cfg["pulse_cfg1"]
    reset_pulse2 = reset_cfg["pulse_cfg2"]

    init_pulse = cfg["init_pulse"]

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "gain")
    gain_sweep = cfg["sweep"]["gain"]

    cfg["sweep"] = {
        "w/o_reset": {"start": 0, "stop": 1.0, "expts": 2},
        "gain": gain_sweep,
    }

    init_pulse["gain"] = sweep2param("gain", gain_sweep)

    # TODO: better way to do this?
    reset_factor = sweep2param("w/o_reset", cfg["sweep"]["w/o_reset"])
    reset_pulse1["gain"] = reset_factor * reset_pulse1["gain"]
    reset_pulse2["gain"] = reset_factor * reset_pulse2["gain"]
    reset_pulse1["length"] = reset_factor * reset_pulse1["length"] + 0.005
    reset_pulse2["length"] = reset_factor * reset_pulse2["length"] + 0.005
    if reset_pulse1["style"] == "flat_top":
        reset_pulse1["length"] += reset_pulse1["raise_pulse"]["length"]
    if reset_pulse2["style"] == "flat_top":
        reset_pulse2["length"] += reset_pulse2["raise_pulse"]["length"]

    pdrs = sweep2array(gain_sweep)  # predicted amplitudes

    prog = ResetProbeProgram(soccfg, cfg)

    signals = sweep_hard_template(
        cfg,
        lambda _, cb: prog.acquire(soc, progress=True, callback=cb),
        LivePlotter1D("Pulse gain", "Amplitude", num_lines=2),
        ticks=(pdrs,),
        signal2real=reset_signal2real,
    )

    pdrs = prog.get_pulse_param("init_pulse", "gain", as_array=True)

    return pdrs, signals
