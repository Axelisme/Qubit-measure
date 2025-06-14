from typing import Optional, Tuple

import numpy as np
from numpy import ndarray
from zcu_tools import make_cfg
from zcu_tools.liveplot.jupyter import LivePlotter1D, LivePlotter2D
from zcu_tools.program.v2 import TwoToneProgram

from ...tools import format_sweep1D, map2adcfreq, sweep2array, sweep2param
from ..template import sweep1D_soft_template, sweep_hard_template


def calc_snr(avg_d: ndarray, std_d: ndarray) -> ndarray:
    contrast = avg_d[1, ...] - avg_d[0, ...]  # (*sweep)
    noise2_i = np.sum(std_d.real**2, axis=0)  # (*sweep)
    noise2_q = np.sum(std_d.imag**2, axis=0)  # (*sweep)
    noise = np.sqrt(noise2_i * contrast.real**2 + noise2_q * contrast.imag**2) / np.abs(
        contrast
    )

    return contrast / noise


def ge_result2signals(avg_d: list, std_d: list) -> Tuple[ndarray, Optional[ndarray]]:
    avg_d = avg_d[0][0].dot([1, 1j])  # (ge, *sweep)
    std_d = std_d[0][0].dot([1, 1j])  # (ge, *sweep)

    return calc_snr(avg_d, std_d), None


def measure_ge_freq_dep(soc, soccfg, cfg) -> Tuple[ndarray, ndarray]:
    cfg = make_cfg(cfg)  # prevent in-place modification

    res_pulse = cfg["dac"]["res_pulse"]
    qub_pulse = cfg["dac"]["qub_pulse"]

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "freq")

    # prepend ge sweep to inner loop
    cfg["sweep"] = {
        "ge": {"start": 0, "stop": qub_pulse["gain"], "expts": 2},
        "freq": cfg["sweep"]["freq"],
    }

    # set with / without pi gain for qubit pulse
    qub_pulse["gain"] = sweep2param("ge", cfg["sweep"]["ge"])

    res_pulse["freq"] = sweep2param("freq", cfg["sweep"]["freq"])

    fpts = sweep2array(cfg["sweep"]["freq"])  # predicted frequency points
    fpts = map2adcfreq(soc, fpts, res_pulse["ch"], cfg["adc"]["chs"][0])

    prog: Optional[TwoToneProgram] = None

    def measure_fn(cfg, callback) -> Tuple[list, list]:
        nonlocal prog
        prog = TwoToneProgram(soccfg, cfg)
        return prog.acquire(soc, progress=True, callback=callback)

    snrs = sweep_hard_template(
        cfg,
        measure_fn,
        LivePlotter1D("Frequency (MHz)", "SNR"),
        ticks=(fpts,),
        result2signals=ge_result2signals,
    )

    # get the actual pulse gains and frequency points
    fpts = prog.get_pulse_param("res_pulse", "freq", as_array=True)

    return fpts, snrs  # fpts


def measure_ge_pdr_dep(soc, soccfg, cfg) -> Tuple[ndarray, ndarray]:
    cfg = make_cfg(cfg)  # prevent in-place modification

    res_pulse = cfg["dac"]["res_pulse"]
    qub_pulse = cfg["dac"]["qub_pulse"]

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "gain")

    # prepend ge sweep to inner loop
    cfg["sweep"] = {
        "ge": {"start": 0, "stop": qub_pulse["gain"], "expts": 2},
        "gain": cfg["sweep"]["gain"],
    }

    # set with / without pi gain for qubit pulse
    qub_pulse["gain"] = sweep2param("ge", cfg["sweep"]["ge"])
    res_pulse["gain"] = sweep2param("gain", cfg["sweep"]["gain"])

    pdrs = sweep2array(cfg["sweep"]["gain"])  # predicted pulse gains

    prog: Optional[TwoToneProgram] = None

    def measure_fn(cfg, callback) -> Tuple[list, list]:
        nonlocal prog
        prog = TwoToneProgram(soccfg, cfg)
        return prog.acquire(soc, progress=True, callback=callback)

    snrs = sweep_hard_template(
        cfg,
        measure_fn,
        LivePlotter1D("Power (a.u)", "SNR"),
        ticks=(pdrs,),
        result2signals=ge_result2signals,
    )

    # get the actual pulse gains and frequency points
    pdrs = prog.get_pulse_param("res_pulse", "gain", as_array=True)

    return pdrs, snrs  # pdrs


def measure_ge_pdr_dep2D(soc, soccfg, cfg) -> Tuple[ndarray, ndarray, ndarray]:
    cfg = make_cfg(cfg)  # prevent in-place modification

    res_pulse = cfg["dac"]["res_pulse"]
    qub_pulse = cfg["dac"]["qub_pulse"]

    # prepend ge sweep to inner loop
    cfg["sweep"] = {
        "ge": {"start": 0, "stop": qub_pulse["gain"], "expts": 2},
        "gain": cfg["sweep"]["gain"],
        "freq": cfg["sweep"]["freq"],
    }

    # set with / without pi gain for qubit pulse
    qub_pulse["gain"] = sweep2param("ge", cfg["sweep"]["ge"])

    res_pulse["gain"] = sweep2param("gain", cfg["sweep"]["gain"])
    res_pulse["freq"] = sweep2param("freq", cfg["sweep"]["freq"])

    pdrs = sweep2array(cfg["sweep"]["gain"])  # predicted pulse gains
    fpts = sweep2array(cfg["sweep"]["freq"])  # predicted frequency points
    fpts = map2adcfreq(soc, fpts, res_pulse["ch"], cfg["adc"]["chs"][0])

    prog: Optional[TwoToneProgram] = None

    def measure_fn(cfg, callback) -> Tuple[list, list]:
        nonlocal prog
        prog = TwoToneProgram(soccfg, cfg)
        return prog.acquire(soc, progress=True, callback=callback)

    snr2D = sweep_hard_template(
        cfg,
        measure_fn,
        LivePlotter2D("Readout Gain", "Frequency (MHz)"),
        ticks=(pdrs, fpts),
        result2signals=ge_result2signals,
    )

    # get the actual pulse gains and frequency points
    pdrs = prog.get_pulse_param("res_pulse", "gain", as_array=True)
    fpts = prog.get_pulse_param("res_pulse", "freq", as_array=True)

    return pdrs, fpts, snr2D  # (pdrs, fpts)


def measure_ge_ro_dep(soc, soccfg, cfg) -> Tuple[ndarray, ndarray]:
    cfg = make_cfg(cfg)  # prevent in-place modification

    qub_pulse = cfg["dac"]["qub_pulse"]

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")
    length_sweep = cfg["sweep"]["length"]

    # replace ge sweep to sweep, and use soft loop for length
    cfg["sweep"] = {"ge": {"start": 0, "stop": qub_pulse["gain"], "expts": 2}}

    # set with / without pi length for qubit pulse
    qub_pulse["gain"] = sweep2param("ge", cfg["sweep"]["ge"])

    lens = sweep2array(length_sweep)  # predicted readout lengths

    cfg["adc"]["ro_length"] = lens[0]
    cfg["dac"]["res_pulse"]["length"] = lens.max() + cfg["adc"]["trig_offset"] + 0.1

    def updateCfg(cfg, _, ro_len):
        cfg["adc"]["ro_length"] = ro_len

    prog: Optional[TwoToneProgram] = None

    def measure_fn(cfg, callback) -> Tuple[list, list]:
        nonlocal prog
        prog = TwoToneProgram(soccfg, cfg)
        return prog.acquire(soc, progress=False, callback=callback)

    snrs = sweep1D_soft_template(
        cfg,
        measure_fn,
        LivePlotter1D("Readout Length (us)", "SNR"),
        xs=lens,
        progress=True,
        updateCfg=updateCfg,
        result2signals=ge_result2signals,
    )

    return lens, snrs
