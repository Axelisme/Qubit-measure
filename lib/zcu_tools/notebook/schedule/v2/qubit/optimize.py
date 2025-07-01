from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy import ndarray
from zcu_tools.liveplot.jupyter import LivePlotter1D, LivePlotter2D
from zcu_tools.program.v2 import TwoToneProgram

from ...tools import format_sweep1D, map2adcfreq, sweep2array, sweep2param
from ..template import sweep1D_soft_template, sweep_hard_template


def calc_snr(avg_d: ndarray, std_d: ndarray) -> ndarray:
    # avg_d: (ge, *sweep)
    # std_d: (ge, *sweep)

    contrast = avg_d[1, ...] - avg_d[0, ...]  # (*sweep)
    noise2_i = np.sum(std_d.real**2, axis=0)  # (*sweep)
    noise2_q = np.sum(std_d.imag**2, axis=0)  # (*sweep)
    noise = np.sqrt(noise2_i * contrast.real**2 + noise2_q * contrast.imag**2) / np.abs(
        contrast
    )

    return contrast / noise


def ge_raw2signal(
    ir: int, avg_d: List[ndarray], std_d: Optional[List[ndarray]]
) -> np.ndarray:
    assert std_d is not None, "std_d should not be None"

    avg_s = avg_d[0][0].dot([1, 1j])  # (ge, *sweep)
    std_s = std_d[0][0].dot([1, 1j])  # (ge, *sweep)

    return calc_snr(avg_s, std_s)


def measure_ge_freq_dep(soc, soccfg, cfg) -> Tuple[ndarray, ndarray]:
    cfg = deepcopy(cfg)  # prevent in-place modification

    res_pulse = cfg["readout"]["pulse_cfg"]
    ro_cfg = cfg["readout"]["ro_cfg"]
    qub_pulse = cfg["qub_pulse"]

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
    fpts = map2adcfreq(soc, fpts, res_pulse["ch"], ro_cfg["ro_ch"])

    prog = TwoToneProgram(soccfg, cfg)

    def measure_fn(
        cfg: Dict[str, Any], cb: Optional[Callable[..., None]]
    ) -> np.ndarray:
        avg_d = prog.acquire(soc, progress=True, callback=cb, record_stderr=True)
        std_d = prog.get_stderr()
        assert std_d is not None, "stds should not be None"

        avg_s = avg_d[0][0].dot([1, 1j])  # (ge, *sweep)
        std_s = std_d[0][0].dot([1, 1j])  # (ge, *sweep)

        return calc_snr(avg_s, std_s)

    snrs = sweep_hard_template(
        cfg,
        measure_fn,
        LivePlotter1D("Frequency (MHz)", "SNR"),
        ticks=(fpts,),
        raw2signal=ge_raw2signal,
    )

    # get the actual pulse gains and frequency points
    fpts = prog.get_pulse_param("readout_pulse", "freq", as_array=True)
    assert isinstance(fpts, np.ndarray), "fpts should be an array"

    return fpts, snrs  # fpts


def measure_ge_pdr_dep(soc, soccfg, cfg) -> Tuple[ndarray, ndarray]:
    cfg = deepcopy(cfg)  # prevent in-place modification

    res_pulse = cfg["readout"]["pulse_cfg"]
    qub_pulse = cfg["qub_pulse"]

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

    prog = TwoToneProgram(soccfg, cfg)

    def measure_fn(
        cfg: Dict[str, Any], cb: Optional[Callable[..., None]]
    ) -> np.ndarray:
        avg_d = prog.acquire(soc, progress=True, callback=cb, record_stderr=True)
        std_d = prog.get_stderr()
        assert std_d is not None, "stds should not be None"

        avg_s = avg_d[0][0].dot([1, 1j])  # (ge, *sweep)
        std_s = std_d[0][0].dot([1, 1j])  # (ge, *sweep)

        return calc_snr(avg_s, std_s)

    snrs = sweep_hard_template(
        cfg,
        measure_fn,
        LivePlotter1D("Power (a.u)", "SNR"),
        ticks=(pdrs,),
        raw2signal=ge_raw2signal,
    )

    # get the actual pulse gains and frequency points
    pdrs = prog.get_pulse_param("readout_pulse", "gain", as_array=True)
    assert isinstance(pdrs, np.ndarray), "pdrs should be an array"

    return pdrs, snrs  # pdrs


def measure_ge_pdr_dep2D(soc, soccfg, cfg) -> Tuple[ndarray, ndarray, ndarray]:
    cfg = deepcopy(cfg)  # prevent in-place modification

    res_pulse = cfg["readout"]["pulse_cfg"]
    ro_cfg = cfg["readout"]["ro_cfg"]
    qub_pulse = cfg["qub_pulse"]

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
    fpts = map2adcfreq(soc, fpts, res_pulse["ch"], ro_cfg["ro_ch"])

    prog = TwoToneProgram(soccfg, cfg)

    def measure_fn(
        cfg: Dict[str, Any], cb: Optional[Callable[..., None]]
    ) -> np.ndarray:
        avg_d = prog.acquire(soc, progress=True, callback=cb, record_stderr=True)
        std_d = prog.get_stderr()
        assert std_d is not None, "stds should not be None"

        avg_s = avg_d[0][0].dot([1, 1j])  # (ge, *sweep)
        std_s = std_d[0][0].dot([1, 1j])  # (ge, *sweep)

        return calc_snr(avg_s, std_s)

    snr2D = sweep_hard_template(
        cfg,
        measure_fn,
        LivePlotter2D("Readout Gain", "Frequency (MHz)"),
        ticks=(pdrs, fpts),
        raw2signal=ge_raw2signal,
    )

    # get the actual pulse gains and frequency points
    pdrs = prog.get_pulse_param("readout_pulse", "gain", as_array=True)
    fpts = prog.get_pulse_param("readout_pulse", "freq", as_array=True)
    assert isinstance(pdrs, np.ndarray), "pdrs should be an array"
    assert isinstance(fpts, np.ndarray), "fpts should be an array"

    return pdrs, fpts, snr2D  # (pdrs, fpts)


def measure_ge_ro_dep(soc, soccfg, cfg) -> Tuple[ndarray, ndarray]:
    cfg = deepcopy(cfg)  # prevent in-place modification

    res_pulse = cfg["readout"]["pulse_cfg"]
    ro_cfg = cfg["readout"]["ro_cfg"]
    qub_pulse = cfg["qub_pulse"]

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")
    length_sweep = cfg["sweep"]["length"]

    # replace ge sweep to sweep, and use soft loop for length
    cfg["sweep"] = {"ge": {"start": 0, "stop": qub_pulse["gain"], "expts": 2}}

    # set with / without pi length for qubit pulse
    qub_pulse["gain"] = sweep2param("ge", cfg["sweep"]["ge"])

    lens = sweep2array(length_sweep)  # predicted readout lengths

    ro_cfg["ro_length"] = lens[0]
    res_pulse["length"] = lens.max() + ro_cfg["trig_offset"] + 0.1

    def updateCfg(cfg, _, ro_len) -> None:
        cfg["readout"]["ro_cfg"]["ro_length"] = ro_len

    def measure_fn(
        cfg: Dict[str, Any], cb: Optional[Callable[..., None]]
    ) -> np.ndarray:
        prog = TwoToneProgram(soccfg, cfg)

        avg_d = prog.acquire(soc, progress=False, callback=cb, record_stderr=True)
        std_d = prog.get_stderr()
        assert std_d is not None, "stds should not be None"

        avg_s = avg_d[0][0].dot([1, 1j])  # (ge, *sweep)
        std_s = std_d[0][0].dot([1, 1j])  # (ge, *sweep)

        return calc_snr(avg_s, std_s)

    snrs = sweep1D_soft_template(
        cfg,
        measure_fn,
        LivePlotter1D("Readout Length (us)", "SNR"),
        xs=lens,
        progress=True,
        updateCfg=updateCfg,
    )

    return lens, snrs
