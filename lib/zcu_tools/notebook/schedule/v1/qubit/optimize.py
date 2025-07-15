from copy import deepcopy
from typing import Tuple

import numpy as np
from numpy import ndarray
from zcu_tools.liveplot.jupyter import LivePlotter1D, LivePlotter2DwithLine
from zcu_tools.program.v1 import TwoToneProgram

from ...tools import check_time_sweep, map2adcfreq, sweep2array
from ..template import sweep1D_soft_template, sweep2D_soft_soft_template


def measure_dist(soc, soccfg, cfg) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    cfg = deepcopy(cfg)  # prevent in-place modification

    qub_pulse = cfg["dac"]["qub_pulse"]
    pi_gain = qub_pulse["gain"]

    qub_pulse["gain"] = 0
    prog = TwoToneProgram(soccfg, cfg)
    avggi, avggq, stdgi, stdgq = prog.acquire(soc, progress=False)

    qub_pulse["gain"] = pi_gain
    prog = TwoToneProgram(soccfg, cfg)
    avgei, avgeq, stdei, stdeq = prog.acquire(soc, progress=False)

    dist_i = [avgei[i] - avggi[i] for i in range(len(avgei))]
    dist_q = [avgeq[i] - avggq[i] for i in range(len(avgeq))]
    noise_i = [np.sqrt(stdgi[i] ** 2 + stdei[i] ** 2) for i in range(len(stdgi))]
    noise_q = [np.sqrt(stdgq[i] ** 2 + stdeq[i] ** 2) for i in range(len(stdgq))]

    return dist_i, dist_q, noise_i, noise_q


def result2snr(
    dist_i: ndarray, dist_q: ndarray, noise_i: ndarray, noise_q: ndarray
) -> ndarray:
    contrast = dist_i + 1j * dist_q
    dist = np.abs(contrast)
    noise = np.sqrt((noise_i * dist_i) ** 2 + (noise_q * dist_q) ** 2) / dist

    return contrast / noise, np.zeros_like(dist)


def measure_ge_pdr_dep2D(soc, soccfg, cfg) -> Tuple[ndarray, ndarray, ndarray]:
    cfg = deepcopy(cfg)  # prevent in-place modification

    res_pulse = cfg["dac"]["res_pulse"]

    freq_cfg = cfg["sweep"]["freq"]
    pdr_cfg = cfg["sweep"]["gain"]
    fpts = sweep2array(freq_cfg, allow_array=True)
    fpts = map2adcfreq(soccfg, fpts, res_pulse["ch"], cfg["adc"]["chs"][0])
    pdrs = sweep2array(pdr_cfg, allow_array=True)

    del cfg["sweep"]  # remove sweep from cfg

    def x_updateCfg(cfg, _, pdr) -> None:
        cfg["dac"]["res_pulse"]["gain"] = pdr

    def y_updateCfg(cfg, _, fpt) -> None:
        cfg["dac"]["res_pulse"]["freq"] = fpt

    def measure_fn(cfg, _) -> Tuple[ndarray, ...]:
        return measure_dist(soc, soccfg, cfg)

    pdrs, fpts, snr2D = sweep2D_soft_soft_template(
        cfg,
        measure_fn,
        LivePlotter2DwithLine("Power (a.u)", "Frequency (MHz)"),
        xs=pdrs,
        ys=fpts,
        x_updateCfg=x_updateCfg,
        y_updateCfg=y_updateCfg,
        result2signals=result2snr,
    )

    return pdrs, fpts, snr2D


def measure_ge_ro_dep(soc, soccfg, cfg) -> Tuple[ndarray, ndarray]:
    cfg = deepcopy(cfg)  # prevent in-place modification

    ro_lens = sweep2array(cfg["sweep"], allow_array=True)
    check_time_sweep(soccfg, ro_lens, ro_ch=cfg["adc"]["chs"][0])

    del cfg["sweep"]  # remove sweep from cfg

    cfg["dac"]["res_pulse"]["length"] = cfg["adc"]["trig_offset"] + ro_lens.max() + 0.1

    def updateCfg(cfg, _, ro_len) -> None:
        cfg["adc"]["ro_length"] = ro_len

    def measure_fn(cfg, _) -> Tuple[ndarray, ...]:
        return measure_dist(soc, soccfg, cfg)

    ro_lens, snrs = sweep1D_soft_template(
        cfg,
        measure_fn,
        LivePlotter1D("Readout Length (us)", "SNR"),
        xs=ro_lens,
        updateCfg=updateCfg,
        result2signals=result2snr,
    )

    return ro_lens, snrs
