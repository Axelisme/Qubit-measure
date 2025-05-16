from copy import deepcopy

import numpy as np
from zcu_tools.program.v1 import TwoToneProgram
from zcu_tools.schedule.flux import set_flux

from ...tools import check_time_sweep, map2adcfreq, sweep2array
from ..template import (
    sweep1D_soft_template,
    sweep2D_maximize_template,
    sweep2D_soft_soft_template,
)


def measure_one(soc, soccfg, cfg):
    cfg = deepcopy(cfg)  # prevent in-place modification

    qub_pulse = cfg["dac"]["qub_pulse"]
    pi_gain = qub_pulse["gain"]

    qub_pulse["gain"] = 0
    prog = TwoToneProgram(soccfg, cfg)
    avggi, avggq, stdgi, stdgq = prog.acquire(soc, progress=False)

    qub_pulse["gain"] = pi_gain
    prog = TwoToneProgram(soccfg, cfg)
    avgei, avgeq, stdei, stdeq = prog.acquire(soc, progress=False)

    dist_i = avgei[0][0] - avggi[0][0]
    dist_q = avgeq[0][0] - avggq[0][0]
    contrast = dist_i + 1j * dist_q
    noise2_i = stdgi[0][0] ** 2 + stdei[0][0] ** 2
    noise2_q = stdgq[0][0] ** 2 + stdeq[0][0] ** 2
    noise = np.sqrt(noise2_i * dist_i**2 + noise2_q * dist_q**2) / np.abs(contrast)

    return contrast / noise, None


def measure_ge_pdr_dep(soc, soccfg, cfg):
    cfg = deepcopy(cfg)  # prevent in-place modification

    res_pulse = cfg["dac"]["res_pulse"]

    freq_cfg = cfg["sweep"]["freq"]
    pdr_cfg = cfg["sweep"]["gain"]
    fpts = sweep2array(freq_cfg, allow_array=True)
    fpts = map2adcfreq(soccfg, fpts, res_pulse["ch"], cfg["adc"]["chs"][0])
    pdrs = sweep2array(pdr_cfg, allow_array=True)

    del cfg["sweep"]  # remove sweep from cfg

    def x_updateCfg(cfg, _, pdr):
        cfg["dac"]["res_pulse"]["gain"] = pdr

    def y_updateCfg(cfg, _, fpt):
        cfg["dac"]["res_pulse"]["freq"] = fpt

    pdrs, fpts, snr2D = sweep2D_soft_soft_template(
        soc,
        soccfg,
        cfg,
        measure_one,
        xs=pdrs,
        ys=fpts,
        x_updateCfg=x_updateCfg,
        y_updateCfg=y_updateCfg,
        xlabel="Power (a.u)",
        ylabel="Frequency (MHz)",
    )

    return pdrs, fpts, snr2D


def measure_ge_pdr_dep_auto(soc, soccfg, cfg, method="Nelder-Mead"):
    cfg = deepcopy(cfg)  # prevent in-place modification

    res_pulse = cfg["dac"]["res_pulse"]

    pdrs = sweep2array(cfg["sweep"]["gain"])  # predicted pulse gains
    fpts = sweep2array(cfg["sweep"]["freq"])  # predicted frequency points
    fpts = map2adcfreq(soc, fpts, res_pulse["ch"], cfg["adc"]["chs"][0])

    del cfg["sweep"]  # remove sweep from cfg

    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

    def measure_signals(pdr, fpt):
        cfg["dac"]["res_pulse"]["gain"] = int(pdr)
        cfg["dac"]["res_pulse"]["freq"] = float(fpt)
        return measure_one(soc, soccfg, cfg)

    return sweep2D_maximize_template(
        measure_signals,
        xs=pdrs,
        ys=fpts,
        xlabel="Power (a.u)",
        ylabel="Frequency (MHz)",
        method=method,
    )


def measure_ge_ro_dep(soc, soccfg, cfg):
    cfg = deepcopy(cfg)  # prevent in-place modification

    ro_lens = sweep2array(cfg["sweep"], allow_array=True)
    check_time_sweep(soccfg, ro_lens, ro_ch=cfg["adc"]["chs"][0])

    del cfg["sweep"]  # remove sweep from cfg

    cfg["dac"]["res_pulse"]["length"] = cfg["adc"]["trig_offset"] + ro_lens.max() + 0.1

    def updateCfg(cfg, _, ro_len):
        cfg["adc"]["ro_length"] = ro_len

    ro_lens, snrs = sweep1D_soft_template(
        soc,
        soccfg,
        cfg,
        measure_one,
        xs=ro_lens,
        updateCfg=updateCfg,
        xlabel="Readout Length (us)",
        ylabel="SNR",
    )

    return ro_lens, snrs


def measure_ge_trig_dep(soc, soccfg, cfg):
    cfg = deepcopy(cfg)  # prevent in-place modification

    offsets = sweep2array(cfg["sweep"], allow_array=True)
    check_time_sweep(soccfg, offsets, ro_ch=cfg["adc"]["chs"][0])

    del cfg["sweep"]  # remove sweep from cfg

    cfg["dac"]["res_pulse"]["length"] = offsets.max() + cfg["adc"]["ro_length"] + 0.1

    def updateCfg(cfg, _, offset):
        cfg["adc"]["trig_offset"] = offset

    offsets, snrs = sweep1D_soft_template(
        soc,
        soccfg,
        cfg,
        measure_one,
        xs=offsets,
        updateCfg=updateCfg,
        xlabel="Readout offset (us)",
        ylabel="SNR",
    )

    return offsets, snrs
