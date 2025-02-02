from copy import deepcopy

import numpy as np
from tqdm.auto import tqdm

from zcu_tools import make_cfg
from zcu_tools.program import TwoToneProgram

from ..tools import map2adcfreq, sweep2array, check_time_sweep
from ..flux import set_flux
from ..instant_show import (
    clear_show,
    init_show,
    init_show2d,
    update_show,
    update_show2d,
)


def measure_one(soc, soccfg, cfg):
    qub_pulse = cfg["dac"]["qub_pulse"]
    pi_gain = qub_pulse["gain"]

    qub_pulse["gain"] = 0
    prog = TwoToneProgram(soccfg, make_cfg(cfg))
    avggi, avggq, stdgi, stdgq = prog.acquire(soc, progress=False, ret_std=True)

    qub_pulse["gain"] = pi_gain
    prog = TwoToneProgram(soccfg, make_cfg(cfg))
    avgei, avgeq, stdei, stdeq = prog.acquire(soc, progress=False, ret_std=True)

    dist_i = avgei[0][0] - avggi[0][0]
    dist_q = avgeq[0][0] - avggq[0][0]
    contrast = dist_i + 1j * dist_q
    noise2_i = stdgi[0][0] ** 2 + stdei[0][0] ** 2
    noise2_q = stdgq[0][0] ** 2 + stdeq[0][0] ** 2
    noise = np.sqrt(noise2_i * dist_i**2 + noise2_q * dist_q**2) / np.abs(contrast)

    return contrast / noise


def measure_ge_pdr_dep(
    soc,
    soccfg,
    cfg,
    instant_show=False,
):
    cfg = deepcopy(cfg)  # prevent in-place modification

    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

    res_pulse = cfg["dac"]["res_pulse"]

    freq_cfg = cfg["sweep"]["freq"]
    pdr_cfg = cfg["sweep"]["gain"]
    fpts = sweep2array(freq_cfg)
    fpts = map2adcfreq(soccfg, fpts, res_pulse["ch"], cfg["adc"]["chs"][0])
    pdrs = sweep2array(pdr_cfg)

    if instant_show:
        fig, ax, dh, im = init_show2d(fpts, pdrs, "Frequency (MHz)", "Power (a.u.)")

    snr2D = np.full((len(pdrs), len(fpts)), np.nan, dtype=np.complex128)
    try:
        pdr_tqdm = tqdm(pdrs, desc="Power", smoothing=0)
        freq_tqdm = tqdm(fpts, desc="Frequency", smoothing=0)

        for i, pdr in enumerate(pdr_tqdm):
            res_pulse["gain"] = pdr

            freq_tqdm.reset()
            freq_tqdm.refresh()
            for j, fpt in enumerate(fpts):
                res_pulse["freq"] = fpt

                snr2D[i, j] = measure_one(soc, soccfg, cfg)
                freq_tqdm.update()

            if instant_show:
                ax.set_title(f"Maximum SNR: {np.nanmax(np.abs(snr2D)):.2f}")
                update_show2d(fig, ax, dh, im, np.abs(snr2D))

        if instant_show:
            clear_show()
    except Exception as e:
        print("Error during measurement:", e)

    return fpts, pdrs, snr2D  # (pdrs, freqs)


def measure_ge_ro_dep(soc, soccfg, cfg, instant_show=False):
    cfg = deepcopy(cfg)  # prevent in-place modification

    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

    res_pulse = cfg["dac"]["res_pulse"]

    ro_lens = sweep2array(cfg["sweep"])
    check_time_sweep(
        soccfg, ro_lens, gen_ch=res_pulse["ch"], ro_ch=cfg["adc"]["chs"][0]
    )

    trig_offset = cfg["adc"]["trig_offset"]

    show_period = int(len(ro_lens) / 10 + 0.99999)
    if instant_show:
        fig, ax, dh, curve = init_show(ro_lens, "Readout Length (us)", "SNR (a.u.)")

    snrs = np.full(len(ro_lens), np.nan, dtype=np.complex128)
    try:
        for i, ro_len in enumerate(tqdm(ro_lens, desc="ro length", smoothing=0)):
            cfg["adc"]["ro_length"] = ro_len
            res_pulse["length"] = trig_offset + ro_len + 1.0

            snrs[i] = measure_one(soc, soccfg, cfg)

            if instant_show and i % show_period == 0:
                update_show(fig, ax, dh, curve, np.abs(snrs))
        else:
            if instant_show:
                update_show(fig, ax, dh, curve, np.abs(snrs))

        if instant_show:
            clear_show()
    except Exception as e:
        print("Error during measurement:", e)

    return ro_lens, snrs


def measure_ge_trig_dep(soc, soccfg, cfg, instant_show=False):
    cfg = deepcopy(cfg)  # prevent in-place modification

    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

    res_pulse = cfg["dac"]["res_pulse"]

    offsets = sweep2array(cfg["sweep"])
    check_time_sweep(
        soccfg, offsets, gen_ch=res_pulse["ch"], ro_ch=cfg["adc"]["chs"][0]
    )
    ro_len = cfg["adc"]["ro_length"]
    orig_offset = cfg["adc"]["trig_offset"]

    show_period = int(len(offsets) / 10 + 0.99999)
    if instant_show:
        fig, ax, dh, curve = init_show(offsets, "Trigger Offset (us)", "SNR (a.u.)")

    snrs = np.full(len(offsets), np.nan, dtype=np.complex128)
    try:
        for i, offset in enumerate(tqdm(offsets, desc="trig offset", smoothing=0)):
            cfg["adc"]["trig_offset"] = offset
            cfg["adc"]["ro_length"] = ro_len + offset - orig_offset

            snrs[i] = measure_one(soc, soccfg, cfg)

            if instant_show and i % show_period == 0:
                update_show(fig, ax, dh, curve, np.abs(snrs))
        else:
            if instant_show:
                update_show(fig, ax, dh, curve, np.abs(snrs))

        if instant_show:
            clear_show()
    except Exception as e:
        print("Error during measurement:", e)

    return offsets, snrs
