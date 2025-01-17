from copy import deepcopy

import numpy as np
from tqdm.auto import tqdm

from zcu_tools import make_cfg
from zcu_tools.program import TwoToneProgram

from ..flux import set_flux
from ..instant_show import clear_show, init_show2d, update_show2d


def measure_one(soc, soccfg, cfg):
    qub_pulse = cfg["dac"]["qub_pulse"]
    pi_gain = qub_pulse["gain"]

    qub_pulse["gain"] = 0
    prog = TwoToneProgram(soccfg, make_cfg(cfg))
    avggi, avggq, stdgi, stdgq = prog.acquire(soc, progress=False, ret_std=True)
    noise2_g = stdgi[0][0] ** 2 + stdgq[0][0] ** 2

    qub_pulse["gain"] = pi_gain
    prog = TwoToneProgram(soccfg, make_cfg(cfg))
    avgei, avgeq, stdei, stdeq = prog.acquire(soc, progress=False, ret_std=True)
    noise2_e = stdei[0][0] ** 2 + stdeq[0][0] ** 2

    # snr2D[i, j] = signals_e - signals_g
    dist_i = avgei[0][0] - avggi[0][0]
    dist_q = avgeq[0][0] - avggq[0][0]
    contrast = dist_i + 1j * dist_q
    noise = np.sqrt(noise2_g * dist_i**2 + noise2_e * dist_q**2) / np.abs(contrast)

    return contrast / noise


def measure_ge_pdr_dep(
    soc,
    soccfg,
    cfg,
    instant_show=False,
):
    cfg = deepcopy(cfg)  # prevent in-place modification

    set_flux(cfg["flux_dev"], cfg["flux"])

    freq_cfg = cfg["sweep"]["freq"]
    pdr_cfg = cfg["sweep"]["gain"]
    fpts = np.linspace(freq_cfg["start"], freq_cfg["stop"], freq_cfg["expts"])
    pdrs = np.arange(pdr_cfg["start"], pdr_cfg["stop"], pdr_cfg["step"])

    res_pulse = cfg["dac"]["res_pulse"]

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
                update_show2d(fig, ax, dh, im, np.abs(snr2D))

        if instant_show:
            clear_show()
    except Exception as e:
        print("Error during measurement:", e)

    return fpts, pdrs, snr2D  # (pdrs, freqs)


def measure_ge_ro_dep(soc, soccfg, cfg, instant_show=False):
    cfg = deepcopy(cfg)  # prevent in-place modification

    set_flux(cfg["flux_dev"], cfg["flux"])

    trig_cfg = cfg["sweep"]["trig_offset"]
    ro_cfg = cfg["sweep"]["ro_length"]
    offsets = np.linspace(trig_cfg["start"], trig_cfg["stop"], trig_cfg["expts"])
    ro_lens = np.linspace(ro_cfg["start"], ro_cfg["stop"], ro_cfg["expts"])

    pulse_len = cfg["dac"]["qub_pulse"]["length"]
    total_len = cfg["relax_delay"] + max(pulse_len, ro_lens.max() + offsets.max())

    if instant_show:
        fig, ax, dh, im = init_show2d(
            offsets, ro_lens, "Trigger offset (us)", "Readout length (us)"
        )

    snr2D = np.full((len(ro_lens), len(offsets)), np.nan, dtype=np.complex128)
    try:
        ro_tqdm = tqdm(ro_lens, desc="ro length", smoothing=0)
        trig_tqdm = tqdm(offsets, desc="trig offset", smoothing=0)

        for i, ro_len in enumerate(ro_tqdm):
            cfg["adc"]["ro_length"] = ro_len

            trig_tqdm.reset()
            trig_tqdm.refresh()
            for j, offset in enumerate(offsets):
                cfg["adc"]["trig_offset"] = offset

                cfg["relax_delay"] = total_len - max(pulse_len, ro_len + offset)

                snr2D[i, j] = measure_one(soc, soccfg, cfg)
                trig_tqdm.update()

            if instant_show:
                update_show2d(fig, ax, dh, im, np.abs(snr2D))

        if instant_show:
            clear_show()
    except Exception as e:
        print("Error during measurement:", e)

    return offsets, ro_lens, snr2D  # (offsets, ro_lens)
