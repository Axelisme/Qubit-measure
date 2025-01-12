from copy import deepcopy

import numpy as np
from tqdm.auto import tqdm

from zcu_tools import make_cfg
from zcu_tools.program import TwoToneProgram

from ..flux import set_flux
from ..instant_show import clear_show, init_show2d, update_show2d


def measure_ge_contrast(
    soc,
    soccfg,
    cfg,
    instant_show=False,
    dynamic_reps=False,
    gain_ref=1000,
):
    cfg = deepcopy(cfg)  # prevent in-place modification

    set_flux(cfg["flux_dev"], cfg["flux"])

    freq_cfg = cfg["sweep"]["freq"]
    pdr_cfg = cfg["sweep"]["gain"]
    fpts = np.linspace(freq_cfg["start"], freq_cfg["stop"], freq_cfg["expts"])
    pdrs = np.arange(pdr_cfg["start"], pdr_cfg["stop"], pdr_cfg["step"])

    res_pulse = cfg["dac"]["res_pulse"]
    qub_pulse = cfg["dac"]["qub_pulse"]
    pi_gain = qub_pulse["gain"]

    reps_ref = cfg["reps"]

    if instant_show:
        fig, ax, dh, im = init_show2d(fpts, pdrs, "Frequency (MHz)", "Power (a.u.)")

    signals2D = np.full((len(pdrs), len(fpts)), np.nan, dtype=np.complex128)
    try:
        pdr_tqdm = tqdm(pdrs, desc="Power", smoothing=0)
        freq_tqdm = tqdm(fpts, desc="Frequency", smoothing=0)

        for i, pdr in enumerate(pdr_tqdm):
            res_pulse["gain"] = pdr

            if dynamic_reps:
                cfg["reps"] = int(reps_ref * gain_ref / pdr)
                if cfg["reps"] < 0.1 * reps_ref:
                    cfg["reps"] = int(0.1 * reps_ref + 0.99)
                elif cfg["reps"] > 10 * reps_ref:
                    cfg["reps"] = int(10 * reps_ref)

            freq_tqdm.reset()
            freq_tqdm.refresh()
            for j, fpt in enumerate(fpts):
                res_pulse["freq"] = fpt

                qub_pulse["gain"] = 0
                prog = TwoToneProgram(soccfg, make_cfg(cfg))
                avgi, avgq = prog.acquire(soc, progress=False)
                signals_g = avgi[0][0] + 1j * avgq[0][0]

                qub_pulse["gain"] = pi_gain
                prog = TwoToneProgram(soccfg, make_cfg(cfg))
                avgi, avgq = prog.acquire(soc, progress=False)
                signals_e = avgi[0][0] + 1j * avgq[0][0]

                signals2D[i, j] = signals_e - signals_g
                freq_tqdm.update()

            pdr_tqdm.update()

            if instant_show:
                update_show2d(fig, ax, dh, im, np.abs(signals2D))

        if instant_show:
            clear_show()
    except Exception as e:
        print("Error during measurement:", e)

    return fpts, pdrs, signals2D  # (pdrs, freqs)


def measure_ge_contrast2(
    soc, soccfg, cfg, instant_show=False, dynamic_reps=False, len_ref=1000
):
    cfg = deepcopy(cfg)  # prevent in-place modification
    if "ro_length" in cfg["adc"]:
        del cfg["adc"]["ro_length"]  # for auto setting

    set_flux(cfg["flux_dev"], cfg["flux"])

    res_cfg = cfg["sweep"]["res_length"]
    ro_cfg = cfg["sweep"]["ro_length"]
    res_lens = np.linspace(res_cfg["start"], res_cfg["stop"], res_cfg["expts"])
    ro_lens = np.arange(ro_cfg["start"], ro_cfg["stop"], ro_cfg["step"])

    res_pulse = cfg["dac"]["res_pulse"]
    qub_pulse = cfg["dac"]["qub_pulse"]
    reps_ref = cfg["reps"]
    pi_gain = qub_pulse["gain"]
    total_len = cfg["relax_delay"] + max(
        res_lens[-1], ro_lens[-1] + cfg["adc"]["trig_offset"]
    )

    if instant_show:
        fig, ax, dh, im = init_show2d(
            res_lens, ro_lens, "Pulse length (us)", "Readout length (us)"
        )

    signals2D = np.full((len(ro_lens), len(res_lens)), np.nan, dtype=np.complex128)
    try:
        ro_tqdm = tqdm(ro_lens, desc="ro_len", smoothing=0)
        res_tqdm = tqdm(res_lens, desc="res_len", smoothing=0)

        for i, ro_len in enumerate(ro_tqdm):
            res_pulse["ro_length"] = ro_len

            if dynamic_reps:
                cfg["reps"] = int(reps_ref * len_ref / ro_len)
                if cfg["reps"] < 0.1 * reps_ref:
                    cfg["reps"] = int(0.1 * reps_ref + 0.99)
                elif cfg["reps"] > 10 * reps_ref:
                    cfg["reps"] = int(10 * reps_ref)

            res_tqdm.reset()
            res_tqdm.refresh()
            for j, res_len in enumerate(res_lens):
                res_pulse["length"] = res_len

                cfg["relax_delay"] = total_len - max(
                    res_len, ro_len + cfg["adc"]["trig_offset"]
                )

                qub_pulse["gain"] = 0
                prog = TwoToneProgram(soccfg, make_cfg(cfg))
                avgi, avgq = prog.acquire(soc, progress=False)
                signals_g = avgi[0][0] + 1j * avgq[0][0]

                qub_pulse["gain"] = pi_gain
                prog = TwoToneProgram(soccfg, make_cfg(cfg))
                avgi, avgq = prog.acquire(soc, progress=False)
                signals_e = avgi[0][0] + 1j * avgq[0][0]

                signals2D[i, j] = signals_e - signals_g
                res_tqdm.update()

            ro_tqdm.update()

            if instant_show:
                update_show2d(fig, ax, dh, im, np.abs(signals2D))

        if instant_show:
            clear_show()
    except Exception as e:
        print("Error during measurement:", e)

    return res_lens, ro_lens, signals2D  # (res_lens, ro_lens)
