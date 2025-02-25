from copy import deepcopy

import numpy as np
from tqdm.auto import tqdm

from zcu_tools import make_cfg
from zcu_tools.program.v1 import TwoToneProgram
from zcu_tools.schedule.flux import set_flux
from zcu_tools.schedule.instant_show import InstantShow
from zcu_tools.schedule.tools import check_time_sweep, map2adcfreq, sweep2array


def measure_one(soc, soccfg, cfg):
    qub_pulse = cfg["dac"]["qub_pulse"]
    pi_gain = qub_pulse["gain"]

    qub_pulse["gain"] = 0
    prog = TwoToneProgram(soccfg, make_cfg(cfg))
    avggi, avggq, stdgi, stdgq = prog.acquire(soc, progress=False, ret_std=True)  # type: ignore

    qub_pulse["gain"] = pi_gain
    prog = TwoToneProgram(soccfg, make_cfg(cfg))
    avgei, avgeq, stdei, stdeq = prog.acquire(soc, progress=False, ret_std=True)  # type: ignore

    dist_i = avgei[0][0] - avggi[0][0]  # type: ignore
    dist_q = avgeq[0][0] - avggq[0][0]  # type: ignore
    contrast = dist_i + 1j * dist_q
    noise2_i = stdgi[0][0] ** 2 + stdei[0][0] ** 2
    noise2_q = stdgq[0][0] ** 2 + stdeq[0][0] ** 2
    noise = np.sqrt(noise2_i * dist_i**2 + noise2_q * dist_q**2) / np.abs(contrast)

    return contrast / noise


def measure_ge_pdr_dep(soc, soccfg, cfg, instant_show=False):
    cfg = deepcopy(cfg)  # prevent in-place modification

    res_pulse = cfg["dac"]["res_pulse"]

    freq_cfg = cfg["sweep"]["freq"]
    pdr_cfg = cfg["sweep"]["gain"]
    fpts = sweep2array(freq_cfg, allow_array=True)
    fpts = map2adcfreq(soccfg, fpts, res_pulse["ch"], cfg["adc"]["chs"][0])
    pdrs = sweep2array(pdr_cfg, allow_array=True)

    del cfg["sweep"]  # remove sweep from cfg

    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

    if instant_show:
        viewer = InstantShow(
            fpts, pdrs, x_label="Frequency (MHz)", y_label="Power (a.u.)"
        )

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
                viewer.ax.set_title(f"Maximum SNR: {np.nanmax(np.abs(snr2D)):.2f}")  # type: ignore
                viewer.update_show(snr2D)

    except KeyboardInterrupt:
        print("Received KeyboardInterrupt, early stopping the program")
    except Exception as e:
        print("Error during measurement:", e)
    finally:
        if instant_show:
            viewer.ax.set_title(f"Maximum SNR: {np.nanmax(np.abs(snr2D)):.2f}")  # type: ignore
            viewer.update_show(snr2D)
            viewer.close_show()

    return pdrs, fpts, snr2D  # (pdrs, freqs)


def measure_ge_ro_dep(soc, soccfg, cfg, instant_show=False):
    cfg = deepcopy(cfg)  # prevent in-place modification

    res_pulse = cfg["dac"]["res_pulse"]

    ro_lens = sweep2array(cfg["sweep"], allow_array=True)
    check_time_sweep(soccfg, ro_lens, ro_ch=cfg["adc"]["chs"][0])

    del cfg["sweep"]  # remove sweep from cfg

    trig_offset = cfg["adc"]["trig_offset"]

    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

    if instant_show:
        show_period = int(len(ro_lens) / 20 + 0.99999)
        viewer = InstantShow(
            ro_lens, x_label="Readout Length (us)", y_label="SNR (a.u.)"
        )

    snrs = np.full(len(ro_lens), np.nan, dtype=np.complex128)
    try:
        for i, ro_len in enumerate(tqdm(ro_lens, desc="ro length", smoothing=0)):
            cfg["adc"]["ro_length"] = ro_len
            res_pulse["length"] = trig_offset + ro_len + 1.0

            snrs[i] = measure_one(soc, soccfg, cfg)

            if instant_show and i % show_period == 0:
                viewer.update_show(snrs)

    except KeyboardInterrupt:
        print("Received KeyboardInterrupt, early stopping the program")
    except Exception as e:
        print("Error during measurement:", e)
    finally:
        if instant_show:
            viewer.update_show(snrs)
            viewer.close_show()

    return ro_lens, snrs


def measure_ge_trig_dep(soc, soccfg, cfg, instant_show=False):
    cfg = deepcopy(cfg)  # prevent in-place modification

    offsets = sweep2array(cfg["sweep"], allow_array=True)
    check_time_sweep(soccfg, offsets)
    ro_len = cfg["adc"]["ro_length"]
    orig_offset = cfg["adc"]["trig_offset"]

    del cfg["sweep"]  # remove sweep from cfg

    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])
    if instant_show:
        show_period = int(len(offsets) / 20 + 0.99999)
        viewer = InstantShow(
            offsets, x_label="Trigger Offset (us)", y_label="SNR (a.u.)"
        )

    snrs = np.full(len(offsets), np.nan, dtype=np.complex128)
    try:
        for i, offset in enumerate(tqdm(offsets, desc="trig offset", smoothing=0)):
            cfg["adc"]["trig_offset"] = offset
            cfg["adc"]["ro_length"] = ro_len + offset - orig_offset

            snrs[i] = measure_one(soc, soccfg, cfg)

            if instant_show and i % show_period == 0:
                viewer.update_show(snrs)

    except KeyboardInterrupt:
        print("Received KeyboardInterrupt, early stopping the program")
    except Exception as e:
        print("Error during measurement:", e)
    finally:
        if instant_show:
            viewer.update_show(snrs)
            viewer.close_show()

    return offsets, snrs
