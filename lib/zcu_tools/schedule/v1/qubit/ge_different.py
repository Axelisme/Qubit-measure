from copy import deepcopy

import numpy as np
from scipy.optimize import minimize
from tqdm.auto import tqdm
from zcu_tools.program.v1 import TwoToneProgram
from zcu_tools.schedule.flux import set_flux
from zcu_tools.schedule.instant_show import (
    InstantShow1D,
    InstantShow2D,
    InstantShowScatter,
)
from zcu_tools.schedule.tools import check_time_sweep, map2adcfreq, sweep2array
from zcu_tools.tools import AsyncFunc, print_traceback


def measure_one(soc, soccfg, cfg):
    cfg = deepcopy(cfg)  # prevent in-place modification

    qub_pulse = cfg["dac"]["qub_pulse"]
    pi_gain = qub_pulse["gain"]

    qub_pulse["gain"] = 0
    prog = TwoToneProgram(soccfg, cfg)
    avggi, avggq, stdgi, stdgq = prog.acquire(soc, progress=False, ret_std=True)

    qub_pulse["gain"] = pi_gain
    prog = TwoToneProgram(soccfg, cfg)
    avgei, avgeq, stdei, stdeq = prog.acquire(soc, progress=False, ret_std=True)

    dist_i = avgei[0][0] - avggi[0][0]
    dist_q = avgeq[0][0] - avggq[0][0]
    contrast = dist_i + 1j * dist_q
    noise2_i = stdgi[0][0] ** 2 + stdei[0][0] ** 2
    noise2_q = stdgq[0][0] ** 2 + stdeq[0][0] ** 2
    noise = np.sqrt(noise2_i * dist_i**2 + noise2_q * dist_q**2) / np.abs(contrast)

    return contrast / noise


def measure_ge_pdr_dep(soc, soccfg, cfg):
    cfg = deepcopy(cfg)  # prevent in-place modification

    res_pulse = cfg["dac"]["res_pulse"]

    freq_cfg = cfg["sweep"]["freq"]
    pdr_cfg = cfg["sweep"]["gain"]
    fpts = sweep2array(freq_cfg, allow_array=True)
    fpts = map2adcfreq(soccfg, fpts, res_pulse["ch"], cfg["adc"]["chs"][0])
    pdrs = sweep2array(pdr_cfg, allow_array=True)

    del cfg["sweep"]  # remove sweep from cfg

    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])  # set initial flux

    snr2D = np.full((len(pdrs), len(fpts)), np.nan, dtype=complex)
    with InstantShow2D(pdrs, fpts, "Power (a.u)", "Frequency (MHz)") as viewer:
        try:
            pdrs_tqdm = tqdm(pdrs, desc="Gain", smoothing=0)
            fpts_tqdm = tqdm(fpts, desc="Freq", smoothing=0)
            with AsyncFunc(viewer.update_show, include_idx=False) as async_draw:
                for i, pdr in enumerate(pdrs):
                    cfg["dac"]["res_pulse"]["gain"] = pdr

                    fpts_tqdm.reset()
                    fpts_tqdm.refresh()
                    for j, fpt in enumerate(fpts):
                        cfg["dac"]["res_pulse"]["freq"] = fpt

                        snr2D[i, j] = measure_one(soc, soccfg, cfg)

                        fpts_tqdm.update()

                        async_draw(i * len(fpts) + j, np.abs(snr2D))
                    pdrs_tqdm.update()

        except KeyboardInterrupt:
            print("Received KeyboardInterrupt, early stopping the program")
        except Exception:
            print("Error during measurement:")
            print_traceback()
        finally:
            viewer.update_show(np.abs(snr2D), (pdrs, fpts))
            pdrs_tqdm.close()
            fpts_tqdm.close()

    return pdrs, fpts, snr2D


def measure_ge_pdr_dep_auto(soc, soccfg, cfg, method="Nelder-Mead"):
    cfg = deepcopy(cfg)  # prevent in-place modification

    res_pulse = cfg["dac"]["res_pulse"]

    pdrs = sweep2array(cfg["sweep"]["gain"])  # predicted pulse gains
    fpts = sweep2array(cfg["sweep"]["freq"])  # predicted frequency points
    fpts = map2adcfreq(soc, fpts, res_pulse["ch"], cfg["adc"]["chs"][0])

    del cfg["sweep"]["gain"]  # program should not use this
    del cfg["sweep"]["freq"]  # program should not use this

    # set again in case of change
    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

    records = []
    with InstantShowScatter("Readout Gain", "Frequency (MHz)", title="SNR") as viewer:
        count = 0

        def loss_func(point, cfg):
            nonlocal count
            cfg = deepcopy(cfg)  # prevent in-place modification

            pdr, fpt = point
            cfg["dac"]["res_pulse"]["gain"] = int(pdr)
            cfg["dac"]["res_pulse"]["freq"] = fpt

            snr = measure_one(soc, soccfg, cfg)
            count += 1

            records.append((pdr, fpt, snr))

            viewer.append_spot(
                pdr, fpt, np.abs(snr), title=f"SNR_{count}: {np.abs(snr):.3e}"
            )

            return -np.abs(snr)

        options = dict(maxiter=(len(pdrs) * len(fpts)) // 5)

        if method in ["Nelder-Mead", "Powell"]:
            options["xatol"] = min(pdrs[1] - pdrs[0], fpts[1] - fpts[0])
        elif method in ["L-BFGS-B"]:
            options["ftol"] = 1e-4  # type: ignore
            options["maxfun"] = options["maxiter"]

        init_point = (0.5 * (pdrs[0] + pdrs[-1]), 0.5 * (fpts[0] + fpts[-1]))
        res = minimize(
            loss_func,
            init_point,
            args=(cfg,),
            method=method,
            bounds=[(pdrs[0], pdrs[-1]), (fpts[0], fpts[-1])],
            options=options,
        )

    if isinstance(res, np.ndarray):
        return res, records
    return res.x, records


def measure_ge_ro_dep(soc, soccfg, cfg):
    cfg = deepcopy(cfg)  # prevent in-place modification

    ro_lens = sweep2array(cfg["sweep"], allow_array=True)
    check_time_sweep(soccfg, ro_lens, ro_ch=cfg["adc"]["chs"][0])

    del cfg["sweep"]  # remove sweep from cfg

    trig_offset = cfg["adc"]["trig_offset"]

    # set flux first
    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

    snrs = np.full(len(ro_lens), np.nan, dtype=np.complex128)
    with InstantShow1D(ro_lens, "Readout Length (us)", "SNR") as viewer:
        try:
            lens_tqdm = tqdm(ro_lens, desc="Readout Length (us)", smoothing=0)
            with AsyncFunc(viewer.update_show, include_idx=False) as async_draw:
                for i, ro_len in enumerate(lens_tqdm):
                    cfg["adc"]["ro_length"] = ro_len
                    cfg["dac"]["res_pulse"]["length"] = trig_offset + ro_len + 1.0

                    snrs[i] = measure_one(soc, soccfg, cfg)

                    async_draw(i, np.abs(snrs))

        except KeyboardInterrupt:
            print("Received KeyboardInterrupt, early stopping the program")
        except Exception:
            print("Error during measurement:")
            print_traceback()
        finally:
            viewer.update_show(np.abs(snrs))

    return ro_lens, snrs


def measure_ge_trig_dep(soc, soccfg, cfg):
    cfg = deepcopy(cfg)  # prevent in-place modification

    offsets = sweep2array(cfg["sweep"], allow_array=True)
    check_time_sweep(soccfg, offsets, ro_ch=cfg["adc"]["chs"][0])

    del cfg["sweep"]  # remove sweep from cfg

    ro_len = cfg["adc"]["ro_length"]
    orig_offset = cfg["adc"]["trig_offset"]

    # set flux first
    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

    snrs = np.full(len(offsets), np.nan, dtype=np.complex128)
    with InstantShow1D(offsets, "Readout offset (us)", "SNR") as viewer:
        try:
            offsets_tqdm = tqdm(offsets, desc="Readout offset (us)", smoothing=0)
            with AsyncFunc(viewer.update_show, include_idx=False) as async_draw:
                for i, offset in enumerate(offsets_tqdm):
                    cfg["adc"]["trig_offset"] = offset
                    cfg["adc"]["ro_length"] = ro_len + offset - orig_offset

                    snrs[i] = measure_one(soc, soccfg, cfg)

                    async_draw(i, np.abs(snrs))

        except KeyboardInterrupt:
            print("Received KeyboardInterrupt, early stopping the program")
        except Exception:
            print("Error during measurement:")
            print_traceback()
        finally:
            viewer.update_show(np.abs(snrs))

    return offsets, snrs
