from copy import deepcopy

import numpy as np
from tqdm.auto import trange

from zcu_tools import make_cfg
from zcu_tools.analysis import fidelity_func, singleshot_analysis
from zcu_tools.analysis.single_shot.base import rotate
from zcu_tools.analysis.single_shot.regression import get_rotate_angle
from zcu_tools.program import SingleShotProgram

from ..flux import set_flux
from ..instant_show import clear_show, init_show, update_show


def measure_fid(soc, soccfg, cfg, threshold, angle, progress=False):
    """return: fidelity, (tp, fp, tn, fn)"""

    set_flux(cfg["flux_dev"], cfg["flux"])
    prog = SingleShotProgram(soccfg, deepcopy(cfg))
    result = prog.acquire_orig(soc, threshold=threshold, angle=angle, progress=progress)
    fp, tp = result[1][0][0]
    fn, tn = 1 - fp, 1 - tp
    return fidelity_func(tp, tn, fp, fn)


def measure_fid_auto(
    soc, soccfg, cfg, plot=False, progress=False, backend="regression"
):
    set_flux(cfg["flux_dev"], cfg["flux"])
    prog = SingleShotProgram(soccfg, deepcopy(cfg))
    i0, q0 = prog.acquire(soc, progress=progress)
    fid, threhold, angle = singleshot_analysis(i0, q0, plot=plot, backend=backend)
    return fid, threhold, angle, np.array(i0 + 1j * q0)


def measure_fid_score(soc, soccfg, cfg):
    set_flux(cfg["flux_dev"], cfg["flux"])
    prog = SingleShotProgram(soccfg, deepcopy(cfg))
    i0, q0 = prog.acquire(soc, progress=False)

    numbins = 200
    Ig, Ie = i0
    Qg, Qe = q0

    # Calculate the angle of rotation
    out_dict = get_rotate_angle(Ig, Qg, Ie, Qe)
    theta = out_dict["theta"]

    # Rotate the IQ data
    Ig, _ = rotate(Ig, Qg, theta)
    Ie, _ = rotate(Ie, Qe, theta)

    # calculate histogram
    Itot = np.concatenate((Ig, Ie))
    xlims = (Itot.min(), Itot.max())
    bins = np.linspace(*xlims, numbins)
    ng, *_ = np.histogram(Ig, bins=bins, range=xlims)
    ne, *_ = np.histogram(Ie, bins=bins, range=xlims)

    # calculate the total dist between g and e
    contrast = np.sum(np.abs(ng - ne))

    return contrast


def perform_fid_scan(
    soc,
    soccfg,
    cfg,
    instant_show,
    reps,
    scan_points,
    update_func=None,
    backend="regression",
):
    if instant_show:
        fig, ax, dh, curve = init_show(scan_points, "Iteration", "Fidelity")

    set_flux(cfg["flux_dev"], cfg["flux"])

    scores = np.full((len(scan_points), reps), np.nan)
    scores[:, 0] = 0
    try:
        for j in trange(reps):
            # instead of scan one by one, randomize the order
            order = np.random.permutation(len(scan_points))
            for i in order:
                point = scan_points[i]
                if update_func:
                    update_func(cfg, point)
                fid = measure_fid_score(soc, soccfg, make_cfg(cfg))
                scores[i, j] = fid

                if instant_show:
                    avg_score = scores.copy()
                    for i in range(avg_score.shape[0]):
                        if np.sum(~np.isnan(avg_score[i])) > 2:
                            # remove max and min value
                            avg_score[i, np.argmax(avg_score[i])] = np.nan
                            avg_score[i, np.argmin(avg_score[i])] = np.nan
                    avg_score = np.nanmean(avg_score, axis=1)
                    update_show(fig, ax, dh, curve, avg_score)
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        pass

    if instant_show:
        clear_show()

    return np.nanmean(scores, axis=1)


def scan_pdr(soc, soccfg, cfg, instant_show=False, reps=5, backend="regression"):
    cfg = deepcopy(cfg)  # prevent in-place modification

    sweep_cfg = cfg["sweep"]
    pdrs = np.arange(sweep_cfg["start"], sweep_cfg["stop"], sweep_cfg["step"])

    def update_pdr(cfg, pdr):
        cfg["dac"]["res_pulse"]["gain"] = pdr

    return pdrs, perform_fid_scan(
        soc, soccfg, cfg, instant_show, reps, pdrs, update_pdr, backend=backend
    )


def scan_offset(soc, soccfg, cfg, instant_show=False, reps=5, backend="regression"):
    cfg = deepcopy(cfg)  # prevent in-place modification
    ro_end = cfg["adc"]["trig_offset"] + cfg["adc"]["ro_length"]

    sweep_cfg = cfg["sweep"]
    offsets = np.arange(sweep_cfg["start"], sweep_cfg["stop"], sweep_cfg["step"])

    assert np.all(offsets < ro_end), "offset should be less than ro_end"

    def update_ro_start(cfg, offset):
        cfg["adc"]["trig_offset"] = offset
        cfg["adc"]["ro_length"] = ro_end - offset

    return offsets, perform_fid_scan(
        soc, soccfg, cfg, instant_show, reps, offsets, update_ro_start, backend=backend
    )


def scan_ro_len(soc, soccfg, cfg, instant_show=False, reps=5, backend="regression"):
    cfg = deepcopy(cfg)  # prevent in-place modification

    sweep_cfg = cfg["sweep"]
    ro_lengths = np.arange(sweep_cfg["start"], sweep_cfg["stop"], sweep_cfg["step"])

    assert np.all(ro_lengths > 0), "ro_length should be positive"

    def update_ro_len(cfg, ro_length):
        cfg["adc"]["ro_length"] = ro_length

    return ro_lengths, perform_fid_scan(
        soc,
        soccfg,
        cfg,
        instant_show,
        reps,
        ro_lengths,
        update_ro_len,
        backend=backend,
    )


def scan_res_len(soc, soccfg, cfg, instant_show=False, reps=5, backend="regression"):
    cfg = deepcopy(cfg)  # prevent in-place modification

    sweep_cfg = cfg["sweep"]
    res_lengths = np.arange(sweep_cfg["start"], sweep_cfg["stop"], sweep_cfg["step"])
    res_length0 = cfg["dac"]["res_pulse"]["length"]
    post_offset = cfg["adc"]["ro_length"] - res_length0

    assert np.all(res_lengths > 0), "res_length should be positive"
    assert np.all(
        res_lengths + post_offset > 0
    ), "negative ro_length detected, please adjust the sweep range"

    def update_res_len(cfg, res_length):
        cfg["dac"]["res_pulse"]["length"] = res_length
        cfg["adc"]["ro_length"] = res_length + post_offset

    return res_lengths, perform_fid_scan(
        soc,
        soccfg,
        cfg,
        instant_show,
        reps,
        res_lengths,
        update_res_len,
        backend=backend,
    )


def scan_freq(soc, soccfg, cfg, instant_show=False, reps=5, backend="regression"):
    cfg = deepcopy(cfg)  # prevent in-place modification

    set_flux(cfg["flux_dev"], cfg["flux"])

    sweep_cfg = cfg["sweep"]
    fpts = np.linspace(sweep_cfg["start"], sweep_cfg["stop"], sweep_cfg["expts"])

    def update_freq(cfg, freq):
        cfg["dac"]["res_pulse"]["freq"] = freq

    return fpts, perform_fid_scan(
        soc, soccfg, cfg, instant_show, reps, fpts, update_freq, backend=backend
    )
