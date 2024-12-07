from copy import deepcopy

import numpy as np
from tqdm.auto import tqdm

from zcu_tools import make_cfg
from zcu_tools.analysis import singleshot_analysis, fidelity_func
from zcu_tools.program import SingleShotProgram


def measure_fid(soc, soccfg, cfg, threshold, angle, progress=False):
    """return: fidelity, (tp, fp, tn, fn)"""
    prog = SingleShotProgram(soccfg, deepcopy(cfg))
    result = prog.acquire_orig(soc, threshold=threshold, angle=angle, progress=progress)
    fp, tp = result[1][0][0]
    fn, tn = 1 - fp, 1 - tp
    return fidelity_func(tp, tn, fp, fn)


def measure_fid_auto(soc, soccfg, cfg, plot=False, progress=False):
    prog = SingleShotProgram(soccfg, deepcopy(cfg))
    i0, q0 = prog.acquire(soc, progress=progress)
    fid, threhold, angle = singleshot_analysis(i0, q0, plot=plot)
    return fid, threhold, angle, np.array(i0 + 1j * q0)


def scan_pdr_fid(soc, soccfg, cfg, instant_show=False):
    cfg = deepcopy(cfg)  # prevent in-place modification

    sweep_cfg = cfg["sweep"]
    pdrs = np.arange(sweep_cfg["start"], sweep_cfg["stop"], sweep_cfg["step"])

    res_pulse = cfg["res_pulse"]

    if instant_show:
        import matplotlib.pyplot as plt
        from IPython.display import clear_output, display

        fig, ax = plt.subplots()
        ax.set_xlabel("Power (a.u.)")
        ax.set_ylabel("Fidelity")
        ax.set_title("Power-dependent measurement")
        curve = ax.plot(pdrs, np.zeros_like(pdrs))[0]
        dh = display(fig, display_id=True)

    fids = np.full(len(pdrs), np.nan)
    for i, pdr in enumerate(tqdm(pdrs)):
        res_pulse["gain"] = pdr
        fid, *_ = measure_fid_auto(soc, soccfg, make_cfg(cfg), progress=False)
        fids[i] = fid

        if instant_show:
            curve.set_ydata(fids)
            ax.relim()
            ax.autoscale(axis="y")
            dh.update(fig)

    if instant_show:
        clear_output()

    return pdrs, fids


def scan_len_fid(soc, soccfg, cfg, instant_show=False):
    cfg = deepcopy(cfg)  # prevent in-place modification
    del cfg["ro_cfg"]["ro_length"]  # let it be auto derived

    sweep_cfg = cfg["sweep"]
    lens = np.linspace(sweep_cfg["start"], sweep_cfg["stop"], sweep_cfg["expts"])

    res_pulse = cfg["res_pulse"]

    if instant_show:
        import matplotlib.pyplot as plt
        from IPython.display import clear_output, display

        fig, ax = plt.subplots()
        ax.set_xlabel("Length (ns)")
        ax.set_ylabel("Fidelity")
        ax.set_title("Length-dependent measurement")
        curve = ax.plot(lens, np.zeros_like(lens))[0]
        dh = display(fig, display_id=True)

    fids = np.full(len(lens), np.nan)
    for i, length in enumerate(tqdm(lens)):
        res_pulse["length"] = length
        fid, *_ = measure_fid_auto(soc, soccfg, make_cfg(cfg), progress=False)
        fids[i] = fid

        if instant_show:
            curve.set_ydata(fids)
            ax.relim()
            ax.autoscale(axis="y")
            dh.update(fig)

    if instant_show:
        clear_output()

    return lens, fids


def scan_freq_fid(soc, soccfg, cfg, instant_show=False):
    cfg = deepcopy(cfg)  # prevent in-place modification

    sweep_cfg = cfg["sweep"]
    fpts = np.linspace(sweep_cfg["start"], sweep_cfg["stop"], sweep_cfg["expts"])

    res_pulse = cfg["res_pulse"]

    if instant_show:
        import matplotlib.pyplot as plt
        from IPython.display import clear_output, display

        fig, ax = plt.subplots()
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Fidelity")
        ax.set_title("Frequency-dependent measurement")
        curve = ax.plot(fpts, np.zeros_like(fpts))[0]
        dh = display(fig, display_id=True)

    fids = np.full(len(fpts), np.nan)
    for i, fpt in enumerate(tqdm(fpts)):
        res_pulse["freq"] = fpt
        fid, *_ = measure_fid_auto(soc, soccfg, make_cfg(cfg), progress=False)
        fids[i] = fid

        if instant_show:
            curve.set_ydata(fids)
            ax.relim()
            ax.autoscale(axis="y")
            dh.update(fig)

    if instant_show:
        clear_output()

    return fpts, fids


def scan_style_fid(soc, soccfg, cfg) -> dict:
    cfg = deepcopy(cfg)  # prevent in-place modification

    sweep_list = cfg["sweep"]

    res_pulse = cfg["res_pulse"]

    fids = {}
    for style in sweep_list:
        res_pulse["style"] = style
        fid, *_ = measure_fid_auto(soc, soccfg, make_cfg(cfg), progress=False)
        fids[style] = fid
    return fids
