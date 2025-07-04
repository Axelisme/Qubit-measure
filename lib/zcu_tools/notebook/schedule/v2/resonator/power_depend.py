from typing import Optional, Tuple

import numpy as np
from zcu_tools.auto import make_cfg
from zcu_tools.liveplot.jupyter import LivePlotter2DwithLine
from zcu_tools.notebook.single_qubit.process import minus_background, rescale
from zcu_tools.program.v2 import OneToneProgram

from ...tools import map2adcfreq, sweep2array, sweep2param
from ..template import sweep2D_soft_hard_template


def signal2real(signals: np.ndarray) -> np.ndarray:
    return rescale(minus_background(np.abs(signals), axis=1), axis=1)


def measure_res_pdr_dep(
    soc, soccfg, cfg, dynamic_avg=False, gain_ref=0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cfg = make_cfg(cfg)  # prevent in-place modification

    res_pulse = cfg["dac"]["res_pulse"]
    pdr_sweep = cfg["sweep"]["gain"]
    fpt_sweep = cfg["sweep"]["freq"]
    reps_ref = cfg["reps"]
    rounds_ref = cfg["rounds"]

    del cfg["sweep"]["gain"]  # use soft for loop here

    res_pulse["freq"] = sweep2param("freq", fpt_sweep)

    pdrs = sweep2array(pdr_sweep, allow_array=True)
    fpts = sweep2array(fpt_sweep)  # predicted frequency points
    fpts = map2adcfreq(soccfg, fpts, res_pulse["ch"], cfg["adc"]["chs"][0])

    res_pulse["gain"] = pdrs[0]  # set initial power

    def updateCfg(cfg, _, pdr) -> None:
        cfg["dac"]["res_pulse"]["gain"] = pdr

        # change reps and rounds based on power
        if dynamic_avg:
            dyn_factor = (gain_ref / pdr) ** 2
            if dyn_factor > 1:
                # increase reps
                cfg["reps"] = int(reps_ref * dyn_factor)
                max_reps = min(100 * reps_ref, 1000000)
                if cfg["reps"] > max_reps:
                    cfg["reps"] = max_reps
            elif cfg["soft_avgs"] > 1:
                # decrease rounds
                cfg["rounds"] = int(rounds_ref * dyn_factor)
                min_avgs = max(int(0.1 * rounds_ref), 1)
                if cfg["rounds"] < min_avgs:
                    cfg["rounds"] = min_avgs
                cfg["soft_avgs"] = cfg["rounds"]  # this two are the smae
            else:
                # decrease reps
                cfg["reps"] = int(reps_ref * dyn_factor)
                min_reps = max(int(0.1 * reps_ref), 1)
                if cfg["reps"] < min_reps:
                    cfg["reps"] = min_reps

    prog: Optional[OneToneProgram] = None

    def measure_fn(cfg, callback) -> Tuple[list, list]:
        nonlocal prog
        prog = OneToneProgram(soccfg, cfg)
        return prog.acquire(soc, progress=False, callback=callback)

    signals2D = sweep2D_soft_hard_template(
        cfg,
        measure_fn,
        LivePlotter2DwithLine("Power (a.u.)", "Frequency (MHz)", line_axis=1, num_lines=10),
        xs=pdrs,
        ys=fpts,
        updateCfg=updateCfg,
        signal2real=signal2real,
    )

    # get the actual frequency points
    fpts = prog.get_pulse_param("res_pulse", "freq", as_array=True)

    return pdrs, fpts, signals2D
