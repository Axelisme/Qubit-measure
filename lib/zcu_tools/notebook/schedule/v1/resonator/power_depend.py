from copy import deepcopy
from typing import Tuple

import numpy as np
from numpy import ndarray
from zcu_tools.liveplot.jupyter import LivePlotter2DwithLine
from zcu_tools.notebook.single_qubit.process import minus_background, rescale
from zcu_tools.program.v1 import OneToneProgram

from ...tools import map2adcfreq, sweep2array
from ..template import sweep2D_soft_soft_template


def signal2real(signals: ndarray) -> ndarray:
    return rescale(minus_background(np.abs(signals), axis=1), axis=1)


def measure_res_pdr_dep(
    soc, soccfg, cfg, dynamic_avg=False, gain_ref=1000
) -> Tuple[ndarray, ndarray, ndarray]:
    cfg = deepcopy(cfg)  # prevent in-place modification

    res_pulse = cfg["dac"]["res_pulse"]
    reps_ref = cfg["reps"]
    rounds_ref = cfg["rounds"]

    fpts = sweep2array(cfg["sweep"]["freq"], allow_array=True)
    fpts = map2adcfreq(soccfg, fpts, res_pulse["ch"], cfg["adc"]["chs"][0])
    pdrs = sweep2array(cfg["sweep"]["gain"], allow_array=True)

    del cfg["sweep"]  # remove sweep for program use

    def x_updateCfg(cfg, _, pdr) -> None:
        cfg["dac"]["res_pulse"]["gain"] = pdr

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

    def y_updateCfg(cfg, _, fpt) -> None:
        cfg["dac"]["res_pulse"]["freq"] = fpt

    def measure_fn(cfg, callback) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
        prog = OneToneProgram(soccfg, cfg)
        return prog.acquire(soc, progress=False, callback=callback)

    pdrs, fpts, signals2D = sweep2D_soft_soft_template(
        cfg,
        measure_fn,
        LivePlotter2DwithLine("Power (a.u.)", "Frequency (MHz)", num_lines=10),
        xs=pdrs,
        ys=fpts,
        x_updateCfg=x_updateCfg,
        y_updateCfg=y_updateCfg,
        signal2real=signal2real,
    )

    return pdrs, fpts, signals2D
