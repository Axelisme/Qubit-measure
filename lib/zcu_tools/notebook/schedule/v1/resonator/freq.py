from copy import deepcopy
from typing import Tuple

from numpy import ndarray
from zcu_tools.liveplot.jupyter import LivePlotter1D
from zcu_tools.program.v1 import OneToneProgram

from ...tools import map2adcfreq, sweep2array
from ..template import sweep1D_soft_template


def measure_res_freq(soc, soccfg, cfg) -> Tuple[ndarray, ndarray]:
    cfg = deepcopy(cfg)  # prevent in-place modification

    res_pulse = cfg["dac"]["res_pulse"]

    fpts = sweep2array(cfg["sweep"], allow_array=True)
    fpts = map2adcfreq(soccfg, fpts, res_pulse["ch"], cfg["adc"]["chs"][0])

    del cfg["sweep"]  # remove sweep for program

    def updateCfg(cfg, _, f):
        cfg["dac"]["res_pulse"]["freq"] = f

    def measure_fn(cfg, callback) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
        prog = OneToneProgram(soccfg, cfg)
        return prog.acquire(soc, progress=False, callback=callback)

    fpts, signals = sweep1D_soft_template(
        cfg,
        measure_fn,
        LivePlotter1D("Frequency (MHz)", "Amplitude"),
        xs=fpts,
        updateCfg=updateCfg,
    )

    return fpts, signals
