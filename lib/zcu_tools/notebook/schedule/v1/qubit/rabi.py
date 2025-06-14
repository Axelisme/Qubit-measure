from copy import deepcopy
from typing import Tuple

import numpy as np
from numpy import ndarray
from zcu_tools.liveplot.jupyter import LivePlotter1D
from zcu_tools.notebook.single_qubit.process import rotate2real
from zcu_tools.program.v1 import RGainTwoToneProgram, TwoToneProgram

from ...tools import format_sweep1D, sweep2array
from ..template import sweep1D_hard_template, sweep1D_soft_template


def signal2real(signals: np.ndarray) -> np.ndarray:
    return rotate2real(signals).real


def measure_lenrabi(soc, soccfg, cfg) -> Tuple[ndarray, ndarray]:
    cfg = deepcopy(cfg)

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")
    len_sweep = cfg["sweep"]["length"]

    lens = sweep2array(len_sweep, allow_array=True)

    del cfg["sweep"]  # remove sweep for program use

    def updateCfg(cfg, _, length) -> None:
        cfg["dac"]["qub_pulse"]["length"] = length

    def measure_fn(cfg, callback) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
        prog = TwoToneProgram(soccfg, cfg)
        return prog.acquire(soc, progress=False, callback=callback)

    lens, signals = sweep1D_soft_template(
        cfg,
        measure_fn,
        LivePlotter1D("Length (us)", "Amplitude"),
        xs=lens,
        updateCfg=updateCfg,
        signal2real=signal2real,
    )

    return lens, signals


def measure_amprabi(soc, soccfg, cfg):
    cfg = deepcopy(cfg)

    pdrs = sweep2array(cfg["sweep"])

    def measure_fn(cfg, callback) -> Tuple[ndarray, ...]:
        prog = RGainTwoToneProgram(soccfg, cfg)
        return prog.acquire(soc, progress=True, callback=callback)

    pdrs, signals = sweep1D_hard_template(
        cfg,
        measure_fn,
        LivePlotter1D("Pulse Power (a.u.)", "Amplitude"),
        xs=pdrs,
        signal2real=signal2real,
    )

    return pdrs, signals
