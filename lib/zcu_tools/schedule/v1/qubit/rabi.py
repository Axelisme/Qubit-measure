from copy import deepcopy

import numpy as np

from zcu_tools.analysis import minus_background
from zcu_tools.program.v1 import RGainTwoToneProgram, TwoToneProgram
from zcu_tools.schedule.tools import sweep2array
from zcu_tools.schedule.v1.template import sweep1D_hard_template, sweep1D_soft_template


def signal2real(signals: np.ndarray) -> np.ndarray:
    return np.abs(minus_background(signals))


def measure_lenrabi(soc, soccfg, cfg):
    cfg = deepcopy(cfg)

    lens = sweep2array(cfg["sweep"], allow_array=True)

    del cfg["sweep"]  # remove sweep for program use

    def updateCfg(cfg, _, length):
        cfg["dac"]["qub_pulse"]["length"] = length

    lens, signals = sweep1D_soft_template(
        soc,
        soccfg,
        cfg,
        TwoToneProgram,
        xs=lens,
        updateCfg=updateCfg,
        xlabel="Length (us)",
        ylabel="Amplitude",
        signal2real=signal2real,
    )

    return lens, signals


def measure_amprabi(soc, soccfg, cfg):
    cfg = deepcopy(cfg)

    pdrs = sweep2array(cfg["sweep"])

    pdrs, signals = sweep1D_hard_template(
        soc,
        soccfg,
        cfg,
        RGainTwoToneProgram,
        xs=pdrs,
        xlabel="Pulse Power (a.u.)",
        ylabel="Amplitude",
        signal2real=signal2real,
    )

    return pdrs, signals
