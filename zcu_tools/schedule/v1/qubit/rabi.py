from copy import deepcopy

import numpy as np

from zcu_tools.program.v1 import RGainTwoToneProgram, TwoToneProgram
from zcu_tools.schedule.tools import sweep2array
from zcu_tools.schedule.v1.template import sweep1D_hard_template, sweep1D_soft_template


def measure_lenrabi(soc, soccfg, cfg, instant_show=False):
    cfg = deepcopy(cfg)

    lens = sweep2array(cfg["sweep"], allow_array=True)

    del cfg["sweep"]  # remove sweep for program use

    def update_cfg(cfg, _, length):
        cfg["dac"]["qub_pulse"]["length"] = length

    lens, signals = sweep1D_soft_template(
        soc,
        soccfg,
        cfg,
        TwoToneProgram,
        xs=lens,
        init_signals=np.full(len(lens), np.nan, dtype=np.complex128),
        progress=True,
        instant_show=instant_show,
        updateCfg=update_cfg,
        xlabel="Length (us)",
        ylabel="Amplitude",
    )

    return lens, signals


def measure_amprabi(soc, soccfg, cfg, instant_show=False):
    cfg = deepcopy(cfg)

    pdrs = sweep2array(cfg["sweep"])

    pdrs, signals = sweep1D_hard_template(
        soc,
        soccfg,
        cfg,
        RGainTwoToneProgram,
        init_xs=pdrs,
        init_signals=np.full(len(pdrs), np.nan, dtype=np.complex128),
        progress=True,
        instant_show=instant_show,
        xlabel="Pulse Power (a.u.)",
        ylabel="Amplitude",
    )

    return pdrs, signals
