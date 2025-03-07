from copy import deepcopy

from zcu_tools.program.v1 import OneToneProgram
from zcu_tools.schedule.tools import map2adcfreq, sweep2array
from zcu_tools.schedule.v1.template import sweep1D_soft_template


def measure_res_freq(soc, soccfg, cfg):
    cfg = deepcopy(cfg)  # prevent in-place modification

    res_pulse = cfg["dac"]["res_pulse"]

    fpts = sweep2array(cfg["sweep"], allow_array=True)
    fpts = map2adcfreq(soccfg, fpts, res_pulse["ch"], cfg["adc"]["chs"][0])

    del cfg["sweep"]  # remove sweep for program

    def updateCfg(cfg, _, f):
        cfg["dac"]["res_pulse"]["freq"] = f

    fpts, signals = sweep1D_soft_template(
        soc,
        soccfg,
        cfg,
        OneToneProgram,
        xs=fpts,
        updateCfg=updateCfg,
        xlabel="Frequency (MHz)",
        ylabel="Amplitude",
    )

    return fpts, signals
