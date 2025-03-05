import numpy as np

from zcu_tools import make_cfg
from zcu_tools.analysis import minus_background
from zcu_tools.program.v2 import OneToneProgram
from zcu_tools.schedule.tools import map2adcfreq, sweep2array, sweep2param
from zcu_tools.schedule.v2.template import sweep2D_soft_hard_template


def signal2real(signals):
    return minus_background(np.abs(signals), axis=1)


def measure_res_flux_dep(soc, soccfg, cfg):
    cfg = make_cfg(cfg)  # prevent in-place modification

    res_pulse = cfg["dac"]["res_pulse"]
    fpt_sweep = cfg["sweep"]["freq"]
    flx_sweep = cfg["sweep"]["flux"]

    del cfg["sweep"]["flux"]  # use soft for loop here

    res_pulse["freq"] = sweep2param("freq", fpt_sweep)

    flxs = sweep2array(flx_sweep, allow_array=True)
    fpts = sweep2array(fpt_sweep)  # predicted frequency points
    fpts = map2adcfreq(soccfg, fpts, res_pulse["ch"], cfg["adc"]["chs"][0])

    cfg["dev"]["flux"] = flxs[0]  # set initial flux

    def updateCfg(cfg, i, flx):
        cfg["dev"]["flux"] = flx

    prog, signals2D = sweep2D_soft_hard_template(
        soc,
        soccfg,
        cfg,
        OneToneProgram,
        xs=flxs,
        ys=fpts,
        xlabel="Flux (a.u.)",
        ylabel="Frequency (MHz)",
        init_signals=np.full((len(flxs), len(fpts)), np.nan, dtype=complex),
        updateCfg=updateCfg,
        signal2real=signal2real,
    )

    # get the actual frequency points
    fpts = prog.get_pulse_param("res_pulse", "freq", as_array=True)

    return flxs, fpts, signals2D
