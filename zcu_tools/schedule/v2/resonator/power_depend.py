import numpy as np

from zcu_tools import make_cfg
from zcu_tools.analysis import minus_background, rescale
from zcu_tools.program.v2 import OneToneProgram
from zcu_tools.schedule.tools import map2adcfreq, sweep2array, sweep2param
from zcu_tools.schedule.v2.template import sweep2D_soft_hard_template


def signal2real(signals):
    return rescale(minus_background(np.abs(signals), axis=1), axis=1)


def measure_res_pdr_dep(soc, soccfg, cfg, dynamic_reps=False, gain_ref=0.1):
    cfg = make_cfg(cfg)  # prevent in-place modification

    res_pulse = cfg["dac"]["res_pulse"]
    pdr_sweep = cfg["sweep"]["gain"]
    fpt_sweep = cfg["sweep"]["freq"]
    reps_ref = cfg["reps"]

    del cfg["sweep"]["gain"]  # use soft for loop here

    res_pulse["freq"] = sweep2param("freq", fpt_sweep)

    pdrs = sweep2array(pdr_sweep, allow_array=True)
    fpts = sweep2array(fpt_sweep)  # predicted frequency points
    fpts = map2adcfreq(soccfg, fpts, res_pulse["ch"], cfg["adc"]["chs"][0])

    res_pulse["gain"] = pdrs[0]  # set initial power

    def updateCfg(cfg, i, pdr):
        res_pulse["gain"] = pdr

        if dynamic_reps:
            cfg["reps"] = int(reps_ref * gain_ref / max(pdr, 1e-6))
            if cfg["reps"] < 0.1 * reps_ref:
                cfg["reps"] = int(0.1 * reps_ref + 0.99)
            elif cfg["reps"] > 10 * reps_ref:
                cfg["reps"] = int(10 * reps_ref)

    prog, signals2D = sweep2D_soft_hard_template(
        soc,
        soccfg,
        cfg,
        OneToneProgram,
        xs=pdrs,
        ys=fpts,
        xlabel="Power (a.u.)",
        ylabel="Frequency (MHz)",
        updateCfg=updateCfg,
        signal2real=signal2real,
    )

    # get the actual frequency points
    fpts = prog.get_pulse_param("res_pulse", "freq", as_array=True)

    return pdrs, fpts, signals2D
