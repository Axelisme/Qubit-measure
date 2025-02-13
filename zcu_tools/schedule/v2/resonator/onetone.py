from copy import deepcopy

import numpy as np

from zcu_tools import make_cfg
from zcu_tools.program.v2 import OneToneProgram
from zcu_tools.schedule.flux import set_flux
from zcu_tools.schedule.instant_show import clear_show, init_show, update_show
from zcu_tools.schedule.tools import format_sweep, sweep2param


def measure_res_freq(soc, soccfg, cfg, instant_show=False):
    cfg = deepcopy(cfg)  # prevent in-place modification

    res_pulse = cfg["dac"]["res_pulse"]
    sweep = cfg["sweep"]

    cfg["sweep"] = format_sweep(sweep, "res_freq")
    res_pulse["freq"] = sweep2param(cfg["sweep"])
    cfg = make_cfg(cfg)

    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

    prog = OneToneProgram(soccfg, cfg)
    fpts = prog.get_pulse_param("res_pulse", "freq", as_array=True)

    if instant_show:
        fig, ax, dh, curve = init_show(fpts, "Frequency (MHz)", "Amplitude")

        def callback(ir, avg_d):
            update_show(fig, ax, dh, curve, np.abs(avg_d[0][0].dot([1, 1j])))
    else:
        callback = None

    IQlist = prog.acquire(
        soc, soft_avgs=cfg["soft_avgs"], progress=True, round_callback=callback
    )
    signals = IQlist[0][0].dot([1, 1j])

    if instant_show:
        update_show(fig, ax, dh, curve, np.abs(signals))
        clear_show(fig, dh)

    return fpts, signals
