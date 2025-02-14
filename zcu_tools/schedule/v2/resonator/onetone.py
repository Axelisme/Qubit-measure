from functools import partial

import numpy as np

from zcu_tools import make_cfg
from zcu_tools.program.v2 import OneToneProgram
from zcu_tools.schedule.flux import set_flux
from zcu_tools.schedule.instant_show import clear_show, init_show, update_show
from zcu_tools.schedule.tools import format_sweep1D, sweep2param


def _measure_res_freq(soc, soccfg, cfg, progress=True, callback=None):
    cfg = make_cfg(cfg)  # prevent in-place modification

    cfg["dac"]["res_pulse"]["freq"] = sweep2param("res_freq", cfg["sweep"]["res_freq"])
    cfg = make_cfg(cfg)

    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

    prog = OneToneProgram(soccfg, cfg)
    fpts = prog.get_pulse_param("res_pulse", "freq", as_array=True)

    # partial fill callback with fpts
    if callback is not None:
        callback = partial(callback, fpts=fpts)

    IQlist = prog.acquire(soc, progress=progress, round_callback=callback)
    signals = IQlist[0][0].dot([1, 1j])

    return fpts, signals


def measure_res_freq(soc, soccfg, cfg, instant_show=False):
    cfg = make_cfg(cfg)  # prevent in-place modification

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "res_freq")
    sweep_cfg = cfg["sweep"]["res_freq"]

    if instant_show:
        # predict fpts
        fpts = np.linspace(sweep_cfg["start"], sweep_cfg["stop"], sweep_cfg["expts"])
        fig, ax, dh, curve = init_show(fpts, "Frequency (MHz)", "Amplitude")

        def callback(_, avg_d, *, fpts):
            amps = np.abs(avg_d[0][0].dot([1, 1j]))
            update_show(fig, ax, dh, curve, amps, fpts)
    else:
        callback = None  # type: ignore

    fpts, signals = _measure_res_freq(soc, soccfg, cfg, callback=callback)

    if instant_show:
        update_show(fig, ax, dh, curve, np.abs(signals), fpts)
        clear_show(fig, dh)

    return fpts, signals
