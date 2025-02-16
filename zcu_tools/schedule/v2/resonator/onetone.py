import numpy as np

from zcu_tools import make_cfg
from zcu_tools.program.v2 import OneToneProgram
from zcu_tools.schedule.flux import set_flux
from zcu_tools.schedule.instant_show import clear_show, init_show, update_show
from zcu_tools.schedule.tools import format_sweep1D, sweep2array, sweep2param


def sweep_onetone(soc, soccfg, cfg, p_attr, progress=True, callback=None, **kwargs):
    prog = OneToneProgram(soccfg, cfg)
    xs = prog.get_pulse_param("res_pulse", p_attr, as_array=True)

    IQlist = prog.acquire(soc, progress=progress, round_callback=callback, **kwargs)
    signals = IQlist[0][0].dot([1, 1j])

    return xs, signals


def measure_res_freq(soc, soccfg, cfg, progress=True, instant_show=False):
    cfg = make_cfg(cfg)  # prevent in-place modification

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "freq")

    sweep_cfg = cfg["sweep"]["freq"]
    cfg["dac"]["res_pulse"]["freq"] = sweep2param("freq", sweep_cfg)

    if instant_show:
        # predict fpts
        fpts = sweep2array(sweep_cfg, False)
        fig, ax, dh, curve = init_show(fpts, "Frequency (MHz)", "Amplitude")

        def callback(ir, sum_d):
            amps = np.abs(sum_d[0][0].dot([1, 1j]) / (ir + 1))
            update_show(fig, ax, dh, curve, amps)
    else:
        callback = None  # type: ignore

    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

    fpts, signals = sweep_onetone(
        soc,
        soccfg,
        cfg,
        p_attr="freq",
        progress=progress,
        callback=callback,
    )

    if instant_show:
        update_show(fig, ax, dh, curve, np.abs(signals), fpts)
        clear_show(fig, dh)

    return fpts, signals
