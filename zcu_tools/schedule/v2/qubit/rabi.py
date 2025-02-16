import numpy as np

from zcu_tools import make_cfg
from zcu_tools.schedule.flux import set_flux
from zcu_tools.schedule.instant_show import close_show, init_show, update_show
from zcu_tools.schedule.tools import format_sweep1D, sweep2array, sweep2param

from .twotone import sweep_twotone


def measure_lenrabi(soc, soccfg, cfg, instant_show=False):
    cfg = make_cfg(cfg)  # prevent in-place modification

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")

    sweep_cfg = cfg["sweep"]["length"]
    cfg["dac"]["qub_pulse"]["length"] = sweep2param("length", sweep_cfg)

    if instant_show:
        # predict lengths
        lens = sweep2array(sweep_cfg, False)
        fig, ax, dh, curve = init_show(lens, "Length (us)", "Amplitude")

        def callback(ir, sum_d):
            amps = np.abs(sum_d[0][0].dot([1, 1j]) / (ir + 1))
            update_show(fig, ax, dh, curve, amps)
    else:
        callback = None  # type: ignore

    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

    lens, signals = sweep_twotone(
        soc,
        soccfg,
        cfg,
        p_attr="length",
        progress=True,
        callback=callback,
    )

    if instant_show:
        update_show(fig, ax, dh, curve, np.abs(signals), lens)
        close_show(fig, dh)

    return lens, signals


def measure_amprabi(soc, soccfg, cfg, instant_show=False):
    cfg = make_cfg(cfg)  # prevent in-place modification

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "gain")

    sweep_cfg = cfg["sweep"]["gain"]
    cfg["dac"]["qub_pulse"]["gain"] = sweep2param("gain", sweep_cfg)

    if instant_show:
        # predict pdrs
        pdrs = sweep2array(sweep_cfg, False)
        fig, ax, dh, curve = init_show(pdrs, "Pulse gain", "Amplitude")

        def callback(ir, sum_d):
            amps = np.abs(sum_d[0][0].dot([1, 1j]) / (ir + 1))
            update_show(fig, ax, dh, curve, amps)
    else:
        callback = None  # type: ignore

    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

    pdrs, signals = sweep_twotone(
        soc,
        soccfg,
        cfg,
        p_attr="gain",
        progress=True,
        callback=callback,
    )

    if instant_show:
        update_show(fig, ax, dh, curve, np.abs(signals), pdrs)
        close_show(fig, dh)

    return pdrs, signals
