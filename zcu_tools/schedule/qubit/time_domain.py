from copy import deepcopy

import numpy as np

from zcu_tools.program import T1Program, T2EchoProgram, T2RamseyProgram

from ..flux import set_flux
from ..instant_show import clear_show, init_show, update_show
from ..tools import check_time_sweep, sweep2array


def safe_sweep2array(soccfg, sweep_cfg):
    ts = sweep2array(
        sweep_cfg, soft_loop=False, err_str="Custom time sweep only for soft loop"
    )
    check_time_sweep(soccfg, ts)

    return ts


def measure_t2ramsey(soc, soccfg, cfg, instant_show=False):
    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

    ts = safe_sweep2array(soccfg, cfg["sweep"])

    if instant_show:
        fig, ax, dh, curve = init_show(ts, "Time (us)", "Amplitude")

        def callback(ir, avg_d):
            avgi, avgq = avg_d[0][0, :, 0], avg_d[0][0, :, 1]
            update_show(fig, ax, dh, curve, np.abs(avgi + 1j * avgq))
    else:
        callback = None

    prog = T2RamseyProgram(soccfg, deepcopy(cfg))
    ts, avgi, avgq = prog.acquire(soc, progress=True, round_callback=callback)
    signals = avgi[0][0] + 1j * avgq[0][0]

    if instant_show:
        update_show(fig, ax, dh, curve, np.abs(signals), ts)
        clear_show()

    return ts, signals


def measure_t1(soc, soccfg, cfg, instant_show=False):
    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

    ts = safe_sweep2array(soccfg, cfg["sweep"])

    if instant_show:
        fig, ax, dh, curve = init_show(ts, "Time (us)", "Amplitude")

        def callback(ir, avg_d):
            avgi, avgq = avg_d[0][0, :, 0], avg_d[0][0, :, 1]
            update_show(fig, ax, dh, curve, np.abs(avgi + 1j * avgq))
    else:
        callback = None

    prog = T1Program(soccfg, deepcopy(cfg))
    ts, avgi, avgq = prog.acquire(soc, progress=True, round_callback=callback)
    signals = avgi[0][0] + 1j * avgq[0][0]

    if instant_show:
        update_show(fig, ax, dh, curve, np.abs(signals), ts)
        clear_show()

    return ts, signals


def measure_t2echo(soc, soccfg, cfg, instant_show=False):
    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

    ts = safe_sweep2array(soccfg, cfg["sweep"])

    if instant_show:
        fig, ax, dh, curve = init_show(2 * ts, "Time (us)", "Amplitude")

        def callback(ir, avg_d):
            avgi, avgq = avg_d[0][0, :, 0], avg_d[0][0, :, 1]
            update_show(fig, ax, dh, curve, np.abs(avgi + 1j * avgq))
    else:
        callback = None

    prog = T2EchoProgram(soccfg, deepcopy(cfg))
    ts, avgi, avgq = prog.acquire(soc, progress=True, round_callback=callback)
    signals = avgi[0][0] + 1j * avgq[0][0]

    if instant_show:
        update_show(fig, ax, dh, curve, np.abs(signals), 2 * ts)
        clear_show()

    return 2 * ts, signals
