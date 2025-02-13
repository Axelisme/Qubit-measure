from copy import deepcopy

import numpy as np

from zcu_tools import make_cfg
from zcu_tools.program.v1 import RFreqTwoToneProgram, RFreqTwoToneProgramWithRedReset
from zcu_tools.schedule.flux import set_flux
from zcu_tools.schedule.instant_show import clear_show, init_show, update_show
from zcu_tools.schedule.tools import sweep2array


def measure_qub_freq(soc, soccfg, cfg, instant_show=False):
    cfg = deepcopy(cfg)  # prevent in-place modification

    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

    prog = RFreqTwoToneProgram(soccfg, make_cfg(cfg))

    fpts = sweep2array(cfg["sweep"], False, "Custom frequency sweep only for soft loop")

    if instant_show:
        fig, ax, dh, curve = init_show(fpts, "Frequency (MHz)", "Amplitude")

        def callback(ir, avg_d):
            avgi, avgq = avg_d[0][0, :, 0], avg_d[0][0, :, 1]
            update_show(fig, ax, dh, curve, np.abs(avgi + 1j * avgq), fpts)
    else:
        callback = None

    fpts, avgi, avgq = prog.acquire(soc, progress=True, round_callback=callback)
    signals = avgi[0][0] + 1j * avgq[0][0]

    if instant_show:
        update_show(fig, ax, dh, curve, np.abs(signals))
        clear_show()

    return fpts, signals


def measure_qub_freq_with_reset(
    soc,
    soccfg,
    cfg,
    instant_show=False,
    r_f=None,
):
    cfg = deepcopy(cfg)  # prevent in-place modification

    assert r_f is not None, "Need resonator frequency for conjugate reset"
    assert cfg["dac"].get("reset") == "pulse", "Need reset=pulse for conjugate reset"
    assert "reset_pulse" in cfg["dac"], "Need reset_pulse for conjugate reset"

    cfg["r_f"] = r_f
    cfg = make_cfg(cfg)

    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

    prog = RFreqTwoToneProgramWithRedReset(soccfg, cfg)

    if instant_show:
        fpts = sweep2array(cfg["sweep"], False)
        fig, ax, dh, curve = init_show(fpts, "Frequency (MHz)", "Amplitude")

        def callback(ir, avg_d):
            avgi, avgq = avg_d[0][0, :, 0], avg_d[0][0, :, 1]
            update_show(fig, ax, dh, curve, np.abs(avgi + 1j * avgq))
    else:
        callback = None

    fpts, avgi, avgq = prog.acquire(soc, progress=True, round_callback=callback)
    signals = avgi[0][0] + 1j * avgq[0][0]

    if instant_show:
        update_show(fig, ax, dh, curve, np.abs(signals))
        clear_show()

    return fpts, signals
