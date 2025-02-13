from copy import deepcopy

import numpy as np
from tqdm.auto import tqdm

from zcu_tools import make_cfg
from zcu_tools.program.v1 import (
    RFreqTwoToneProgram,
    RFreqTwoToneProgramWithRedReset,
    TwoToneProgram,
)
from zcu_tools.schedule.flux import set_flux
from zcu_tools.schedule.instant_show import clear_show, init_show, update_show
from zcu_tools.schedule.tools import sweep2array


def measure_qub_freq(
    soc,
    soccfg,
    cfg,
    instant_show=False,
    soft_loop=False,
):
    cfg = deepcopy(cfg)  # prevent in-place modification

    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

    qub_pulse = cfg["dac"]["qub_pulse"]

    fpts = sweep2array(
        cfg["sweep"], soft_loop, "Custom frequency sweep only for soft loop"
    )

    if instant_show:
        fig, ax, dh, curve = init_show(fpts, "Frequency (MHz)", "Amplitude")

    if soft_loop:
        print("Use TwoToneProgram for soft loop")

        show_period = int(len(fpts) / 10 + 0.99)

        signals = np.full(len(fpts), np.nan, dtype=np.complex128)
        for i, fpt in enumerate(tqdm(fpts, desc="Frequency", smoothing=0)):
            qub_pulse["freq"] = float(fpt)

            prog = TwoToneProgram(soccfg, make_cfg(cfg))
            avgi, avgq = prog.acquire(soc, progress=False)
            signals[i] = avgi[0][0] + 1j * avgq[0][0]

            if instant_show and i % show_period == 0:
                update_show(fig, ax, dh, curve, np.abs(signals))

    else:
        show_period = int(cfg["rounds"] / 10 + 0.9999)

        print("Use RFreqTwoToneProgram for hard loop")
        prog = RFreqTwoToneProgram(soccfg, make_cfg(cfg))

        if instant_show:

            def callback(ir, avg_d):
                avgi, avgq = avg_d[0][0, :, 0], avg_d[0][0, :, 1]
                update_show(fig, ax, dh, curve, np.abs(avgi + 1j * avgq))
        else:
            callback = None

        fpts, avgi, avgq = prog.acquire(
            soc, progress=True, round_callback=callback, callback_period=show_period
        )
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
    soft_loop=False,
    r_f=None,
):
    cfg = deepcopy(cfg)  # prevent in-place modification

    assert r_f is not None, "Need resonator frequency for conjugate reset"
    assert cfg["dac"].get("reset") == "pulse", "Need reset=pulse for conjugate reset"
    assert "reset_pulse" in cfg["dac"], "Need reset_pulse for conjugate reset"

    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

    qub_pulse = cfg["dac"]["qub_pulse"]
    reset_pulse = cfg["dac"]["reset_pulse"]

    fpts = sweep2array(
        cfg["sweep"], soft_loop, "Custom frequency sweep only for soft loop"
    )

    if instant_show:
        fig, ax, dh, curve = init_show(fpts, "Frequency (MHz)", "Amplitude")

    if soft_loop:
        print("Use TwoToneProgram for soft loop")

        show_period = int(len(fpts) / 10 + 0.99)

        signals = np.full(len(fpts), np.nan, dtype=np.complex128)
        for i, fpt in enumerate(tqdm(fpts, desc="Frequency", smoothing=0)):
            fpt = float(fpt)
            qub_pulse["freq"] = fpt
            reset_pulse["freq"] = r_f - fpt

            prog = TwoToneProgram(soccfg, make_cfg(cfg))
            avgi, avgq = prog.acquire(soc, progress=False)
            signals[i] = avgi[0][0] + 1j * avgq[0][0]

            if instant_show and i % show_period == 0:
                update_show(fig, ax, dh, curve, np.abs(signals))

    else:
        cfg["r_f"] = r_f
        print("Use RFreqTwoToneProgramWithRedReset for hard loop")
        prog = RFreqTwoToneProgramWithRedReset(soccfg, make_cfg(cfg))

        if instant_show:
            show_period = int(cfg["rounds"] / 10 + 0.9999)

            def callback(ir, avg_d):
                avgi, avgq = avg_d[0][0, :, 0], avg_d[0][0, :, 1]
                update_show(fig, ax, dh, curve, np.abs(avgi + 1j * avgq))
        else:
            callback = None
            show_period = None

        fpts, avgi, avgq = prog.acquire(
            soc, progress=True, round_callback=callback, callback_period=show_period
        )
        signals = avgi[0][0] + 1j * avgq[0][0]

    if instant_show:
        update_show(fig, ax, dh, curve, np.abs(signals))
        clear_show()

    return fpts, signals
