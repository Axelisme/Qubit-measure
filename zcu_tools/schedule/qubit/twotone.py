from copy import deepcopy

import numpy as np
from tqdm.auto import tqdm

from zcu_tools import make_cfg
from zcu_tools.program import (
    RFreqTwoToneProgram,
    RFreqTwoToneProgramWithRedReset,
    TwoToneProgram,
)

from ..tools import sweep2array
from ..flux import set_flux
from ..instant_show import clear_show, init_show, update_show


def measure_qub_freq(
    soc,
    soccfg,
    cfg,
    instant_show=False,
    soft_loop=False,
    conjugate_reset=False,
    r_f=None,
):
    cfg = deepcopy(cfg)  # prevent in-place modification

    if conjugate_reset:
        assert r_f is not None, "Need resonator frequency for conjugate reset"
        assert cfg.get("reset") == "pulse", "Need reset=pulse for conjugate reset"
        assert "reset_pulse" in cfg["dac"], "Need reset_pulse for conjugate reset"

    set_flux(cfg["dev"]["flux_dev"], cfg["flux"])

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
            fpt = float(fpt)
            qub_pulse["freq"] = fpt
            if conjugate_reset:
                cfg["dac"]["reset_pulse"]["freq"] = r_f - fpt

            prog = TwoToneProgram(soccfg, make_cfg(cfg))
            avgi, avgq = prog.acquire(soc, progress=False)
            signals[i] = avgi[0][0] + 1j * avgq[0][0]

            if instant_show and i % show_period == 0:
                update_show(fig, ax, dh, curve, np.abs(signals))
        else:
            if instant_show:
                update_show(fig, ax, dh, curve, np.abs(signals))

    else:
        show_period = int(cfg["soft_avgs"] / 10 + 0.9999)

        if conjugate_reset:
            cfg["r_f"] = r_f
            print("Use RFreqTwoToneProgramWithRedReset for hard loop")
            prog = RFreqTwoToneProgramWithRedReset(soccfg, make_cfg(cfg))
        else:
            print("Use RFreqTwoToneProgram for hard loop")
            prog = RFreqTwoToneProgram(soccfg, make_cfg(cfg))

        if instant_show:

            def callback(ir, avg_d):
                if ir % show_period == 0:
                    avgi, avgq = avg_d[0, 0, :, 0], avg_d[0, 0, :, 1]
                    update_show(fig, ax, dh, curve, np.abs(avgi + 1j * avgq))
        else:
            callback = None

        fpts, avgi, avgq = prog.acquire(soc, progress=True, round_callback=callback)
        signals = avgi[0][0] + 1j * avgq[0][0]

        if instant_show:
            update_show(fig, ax, dh, curve, np.abs(signals))

    if instant_show:
        clear_show()

    return fpts, signals
