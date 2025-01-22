from copy import deepcopy

import numpy as np
from tqdm.auto import tqdm

from zcu_tools import make_cfg
from zcu_tools.program2 import (
    RFreqTwoToneProgram,
    RFreqTwoToneProgramWithRedReset,
    TwoToneProgram,
)

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
    sub_ground=False,
):
    cfg = deepcopy(cfg)  # prevent in-place modification

    if conjugate_reset:
        assert r_f is not None, "Need resonator frequency for conjugate reset"
        assert cfg.get("reset") == "pulse", "Need reset=pulse for conjugate reset"
        assert "reset_pulse" in cfg["dac"], "Need reset_pulse for conjugate reset"

    set_flux(cfg["flux_dev"], cfg["flux"])

    sweep_cfg = cfg["sweep"]
    if isinstance(sweep_cfg, dict):
        fpts = np.linspace(sweep_cfg["start"], sweep_cfg["stop"], sweep_cfg["expts"])
    else:
        assert soft_loop, "Hard loop only supports linear sweep"
        fpts = np.array(sweep_cfg)

    qub_pulse = cfg["dac"]["qub_pulse"]
    if soft_loop:
        print("Use TwoToneProgram for soft loop")

        show_period = int(len(fpts) / 10 + 0.99)
        if instant_show:
            fig, ax, dh, curve = init_show(fpts, "Frequency (MHz)", "Amplitude")

        signals = np.full(len(fpts), np.nan, dtype=np.complex128)
        for i, fpt in enumerate(tqdm(fpts, desc="Frequency", smoothing=0)):
            fpt = float(fpt)
            qub_pulse["freq"] = fpt
            if conjugate_reset:
                cfg["dac"]["reset_pulse"]["freq"] = r_f - fpt

            prog = TwoToneProgram(soccfg, make_cfg(cfg))
            avgi, avgq = prog.acquire(soc, progress=False)
            signals[i] = avgi[0][0] + 1j * avgq[0][0]

            if sub_ground:
                g_cfg = make_cfg(cfg)
                g_cfg["dac"]["qub_pulse"]["gain"] = 0  # set gain to 0
                g_prog = TwoToneProgram(soccfg, g_cfg)
                g_avgi, g_avgq = g_prog.acquire(soc, progress=False)
                signals[i] -= g_avgi[0][0] + 1j * g_avgq[0][0]

            if instant_show and i % show_period == 0:
                update_show(fig, ax, dh, curve, np.abs(signals))
        else:
            if instant_show:
                update_show(fig, ax, dh, curve, np.abs(signals))

        if instant_show:
            clear_show()

    else:
        if conjugate_reset:
            cfg["r_f"] = r_f
            print("Use RFreqTwoToneProgramWithRedReset for hard loop")
            prog = RFreqTwoToneProgramWithRedReset(soccfg, make_cfg(cfg))
            fpts, avgi, avgq = prog.acquire(soc, progress=True)
            signals = avgi[0][0] + 1j * avgq[0][0]
            if sub_ground:
                g_cfg = make_cfg(cfg)
                g_cfg["dac"]["qub_pulse"]["gain"] = 0
                g_prog = RFreqTwoToneProgramWithRedReset(soccfg, g_cfg)
                _, g_avgi, g_avgq = g_prog.acquire(soc, progress=False)
                signals -= g_avgi[0][0] + 1j * g_avgq[0][0]
        else:
            print("Use RFreqTwoToneProgram for hard loop")
            prog = RFreqTwoToneProgram(soccfg, make_cfg(cfg))
            fpts, avgi, avgq = prog.acquire(soc, progress=True)
            signals = avgi[0][0] + 1j * avgq[0][0]
        fpts = np.array([prog.reg2freq(f, gen_ch=qub_pulse["ch"]) for f in fpts])

    return fpts, signals
