from copy import deepcopy

import numpy as np
from tqdm.auto import tqdm

from zcu_tools import make_cfg
from zcu_tools.analysis import NormalizeData
from zcu_tools.program.v1 import (
    RFreqTwoToneProgram,
    RFreqTwoToneProgramWithRedReset,
    TwoToneProgram,
)
from zcu_tools.schedule.flux import set_flux
from zcu_tools.schedule.instant_show import clear_show, init_show2d, update_show2d
from zcu_tools.schedule.tools import sweep2array


def measure_qub_flux_dep(
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

    if cfg["dev"]["flux_dev"] == "none":
        raise NotImplementedError("Flux sweep but get flux_dev == 'none'")

    freq_cfg = cfg["sweep"]["freq"]
    flux_cfg = cfg["sweep"]["flux"]
    fpts = sweep2array(freq_cfg, soft_loop, "Custom frequency sweep only for soft loop")
    flxs = sweep2array(flux_cfg)

    qub_pulse = cfg["dac"]["qub_pulse"]

    flux_tqdm = tqdm(flxs, desc="Flux", smoothing=0)

    if instant_show:
        fig, ax, dh, im = init_show2d(flxs, fpts, "Flux", "Frequency (MHz)")

    if soft_loop:
        print("Use TwoToneProgram for soft loop")
        freq_tqdm = tqdm(fpts, desc="Frequency", smoothing=0)
    else:
        if conjugate_reset:
            print("Use RFreqTwoToneProgramWithRedReset for hard loop")
        else:
            print("Use RFreqTwoToneProgram for hard loop")
        cfg["sweep"] = cfg["sweep"]["freq"]

    signals2D = np.full((len(flxs), len(fpts)), np.nan, dtype=np.complex128)
    try:
        for i, flx in enumerate(flux_tqdm):
            cfg["flux"] = flx
            set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

            if soft_loop:
                freq_tqdm.reset()
                freq_tqdm.refresh()
                for j, f in enumerate(fpts):
                    fpt = float(f)
                    qub_pulse["freq"] = fpt
                    if conjugate_reset:
                        cfg["dac"]["reset_pulse"]["freq"] = r_f - fpt

                    prog = TwoToneProgram(soccfg, make_cfg(cfg))
                    avgi, avgq = prog.acquire(soc, progress=False)
                    signals2D[i, j] = avgi[0][0] + 1j * avgq[0][0]
                    freq_tqdm.update()
            else:
                if conjugate_reset:
                    cfg["r_f"] = r_f
                    prog = RFreqTwoToneProgramWithRedReset(soccfg, make_cfg(cfg))
                    fpts, avgi, avgq = prog.acquire(soc, progress=True)
                    signals2D[i] = avgi[0][0] + 1j * avgq[0][0]
                else:
                    prog = RFreqTwoToneProgram(soccfg, make_cfg(cfg))
                    fpts, avgi, avgq = prog.acquire(soc, progress=True)
                    signals2D[i] = avgi[0][0] + 1j * avgq[0][0]

            if instant_show:
                amps = NormalizeData(signals2D, axis=1, rescale=True) ** 1.5
                update_show2d(fig, ax, dh, im, amps.T)

        if instant_show:
            clear_show(fig, dh)
    except Exception as e:
        print("Error during measurement:", e)

    return fpts, flxs, signals2D
