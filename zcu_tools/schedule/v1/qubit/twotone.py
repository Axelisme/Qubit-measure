from copy import deepcopy

import numpy as np

from zcu_tools import make_cfg
from zcu_tools.program.v1 import RFreqTwoToneProgram, RFreqTwoToneProgramWithRedReset
from zcu_tools.schedule.flux import set_flux
from zcu_tools.schedule.instant_show import clear_show, init_show, update_show
from zcu_tools.schedule.tools import sweep2array


def measure_qub_freq(soc, soccfg, cfg, instant_show=False, reset_rf=None):
    cfg = deepcopy(cfg)  # prevent in-place modification

    if reset_rf is not None:
        cfg["r_f"] = reset_rf
        assert cfg.get("reset") == "pulse", "Need reset=pulse for conjugate reset"
        assert "reset_pulse" in cfg["dac"], "Need reset_pulse for conjugate reset"

    fpts = sweep2array(cfg["sweep"], False, "Custom frequency sweep only for soft loop")

    if instant_show:
        fig, ax, dh, curve = init_show(fpts, "Frequency (MHz)", "Amplitude")

        def callback(ir, sum_d):
            amps = np.abs(sum_d[0][0].dot([1, 1j]) / (ir + 1))
            update_show(fig, ax, dh, curve, amps)
    else:
        callback = None  # type: ignore

    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

    prog_cls = (
        RFreqTwoToneProgram if reset_rf is None else RFreqTwoToneProgramWithRedReset
    )
    prog = prog_cls(soccfg, make_cfg(cfg))
    fpts, avgi, avgq = prog.acquire(soc, progress=True, round_callback=callback)
    signals = avgi[0][0] + 1j * avgq[0][0]

    if instant_show:
        update_show(fig, ax, dh, curve, np.abs(signals))
        clear_show(fig, dh)

    return fpts, signals
