from copy import deepcopy

import numpy as np
from zcu_tools.analysis import minus_background
from zcu_tools.program.v1 import RFreqTwoToneProgram, RFreqTwoToneProgramWithRedReset
from zcu_tools.schedule.tools import sweep2array
from zcu_tools.schedule.v1.template import sweep2D_soft_hard_template
from zcu_tools.schedule.v1.qubit.twotone import qub_signal2snr


def signal2real(signals):
    return np.abs(minus_background(signals, axis=1))


def measure_qub_flux_dep(soc, soccfg, cfg, reset_rf=None, earlystop_snr=None):
    cfg = deepcopy(cfg)  # prevent in-place modification

    if cfg["dev"]["flux_dev"] == "none":
        raise ValueError("Flux sweep but get flux_dev == 'none'")

    if reset_rf is not None:
        assert cfg.get("reset") == "pulse", "Need reset=pulse for conjugate reset"
        assert "reset_pulse" in cfg["dac"], "Need reset_pulse for conjugate reset"
        cfg["r_f"] = reset_rf

    flx_sweep = cfg["sweep"]["flux"]
    fpt_sweep = cfg["sweep"]["freq"]
    flxs = sweep2array(flx_sweep, allow_array=True)
    fpts = sweep2array(fpt_sweep)

    cfg["sweep"] = cfg["sweep"]["freq"]  # change sweep to freq

    def updateCfg(cfg, _, flx):
        cfg["dev"]["flux"] = flx

    if earlystop_snr is not None:

        def checker(signals):
            snr = qub_signal2snr(signals)
            return snr >= earlystop_snr, f"Current SNR: {snr:.2g}"

    else:
        checker = None

    flxs, fpts, signals2D = sweep2D_soft_hard_template(
        soc,
        soccfg,
        cfg,
        RFreqTwoToneProgram if reset_rf is None else RFreqTwoToneProgramWithRedReset,
        xs=flxs,
        ys=fpts,
        updateCfg=updateCfg,
        xlabel="Flux (a.u.)",
        ylabel="Frequency (MHz)",
        signal2real=signal2real,
        early_stop_checker=checker,
    )

    return flxs, fpts, signals2D
