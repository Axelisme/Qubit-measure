from copy import deepcopy

import numpy as np

from zcu_tools.analysis import minus_background
from zcu_tools.program.v1 import RFreqTwoToneProgram, RFreqTwoToneProgramWithRedReset
from zcu_tools.schedule.tools import sweep2array
from zcu_tools.schedule.v1.template import sweep2D_soft_soft_template


def signal2real(signals):
    return np.abs(minus_background(signals, axis=1))


def measure_qub_flux_dep(soc, soccfg, cfg, reset_rf=None):
    cfg = deepcopy(cfg)  # prevent in-place modification

    if cfg["dev"]["flux_dev"] == "none":
        raise ValueError("Flux sweep but get flux_dev == 'none'")

    if reset_rf is not None:
        assert cfg.get("reset") == "pulse", "Need reset=pulse for conjugate reset"
        assert "reset_pulse" in cfg["dac"], "Need reset_pulse for conjugate reset"
        cfg["r_f"] = reset_rf

    fpt_sweep = cfg["sweep"]["freq"]
    flx_sweep = cfg["sweep"]["flux"]
    fpts = sweep2array(fpt_sweep, allow_array=True)
    flxs = sweep2array(flx_sweep, allow_array=True)

    del cfg["sweep"]  # remove sweep for program use

    def x_updateCfg(cfg, _, flx):
        cfg["dev"]["flux"] = flx

    def y_updateCfg(cfg, _, fpt):
        cfg["dac"]["qub_pulse"]["freq"] = fpt

    flxs, fpts, signals2D = sweep2D_soft_soft_template(
        soc,
        soccfg,
        cfg,
        RFreqTwoToneProgram if reset_rf is None else RFreqTwoToneProgramWithRedReset,
        xs=flxs,
        ys=fpts,
        x_updateCfg=x_updateCfg,
        y_updateCfg=y_updateCfg,
        xlabel="Flux (a.u.)",
        ylabel="Frequency (MHz)",
        signal2real=signal2real,
    )

    return flxs, fpts, signals2D
