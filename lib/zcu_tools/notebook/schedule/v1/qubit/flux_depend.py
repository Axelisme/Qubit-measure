from copy import deepcopy

import numpy as np
from zcu_tools.notebook.single_qubit.process import minus_background
from zcu_tools.program.v1 import RFreqTwoToneProgram

from ...tools import sweep2array
from ..template import sweep2D_soft_hard_template
from .twotone import qub_signal2snr


def signal2real(signals):
    return np.abs(minus_background(signals, axis=1))


def measure_qub_flux_dep(soc, soccfg, cfg, earlystop_snr=None):
    cfg = deepcopy(cfg)  # prevent in-place modification

    if cfg["dev"]["flux_dev"] == "none":
        raise ValueError("Flux sweep but get flux_dev == 'none'")

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
        RFreqTwoToneProgram,
        xs=flxs,
        ys=fpts,
        updateCfg=updateCfg,
        xlabel="Flux (a.u.)",
        ylabel="Frequency (MHz)",
        signal2real=signal2real,
        early_stop_checker=checker,
    )

    return flxs, fpts, signals2D
