import numpy as np

from zcu_tools import make_cfg
from zcu_tools.analysis import minus_background
from zcu_tools.program.v2 import TwoToneProgram
from zcu_tools.schedule.tools import sweep2array, sweep2param
from zcu_tools.schedule.v2.template import sweep2D_soft_hard_template


def signal2real(signals):
    return np.abs(minus_background(signals, axis=1))


def measure_qub_flux_dep(soc, soccfg, cfg, reset_rf=None):
    cfg = make_cfg(cfg)  # prevent in-place modification

    if cfg["dev"]["flux_dev"] == "none":
        raise ValueError("Flux sweep but get flux_dev == 'none'")

    qub_pulse = cfg["dac"]["qub_pulse"]
    fpt_sweep = cfg["sweep"]["freq"]
    flx_sweep = cfg["sweep"]["flux"]
    qub_pulse["freq"] = sweep2param("freq", fpt_sweep)

    del cfg["sweep"]["flux"]  # use for loop here

    if reset_rf is not None:
        assert cfg["dac"]["reset"] == "pulse", "Need reset=pulse for conjugate reset"
        assert "reset_pulse" in cfg["dac"], "Need reset_pulse for conjugate reset"
        cfg["dac"]["reset_pulse"]["freq"] = reset_rf - qub_pulse["freq"]

    flxs = sweep2array(flx_sweep, allow_array=True)
    fpts = sweep2array(fpt_sweep)

    cfg["dev"]["flux"] = flxs[0]  # set initial flux

    def updateCfg(cfg, i, flx):
        cfg["dev"]["flux"] = flx

    prog, signals2D = sweep2D_soft_hard_template(
        soc,
        soccfg,
        cfg,
        TwoToneProgram,
        xs=flxs,
        ys=fpts,
        xlabel="Flux (a.u.)",
        ylabel="Frequency (MHz)",
        updateCfg=updateCfg,
        signal2real=signal2real,
    )

    # get the actual frequency points
    fpts = prog.get_pulse_param("qub_pulse", "freq", as_array=True)

    return flxs, fpts, signals2D
