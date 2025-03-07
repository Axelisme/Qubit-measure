import numpy as np

from zcu_tools import make_cfg
from zcu_tools.analysis import minus_background
from zcu_tools.program.v2 import TwoToneProgram
from zcu_tools.schedule.tools import sweep2array, sweep2param
from zcu_tools.schedule.v2.template import sweep_hard_template


def signals2reals(signals: np.ndarray) -> np.ndarray:
    return np.abs(minus_background(signals, axis=0))


def measure_qub_pdr_dep(soc, soccfg, cfg):
    cfg = make_cfg(cfg)  # prevent in-place modification

    # make sure gain is the outer loop
    if list(cfg["sweep"].keys())[0] == "freq":
        cfg["sweep"] = {
            "gain": cfg["sweep"]["gain"],
            "freq": cfg["sweep"]["freq"],
        }

    qub_pulse = cfg["dac"]["qub_pulse"]
    qub_pulse["gain"] = sweep2param("gain", cfg["sweep"]["gain"])
    qub_pulse["freq"] = sweep2param("freq", cfg["sweep"]["freq"])

    pdrs = sweep2array(cfg["sweep"]["gain"])  # predicted pulse gains
    fpts = sweep2array(cfg["sweep"]["freq"])  # predicted frequency points

    prog, signals = sweep_hard_template(
        soc,
        soccfg,
        cfg,
        TwoToneProgram,
        init_signals=np.full((len(pdrs), len(fpts)), np.nan, dtype=complex),
        ticks=(fpts, pdrs),
        progress=True,
        xlabel="Frequency (MHz)",
        ylabel="Pulse Gain",
        signal2real=signals2reals,
    )

    # get the actual pulse gains and frequency points
    pdrs = prog.get_pulse_param("qub_pulse", "gain", as_array=True)
    fpts = prog.get_pulse_param("qub_pulse", "freq", as_array=True)

    return pdrs, fpts, signals  # (pdrs, fpts)
