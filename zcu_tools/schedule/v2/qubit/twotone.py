import numpy as np

from zcu_tools import make_cfg
from zcu_tools.program.v2 import TwoToneProgram
from zcu_tools.schedule.tools import format_sweep1D, sweep2array, sweep2param
from zcu_tools.schedule.v2.template import sweep_template


def measure_qub_freq(soc, soccfg, cfg, progress=True, instant_show=False):
    cfg = make_cfg(cfg)  # prevent in-place modification

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "freq")

    sweep_cfg = cfg["sweep"]["freq"]
    cfg["dac"]["qub_pulse"]["freq"] = sweep2param("freq", sweep_cfg)

    fpts = sweep2array(sweep_cfg)  # predicted frequency points

    prog, signals = sweep_template(
        soc,
        soccfg,
        cfg,
        TwoToneProgram,
        init_signals=np.full(len(fpts), np.nan, dtype=complex),
        ticks=(fpts,),
        progress=progress,
        instant_show=instant_show,
        xlabel="Frequency (MHz)",
        ylabel="Amplitude",
    )

    # get the actual frequency points
    fpts = prog.get_pulse_param("qub_pulse", "freq", as_array=True)

    return fpts, signals


def measure_qub_freq_with_reset(
    soc, soccfg, cfg, r_f, progress=True, instant_show=False
):
    cfg = make_cfg(cfg)  # prevent in-place modification

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "freq")

    assert cfg["dac"].get("reset") == "pulse", "Need reset=pulse for conjugate reset"
    assert "reset_pulse" in cfg["dac"], "Need reset_pulse for conjugate reset"

    sweep_cfg = cfg["sweep"]["freq"]
    params = sweep2param("freq", sweep_cfg)
    cfg["dac"]["qub_pulse"]["freq"] = params
    cfg["dac"]["reset_pulse"]["freq"] = r_f - params  # conjugate reset

    fpts = sweep2array(sweep_cfg)  # predicted frequency points

    prog, signals = sweep_template(
        soc,
        soccfg,
        cfg,
        TwoToneProgram,
        init_signals=np.full(len(fpts), np.nan, dtype=complex),
        ticks=(fpts,),
        progress=progress,
        instant_show=instant_show,
        xlabel="Frequency (MHz)",
        ylabel="Amplitude",
    )

    # get the actual frequency points
    fpts = prog.get_pulse_param("qub_pulse", "freq", as_array=True)

    return fpts, signals
