import numpy as np

from zcu_tools import make_cfg
from zcu_tools.program.v2 import TwoToneProgram
from zcu_tools.schedule.tools import format_sweep1D, sweep2array, sweep2param
from zcu_tools.schedule.v2.template import sweep_template


def measure_lenrabi(soc, soccfg, cfg, instant_show=False):
    cfg = make_cfg(cfg)  # prevent in-place modification

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")

    sweep_cfg = cfg["sweep"]["length"]
    cfg["dac"]["qub_pulse"]["length"] = sweep2param("length", sweep_cfg)

    lens = sweep2array(sweep_cfg, False)  # predicted lengths

    prog, signals = sweep_template(
        soc,
        soccfg,
        cfg,
        TwoToneProgram,
        init_signals=np.full(len(lens), np.nan, dtype=complex),
        ticks=(lens,),
        progress=True,
        instant_show=instant_show,
        xlabel="Length (us)",
        ylabel="Amplitude",
    )

    # get the actual lengths
    lens = prog.get_pulse_param("qub_pulse", "length", as_array=True)

    return lens, signals


def measure_amprabi(soc, soccfg, cfg, instant_show=False):
    cfg = make_cfg(cfg)  # prevent in-place modification

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "gain")

    sweep_cfg = cfg["sweep"]["gain"]
    cfg["dac"]["qub_pulse"]["gain"] = sweep2param("gain", sweep_cfg)

    amps = sweep2array(sweep_cfg, False)  # predicted amplitudes

    prog, signals = sweep_template(
        soc,
        soccfg,
        cfg,
        TwoToneProgram,
        init_signals=np.full(len(amps), np.nan, dtype=complex),
        ticks=(amps,),
        progress=True,
        instant_show=instant_show,
        xlabel="Pulse gain",
        ylabel="Amplitude",
    )

    # get the actual amplitudes
    amps = prog.get_pulse_param("qub_pulse", "gain", as_array=True)

    return amps, signals
