import numpy as np

from zcu_tools import make_cfg
from zcu_tools.analysis import NormalizeData
from zcu_tools.program.v2 import T1Program, T2EchoProgram, T2RamseyProgram
from zcu_tools.schedule.tools import format_sweep1D, sweep2array, sweep2param
from zcu_tools.schedule.v2.template import sweep_template


def measure_t2ramsey(soc, soccfg, cfg, instant_show=False):
    cfg = make_cfg(cfg)  # prevent in-place modification

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")

    sweep_cfg = cfg["sweep"]["length"]
    cfg["dac"]["ramsey_length"] = sweep2param("length", sweep_cfg)

    ts = sweep2array(sweep_cfg, False)  # predicted times

    prog, signals = sweep_template(
        soc,
        soccfg,
        cfg,
        T2RamseyProgram,
        init_signals=np.full(len(ts), np.nan, dtype=complex),
        ticks=(ts,),
        progress=True,
        instant_show=instant_show,
        signal2amp=lambda x: NormalizeData(x, rescale=False),
        xlabel="Time (us)",
        ylabel="Amplitude",
    )

    # get the actual times
    ts = prog.get_time_param("t2ramsey_length", "t", as_array=True)

    return ts, signals


def measure_t2echo(soc, soccfg, cfg, instant_show=False):
    cfg = make_cfg(cfg)  # prevent in-place modification

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")

    sweep_cfg = cfg["sweep"]["length"]
    cfg["dac"]["ramsey_length"] = sweep2param("length", sweep_cfg)

    ts = 2 * sweep2array(sweep_cfg, False)  # predicted times

    prog, signals = sweep_template(
        soc,
        soccfg,
        cfg,
        T2EchoProgram,
        init_signals=np.full(len(ts), np.nan, dtype=complex),
        ticks=(ts,),
        progress=True,
        instant_show=instant_show,
        signal2amp=lambda x: NormalizeData(x, rescale=False),
        xlabel="Time (us)",
        ylabel="Amplitude",
    )

    # get the actual times
    ts = 2 * prog.get_time_param("t2echo_length", "t", as_array=True)

    return ts, signals


def measure_t1(soc, soccfg, cfg, instant_show=False):
    cfg = make_cfg(cfg)  # prevent in-place modification

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")

    sweep_cfg = cfg["sweep"]["length"]
    cfg["dac"]["t1_length"] = sweep2param("length", sweep_cfg)

    ts = sweep2array(sweep_cfg, False)  # predicted times

    prog, signals = sweep_template(
        soc,
        soccfg,
        cfg,
        T1Program,
        init_signals=np.full(len(ts), np.nan, dtype=complex),
        ticks=(ts,),
        progress=True,
        instant_show=instant_show,
        signal2amp=lambda x: NormalizeData(x, rescale=False),
        xlabel="Time (us)",
        ylabel="Amplitude",
    )

    # get the actual times
    ts = prog.get_time_param("t1_length", "t", as_array=True)

    return ts, signals
