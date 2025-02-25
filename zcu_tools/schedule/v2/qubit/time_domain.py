import numpy as np

from zcu_tools import make_cfg
from zcu_tools.program.v2 import T1Program, T2EchoProgram, T2RamseyProgram
from zcu_tools.schedule.tools import format_sweep1D, sweep2array, sweep2param
from zcu_tools.schedule.v2.template import sweep1D_soft_template, sweep_hard_template


def measure_t2ramsey(soc, soccfg, cfg, instant_show=False):
    cfg = make_cfg(cfg)  # prevent in-place modification

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")

    sweep_cfg = cfg["sweep"]["length"]
    ts = sweep2array(sweep_cfg, allow_array=True)  # predicted times

    kwargs = dict(
        init_signals=np.full(len(ts), np.nan, dtype=complex),
        instant_show=instant_show,
        progress=True,
        xlabel="Time (us)",
        ylabel="Amplitude",
    )
    if isinstance(sweep_cfg, dict):
        # linear hard sweep
        cfg["dac"]["t2r_length"] = sweep2param("length", sweep_cfg)
        prog, signals = sweep_hard_template(
            soc, soccfg, cfg, T2RamseyProgram, ticks=(ts,), **kwargs
        )

        # get the actual times
        ts = prog.get_time_param("t2ramsey_length", "t", as_array=True)

    elif isinstance(sweep_cfg, np.ndarray) or isinstance(sweep_cfg, list):
        # custom soft sweep
        del cfg["sweep"]  # program should not use this

        cfg["dac"]["t2r_length"] = ts[0]  # initial value

        def updateCfg(cfg, i, t):
            cfg["dac"]["t2r_length"] = t

        ts, signals = sweep1D_soft_template(
            soc, soccfg, cfg, T2RamseyProgram, xs=ts, updateCfg=updateCfg, **kwargs
        )

    return ts, signals


def measure_t2echo(soc, soccfg, cfg, instant_show=False):
    cfg = make_cfg(cfg)  # prevent in-place modification

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")

    sweep_cfg = cfg["sweep"]["length"]
    ts = 2 * sweep2array(sweep_cfg)  # predicted times

    kwargs = dict(
        init_signals=np.full(len(ts), np.nan, dtype=complex),
        instant_show=instant_show,
        progress=True,
        xlabel="Time (us)",
        ylabel="Amplitude",
    )
    if isinstance(sweep_cfg, dict):
        # linear hard sweep
        cfg["dac"]["t2e_half"] = sweep2param("length", sweep_cfg)
        prog, signals = sweep_hard_template(
            soc, soccfg, cfg, T2EchoProgram, ticks=(ts,), **kwargs
        )

        # get the actual times
        ts = 2 * prog.get_time_param("t2e_half", "t", as_array=True)

    elif isinstance(sweep_cfg, np.ndarray) or isinstance(sweep_cfg, list):
        # custom soft sweep
        del cfg["sweep"]  # program should not use this

        cfg["dac"]["t2e_half"] = ts[0] / 2  # initial value

        def updateCfg(cfg, i, t):
            cfg["dac"]["t2e_half"] = t / 2  # half the time

        ts, signals = sweep1D_soft_template(
            soc, soccfg, cfg, T2EchoProgram, xs=2 * ts, updateCfg=updateCfg, **kwargs
        )

    return ts, signals


def measure_t1(soc, soccfg, cfg, instant_show=False):
    cfg = make_cfg(cfg)  # prevent in-place modification

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")

    sweep_cfg = cfg["sweep"]["length"]
    ts = sweep2array(sweep_cfg)  # predicted times

    kwargs = dict(
        init_signals=np.full(len(ts), np.nan, dtype=complex),
        instant_show=instant_show,
        progress=True,
        xlabel="Time (us)",
        ylabel="Amplitude",
    )
    if isinstance(sweep_cfg, dict):
        # linear hard sweep
        cfg["dac"]["t1_length"] = sweep2param("length", sweep_cfg)
        prog, signals = sweep_hard_template(
            soc, soccfg, cfg, T1Program, ticks=(ts,), **kwargs
        )

        # get the actual times
        ts = prog.get_time_param("t1_length", "t", as_array=True)

    elif isinstance(sweep_cfg, np.ndarray) or isinstance(sweep_cfg, list):
        # custom soft sweep
        del cfg["sweep"]

        cfg["dac"]["t1_length"] = ts[0]  # initial value

        def updateCfg(cfg, i, t):
            cfg["dac"]["t1_length"] = t

        ts, signals = sweep1D_soft_template(
            soc, soccfg, cfg, T1Program, xs=ts, updateCfg=updateCfg, **kwargs
        )

    return ts, signals
