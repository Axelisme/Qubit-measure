import numpy as np

from zcu_tools.analysis import NormalizeData
from zcu_tools.program.v1 import T1Program, T2EchoProgram, T2RamseyProgram
from zcu_tools.schedule.tools import check_time_sweep, sweep2array
from zcu_tools.schedule.v1.template import sweep1D_hard_template


def safe_sweep2array(soccfg, sweep_cfg):
    ts = sweep2array(
        sweep_cfg, soft_loop=False, err_str="Custom time sweep only for soft loop"
    )
    check_time_sweep(soccfg, ts)

    return ts


def measure_t2ramsey(soc, soccfg, cfg, instant_show=False):
    ts = safe_sweep2array(soccfg, cfg["sweep"])

    ts, signals = sweep1D_hard_template(
        soc,
        soccfg,
        cfg,
        T2RamseyProgram,
        init_xs=ts,
        init_signals=np.full(len(ts), np.nan, dtype=np.complex128),
        progress=True,
        instant_show=instant_show,
        signal2amp=lambda x: NormalizeData(x, rescale=False),
        xlabel="Time (us)",
        ylabel="Amplitude",
    )

    return ts, signals


def measure_t1(soc, soccfg, cfg, instant_show=False):
    ts = safe_sweep2array(soccfg, cfg["sweep"])

    ts, signals = sweep1D_hard_template(
        soc,
        soccfg,
        cfg,
        T1Program,
        init_xs=ts,
        init_signals=np.full(len(ts), np.nan, dtype=np.complex128),
        progress=True,
        instant_show=instant_show,
        signal2amp=lambda x: NormalizeData(x, rescale=False),
        xlabel="Time (us)",
        ylabel="Amplitude",
    )

    return ts, signals


def measure_t2echo(soc, soccfg, cfg, instant_show=False):
    ts = safe_sweep2array(soccfg, cfg["sweep"])

    ts, signals = sweep1D_hard_template(
        soc,
        soccfg,
        cfg,
        T2EchoProgram,
        init_xs=2 * ts,
        init_signals=np.full(len(ts), np.nan, dtype=np.complex128),
        progress=True,
        instant_show=instant_show,
        signal2amp=lambda x: NormalizeData(x, rescale=False),
        xlabel="Time (us)",
        ylabel="Amplitude",
    )

    return 2 * ts, signals
