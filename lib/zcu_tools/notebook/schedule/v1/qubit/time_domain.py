from copy import deepcopy

import numpy as np
from zcu_tools.notebook.single_qubit.process import minus_background
from zcu_tools.program.v1 import T1Program, T2EchoProgram, T2RamseyProgram

from ...tools import check_time_sweep, sweep2array
from ..template import sweep1D_hard_template


def signals2real(signals: np.ndarray) -> np.ndarray:
    return np.abs(minus_background(signals))


def measure_t2ramsey(soc, soccfg, cfg, detune=0.0):
    cfg = deepcopy(cfg)

    ts = sweep2array(cfg["sweep"])
    check_time_sweep(soccfg, ts)

    cfg["detune"] = detune

    # linear hard sweep
    ts, signals = sweep1D_hard_template(
        soc,
        soccfg,
        cfg,
        T2RamseyProgram,
        xs=ts,
        xlabel="Time (us)",
        ylabel="Amplitude",
        signal2real=signals2real,
    )

    return ts, signals


def measure_t1(soc, soccfg, cfg):
    ts = sweep2array(cfg["sweep"])
    check_time_sweep(soccfg, ts)

    ts, signals = sweep1D_hard_template(
        soc,
        soccfg,
        cfg,
        T1Program,
        xs=ts,
        xlabel="Time (us)",
        ylabel="Amplitude",
        signal2real=signals2real,
    )

    return ts, signals


def measure_t2echo(soc, soccfg, cfg, detune=0.0):
    ts = sweep2array(cfg["sweep"])
    check_time_sweep(soccfg, ts)

    cfg["detune"] = detune

    ts, signals = sweep1D_hard_template(
        soc,
        soccfg,
        cfg,
        T2EchoProgram,
        xs=2 * ts,
        xlabel="Time (us)",
        ylabel="Amplitude",
        signal2real=signals2real,
    )

    return 2 * ts, signals
