from copy import deepcopy
from typing import Tuple

from numpy import ndarray
from zcu_tools.liveplot.jupyter import LivePlotter1D
from zcu_tools.notebook.single_qubit.process import rotate2real
from zcu_tools.program.v1 import T1Program, T2EchoProgram, T2RamseyProgram

from ...tools import check_time_sweep, sweep2array
from ..template import sweep1D_hard_template


def signals2real(signals: ndarray) -> ndarray:
    return rotate2real(signals).real


def measure_t2ramsey(soc, soccfg, cfg, detune=0.0) -> Tuple[ndarray, ndarray]:
    cfg = deepcopy(cfg)

    ts = sweep2array(cfg["sweep"])
    check_time_sweep(soccfg, ts)

    cfg["detune"] = detune

    def measure_fn(cfg, callback) -> Tuple[ndarray, ...]:
        prog = T2RamseyProgram(soccfg, cfg)
        return prog.acquire(soc, progress=True, callback=callback)

    # linear hard sweep
    ts, signals = sweep1D_hard_template(
        cfg,
        measure_fn,
        LivePlotter1D("Time (us)", "Amplitude"),
        xs=ts,
        signal2real=signals2real,
    )

    return ts, signals


def measure_t1(soc, soccfg, cfg, backend_mode=False) -> Tuple[ndarray, ndarray]:
    ts = sweep2array(cfg["sweep"])
    check_time_sweep(soccfg, ts)

    def measure_fn(cfg, callback) -> Tuple[ndarray, ...]:
        prog = T1Program(soccfg, cfg)
        return prog.acquire(soc, progress=not backend_mode, callback=callback)

    ts, signals = sweep1D_hard_template(
        cfg,
        measure_fn,
        LivePlotter1D("Time (us)", "Amplitude", disable=backend_mode),
        xs=ts,
        signal2real=signals2real,
    )

    return ts, signals


def measure_t2echo(soc, soccfg, cfg, detune=0.0) -> Tuple[ndarray, ndarray]:
    ts = sweep2array(cfg["sweep"])
    check_time_sweep(soccfg, ts)

    cfg["detune"] = detune

    def measure_fn(cfg, callback) -> Tuple[ndarray, ...]:
        prog = T2EchoProgram(soccfg, cfg)
        return prog.acquire(soc, progress=True, callback=callback)

    ts, signals = sweep1D_hard_template(
        cfg,
        measure_fn,
        LivePlotter1D("Time (us)", "Amplitude"),
        xs=2 * ts,
        signal2real=signals2real,
    )

    return 2 * ts, signals
