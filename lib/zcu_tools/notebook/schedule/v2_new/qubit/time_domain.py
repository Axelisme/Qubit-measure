# type: ignore

from typing import Optional, Tuple

import numpy as np
from zcu_tools.auto import make_cfg
from zcu_tools.notebook.single_qubit.process import rotate2real
from zcu_tools.program.v2 import T1Program, T2EchoProgram, T2RamseyProgram
from zcu_tools.program.v2.base.simulate import SimulateProgramV2
from zcu_tools.liveplot.jupyter import LivePlotter1D

from ...tools import format_sweep1D, sweep2array, sweep2param
from ..template import sweep_hard_template


def qub_signals2reals(signals):
    return rotate2real(signals).real


def measure_t2ramsey(
    soc, soccfg, cfg, detune: float = 0
) -> Tuple[np.ndarray, np.ndarray]:
    cfg = make_cfg(cfg)  # prevent in-place modification

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")
    sweep_cfg = cfg["sweep"]["length"]

    cfg["detune"] = detune
    cfg["dac"]["t2r_length"] = sweep2param("length", sweep_cfg)

    ts = sweep2array(sweep_cfg)  # predicted times

    prog: Optional[T2RamseyProgram] = None

    def measure_fn(cfg, callback) -> list:
        nonlocal prog
        prog = T2RamseyProgram(soccfg, cfg)
        return prog.acquire(soc, progress=True, callback=callback)

    # linear hard sweep
    signals = sweep_hard_template(
        cfg,
        measure_fn,
        LivePlotter1D("Time (us)", "Amplitude"),
        ticks=(ts,),
        signal2real=qub_signals2reals,
    )

    # get the actual times
    _ts: np.ndarray = prog.get_time_param("t2r_length", "t", as_array=True)
    # TODO: check if this is correct
    ts = _ts + ts[0] - _ts[0]  # adjust to start from the first time

    return ts, signals


def measure_t2echo(
    soc, soccfg, cfg, detune: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    cfg = make_cfg(cfg)  # prevent in-place modification

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")
    sweep_cfg = cfg["sweep"]["length"]

    cfg["detune"] = detune
    cfg["dac"]["t2e_half"] = sweep2param("length", sweep_cfg)

    ts = (
        2 * sweep2array(sweep_cfg) + cfg["dac"]["pi_pulse"]["length"]
    )  # predicted times

    prog: Optional[T2EchoProgram] = None

    def measure_fn(cfg, callback) -> list:
        nonlocal prog
        prog = T2EchoProgram(soccfg, cfg)
        return prog.acquire(soc, progress=True, callback=callback)

    # linear hard sweep
    signals = sweep_hard_template(
        cfg,
        measure_fn,
        LivePlotter1D("Time (us)", "Amplitude"),
        ticks=(ts,),
        signal2real=qub_signals2reals,
    )

    # get the actual times
    _ts: np.ndarray = (
        2 * prog.get_time_param("t2e_half", "t", as_array=True)
        + cfg["dac"]["pi_pulse"]["length"]
    )  # type: ignore
    # TODO: check if this is correct
    ts = _ts + ts[0] - _ts[0]  # adjust to start from the first time

    return ts, signals


def measure_t1(
    soc, soccfg, cfg, backend_mode: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    cfg = make_cfg(cfg)  # prevent in-place modification

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")
    sweep_cfg = cfg["sweep"]["length"]

    cfg["dac"]["t1_length"] = sweep2param("length", sweep_cfg)

    ts = sweep2array(sweep_cfg)  # predicted times

    prog: Optional[T1Program] = None

    def measure_fn(cfg, callback) -> list:
        nonlocal prog
        prog = T1Program(soccfg, cfg)
        return prog.acquire(soc, progress=not backend_mode, callback=callback)

    # linear hard sweep
    signals = sweep_hard_template(
        cfg,
        measure_fn,
        LivePlotter1D("Time (us)", "Amplitude", disable=backend_mode),
        ticks=(ts,),
        signal2real=qub_signals2reals,
    )

    # get the actual times
    _ts: np.ndarray = prog.get_time_param("t1_length", "t", as_array=True)
    # TODO: check if this is correct
    ts = _ts + ts[0] - _ts[0]  # adjust to start from the first time

    return ts, signals


def visualize_t2ramsey(
    soccfg, cfg, detune: float = 0.0, *, time_fly: float = 0.0
) -> None:
    cfg = make_cfg(cfg)  # prevent in-place modification

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")
    sweep_cfg = cfg["sweep"]["length"]

    cfg["detune"] = detune
    cfg["dac"]["t2r_length"] = sweep2param("length", sweep_cfg)

    visualizer = SimulateProgramV2(T2RamseyProgram, soccfg, cfg)
    visualizer.visualize(time_fly=time_fly)


def visualize_t2echo(
    soccfg, cfg, detune: float = 0.0, *, time_fly: float = 0.0
) -> None:
    cfg = make_cfg(cfg)  # prevent in-place modification

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")
    sweep_cfg = cfg["sweep"]["length"]

    cfg["detune"] = detune
    cfg["dac"]["t2e_half"] = sweep2param("length", sweep_cfg)

    visualizer = SimulateProgramV2(T2EchoProgram, soccfg, cfg)
    visualizer.visualize(time_fly=time_fly)


def visualize_t1(soccfg, cfg, *, time_fly: float = 0.0) -> None:
    cfg = make_cfg(cfg)  # prevent in-place modification

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")
    cfg["dac"]["t1_length"] = sweep2param("length", cfg["sweep"]["length"])

    visualizer = SimulateProgramV2(T1Program, soccfg, cfg)
    visualizer.visualize(time_fly=time_fly)
