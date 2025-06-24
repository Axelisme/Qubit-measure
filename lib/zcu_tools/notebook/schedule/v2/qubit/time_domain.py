from copy import deepcopy
from typing import Tuple

import numpy as np
from zcu_tools.liveplot.jupyter import LivePlotter1D
from zcu_tools.notebook.single_qubit.process import rotate2real
from zcu_tools.program.v2 import T1Program, T2EchoProgram, T2RamseyProgram
from zcu_tools.program.v2.base.simulate import SimulateProgramV2

from ...tools import format_sweep1D, sweep2array, sweep2param
from ..template import sweep_hard_template


def qub_signals2reals(signals: np.ndarray) -> np.ndarray:
    return rotate2real(signals).real


def measure_t2ramsey(
    soc, soccfg, cfg, detune: float = 0
) -> Tuple[np.ndarray, np.ndarray]:
    cfg = deepcopy(cfg)  # prevent in-place modification

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")
    sweep_cfg = cfg["sweep"]["length"]

    cfg["pi2_pulse1"] = deepcopy(cfg["pi2_pulse"])
    cfg["pi2_pulse2"] = deepcopy(cfg["pi2_pulse"])

    t2r_spans = sweep2param("length", sweep_cfg)
    cfg["pi2_pulse1"]["post_delay"] = t2r_spans
    cfg["pi2_pulse2"]["phase"] = cfg["pi2_pulse2"]["phase"] + 360 * detune * t2r_spans

    ts = sweep2array(sweep_cfg)  # predicted times

    prog = T2RamseyProgram(soccfg, cfg)

    # linear hard sweep
    signals = sweep_hard_template(
        cfg,
        lambda _, cb: prog.acquire(soc, progress=True, callback=cb),
        LivePlotter1D("Time (us)", "Amplitude"),
        ticks=(ts,),
        signal2real=qub_signals2reals,
    )

    # get the actual times
    real_ts = prog.get_time_param("pi2_pulse1_post_delay", "t", as_array=True)
    # TODO: make sure this is correct
    real_ts += ts[0] - real_ts[0]  # adjust to start from the first time

    return real_ts, signals


def measure_t2echo(
    soc, soccfg, cfg, detune: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    cfg = deepcopy(cfg)  # prevent in-place modification

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")
    sweep_cfg = cfg["sweep"]["length"]

    cfg["pi2_pulse1"] = deepcopy(cfg["pi2_pulse"])
    cfg["pi2_pulse2"] = deepcopy(cfg["pi2_pulse"])

    t2e_half_spans = 0.5 * sweep2param("length", sweep_cfg)
    cfg["pi2_pulse1"]["post_delay"] = t2e_half_spans
    cfg["pi_pulse"]["post_delay"] = t2e_half_spans
    cfg["pi_pulse"]["phase"] = cfg["pi_pulse"]["phase"] + 360 * detune * t2e_half_spans
    cfg["pi2_pulse2"]["phase"] = (
        cfg["pi2_pulse2"]["phase"] + 2 * 360 * detune * t2e_half_spans
    )

    ts = 2 * sweep2array(sweep_cfg) + cfg["pi_pulse"]["length"]  # predicted times

    prog = T2EchoProgram(soccfg, cfg)

    # linear hard sweep
    signals = sweep_hard_template(
        cfg,
        lambda _, cb: prog.acquire(soc, progress=True, callback=cb),
        LivePlotter1D("Time (us)", "Amplitude"),
        ticks=(ts,),
        signal2real=qub_signals2reals,
    )

    # get the actual times
    real_ts1 = prog.get_time_param("pi2_pulse1_post_delay", "t", as_array=True)
    real_ts2 = prog.get_time_param("pi_pulse_post_delay", "t", as_array=True)
    real_ts = real_ts1 + real_ts2
    # TODO: check if this is correct
    real_ts += ts[0] - real_ts[0]  # adjust to start from the first time

    return real_ts, signals


def measure_t1(
    soc, soccfg, cfg, liveplot: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    cfg = deepcopy(cfg)  # prevent in-place modification

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")
    sweep_cfg = cfg["sweep"]["length"]

    cfg["pi_pulse"]["post_delay"] = sweep2param("length", sweep_cfg)

    ts = sweep2array(sweep_cfg)  # predicted times

    prog = T1Program(soccfg, cfg)

    # linear hard sweep
    signals = sweep_hard_template(
        cfg,
        lambda _, cb: prog.acquire(soc, progress=liveplot, callback=cb),
        LivePlotter1D("Time (us)", "Amplitude", disable=not liveplot),
        ticks=(ts,),
        signal2real=qub_signals2reals,
        catch_interrupt=liveplot,
    )

    # get the actual times
    real_ts = prog.get_time_param("pi_pulse_post_delay", "t", as_array=True)
    # TODO: make sure this is correct
    real_ts += ts[0] - real_ts[0]  # adjust to start from the first time

    return real_ts, signals


def visualize_t2ramsey(
    soccfg, cfg, detune: float = 0.0, *, time_fly: float = 0.0
) -> None:
    cfg = deepcopy(cfg)  # prevent in-place modification

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")
    sweep_cfg = cfg["sweep"]["length"]

    cfg["pi2_pulse1"] = deepcopy(cfg["pi2_pulse"])
    cfg["pi2_pulse2"] = deepcopy(cfg["pi2_pulse"])

    t2r_spans = sweep2param("length", sweep_cfg)
    cfg["pi2_pulse1"]["post_delay"] = t2r_spans
    cfg["pi2_pulse2"]["phase"] = cfg["pi2_pulse2"]["phase"] + 360 * detune * t2r_spans

    visualizer = SimulateProgramV2(T2RamseyProgram, soccfg, cfg)
    visualizer.visualize(time_fly=time_fly)


def visualize_t2echo(
    soccfg, cfg, detune: float = 0.0, *, time_fly: float = 0.0
) -> None:
    cfg = deepcopy(cfg)  # prevent in-place modification

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")
    sweep_cfg = cfg["sweep"]["length"]

    cfg["pi2_pulse1"] = deepcopy(cfg["pi2_pulse"])
    cfg["pi2_pulse2"] = deepcopy(cfg["pi2_pulse"])

    t2e_half_spans = 0.5 * sweep2param("length", sweep_cfg)
    cfg["pi2_pulse1"]["post_delay"] = t2e_half_spans
    cfg["pi_pulse"]["post_delay"] = t2e_half_spans
    cfg["pi_pulse"]["phase"] = cfg["pi_pulse"]["phase"] + 360 * detune * t2e_half_spans
    cfg["pi2_pulse2"]["phase"] = (
        cfg["pi2_pulse2"]["phase"] + 2 * 360 * detune * t2e_half_spans
    )

    visualizer = SimulateProgramV2(T2EchoProgram, soccfg, cfg)
    visualizer.visualize(time_fly=time_fly)


def visualize_t1(soccfg, cfg, *, time_fly: float = 0.0) -> None:
    cfg = deepcopy(cfg)  # prevent in-place modification

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")

    cfg["pi_pulse"]["post_delay"] = sweep2param("length", cfg["sweep"]["length"])

    visualizer = SimulateProgramV2(T1Program, soccfg, cfg)
    visualizer.visualize(time_fly=time_fly)
