from copy import deepcopy
from typing import Tuple

import numpy as np
from zcu_tools.liveplot.jupyter import LivePlotter1D
from zcu_tools.notebook.single_qubit.process import rotate2real
from zcu_tools.program.v2 import ModularProgramV2, Pulse, make_readout, make_reset

from ...tools import format_sweep1D, sweep2array, sweep2param
from ..template import sweep_hard_template


def qub_signals2reals(signals: np.ndarray) -> np.ndarray:
    return rotate2real(signals).real  # type: ignore


def measure_t2ramsey(
    soc, soccfg, cfg, detune: float = 0
) -> Tuple[np.ndarray, np.ndarray]:
    cfg = deepcopy(cfg)  # prevent in-place modification

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")
    sweep_cfg = cfg["sweep"]["length"]

    t2r_spans = sweep2param("length", sweep_cfg)

    prog = ModularProgramV2(
        soccfg,
        cfg,
        modules=[
            make_reset("reset", reset_cfg=cfg["reset"]),
            Pulse(
                name="pi2_pulse1",
                cfg={
                    **cfg["pi2_pulse"],
                    "post_delay": t2r_spans,
                },
            ),
            Pulse(
                name="pi2_pulse2",
                cfg={
                    **cfg["pi2_pulse"],
                    "phase": cfg["pi2_pulse"]["phase"] + 360 * detune * t2r_spans,
                },
            ),
            make_readout("readout", readout_cfg=cfg["readout"]),
        ],
    )

    # linear hard sweep
    ts = sweep2array(sweep_cfg)  # predicted times
    signals = sweep_hard_template(
        cfg,
        lambda _, cb: prog.acquire(soc, progress=True, callback=cb)[0][0].dot([1, 1j]),
        LivePlotter1D("Time (us)", "Amplitude"),
        ticks=(ts,),
        signal2real=qub_signals2reals,
    )

    # get the actual times
    real_ts = prog.get_time_param("pi2_pulse1_post_delay", "t", as_array=True)
    assert isinstance(real_ts, np.ndarray), "real_ts should be an array"
    # TODO: make sure this is correct
    real_ts += ts[0] - real_ts[0]  # adjust to start from the first time

    return real_ts, signals


def measure_t2echo(
    soc, soccfg, cfg, detune: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    cfg = deepcopy(cfg)  # prevent in-place modification

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")
    sweep_cfg = cfg["sweep"]["length"]

    t2e_spans = sweep2param("length", sweep_cfg)

    prog = ModularProgramV2(
        soccfg,
        cfg,
        modules=[
            make_reset("reset", reset_cfg=cfg["reset"]),
            Pulse(
                name="pi2_pulse1",
                cfg={
                    **cfg["pi2_pulse"],
                    "post_delay": 0.5 * t2e_spans,
                },
            ),
            Pulse(
                name="pi_pulse",
                cfg={
                    **cfg["pi_pulse"],
                    "post_delay": 0.5 * t2e_spans,
                },
            ),
            Pulse(
                name="pi2_pulse2",
                cfg={
                    **cfg["pi2_pulse"],
                    "phase": cfg["pi2_pulse"]["phase"] + 360 * detune * t2e_spans,
                },
            ),
            make_readout("readout", readout_cfg=cfg["readout"]),
        ],
    )

    # linear hard sweep
    ts = sweep2array(sweep_cfg)  # predicted times
    signals = sweep_hard_template(
        cfg,
        lambda _, cb: prog.acquire(soc, progress=True, callback=cb)[0][0].dot([1, 1j]),
        LivePlotter1D("Time (us)", "Amplitude"),
        ticks=(ts,),
        signal2real=qub_signals2reals,
    )

    # get the actual times
    real_ts1 = prog.get_time_param("pi2_pulse1_post_delay", "t", as_array=True)
    real_ts2 = prog.get_time_param("pi_pulse_post_delay", "t", as_array=True)
    assert isinstance(real_ts1, np.ndarray), "real_ts1 should be an array"
    assert isinstance(real_ts2, np.ndarray), "real_ts2 should be an array"
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

    prog = ModularProgramV2(
        soccfg,
        cfg,
        modules=[
            make_reset("reset", reset_cfg=cfg["reset"]),
            Pulse(
                name="pi_pulse",
                cfg={
                    **cfg["pi_pulse"],
                    "post_delay": sweep2param("length", sweep_cfg),
                },
            ),
            make_readout("readout", readout_cfg=cfg["readout"]),
        ],
    )

    # linear hard sweep
    ts = sweep2array(sweep_cfg)  # predicted times
    signals = sweep_hard_template(
        cfg,
        lambda _, cb: prog.acquire(soc, progress=liveplot, callback=cb)[0][0].dot(
            [1, 1j]
        ),
        LivePlotter1D("Time (us)", "Amplitude", disable=not liveplot),
        ticks=(ts,),
        signal2real=qub_signals2reals,
        catch_interrupt=liveplot,
    )

    # get the actual times
    real_ts = prog.get_time_param("pi_pulse_post_delay", "t", as_array=True)
    assert isinstance(real_ts, np.ndarray), "real_ts should be an array"
    # TODO: make sure this is correct
    real_ts += ts[0] - real_ts[0]  # adjust to start from the first time

    return real_ts, signals
