"""Workflow-owned T1 and Hahn-echo sequences, live plots and IQ projections."""

from copy import deepcopy

import numpy as np
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.v2.runtime import RunOptions
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.experiment.v2.utils.snr import snr_checker
from zcu_tools.experiment.v2.utils.t1_sampling import materialize_nonuniform_t1_delays
from zcu_tools.program.v2 import (
    Delay,
    DelayAuto,
    LoadValue,
    Pulse,
    Readout,
    TimelineBuilder,
)
from zcu_tools.utils.process import rotate2real

from zcu_lab.v2.support import live_line
from zcu_lab.v2.twotone.time_domain.detune import actual_detune
from zcu_lab.v2.twotone.time_domain.t1.core import T1Cfg, T1Result
from zcu_lab.v2.twotone.time_domain.t2echo.core import T2EchoCfg, T2EchoResult


def t1_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    """Project complex T1 IQ onto its signal axis without changing the input."""
    return rotate2real(signals).real


def t2echo_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    """Project complex echo IQ onto its signal axis without changing the input."""
    return rotate2real(signals).real


def run_t1(cfg: T1Cfg, *, context: RunContext, retry: int) -> T1Result:
    """Measure reset → π → delay → readout on a linear or decay grid.

    delay is a SweepCfg time window in us. Decay sampling concentrates
    points near the start; quantized points must remain distinct.
    Return quantized delays and mean complex IQ with shape (time,).
    Unmeasured points remain NaN. context supplies acquisition and plots;
    retry is the workflow's transient acquisition retry count. cfg is unchanged.
    """
    cfg = deepcopy(cfg)
    if cfg.sampling == "linear":
        return _run_t1_linear(cfg, context=context, retry=retry)
    return _run_t1_decay(cfg, context=context, retry=retry)


def _run_t1_linear(cfg: T1Cfg, *, context: RunContext, retry: int) -> T1Result:
    lengths = sweep2array(cfg.delay, "time", {"soccfg": context.soccfg})
    fig, ax = context.plots.subplots(num="measurement")
    ax.set(xlabel="Time (us)", ylabel="Amplitude")
    ax.set_title("T1 relaxation")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.grid()
    viewer = live_line(ax)

    @context.plots.on_plot_thread
    def update(data):
        return viewer.set(lengths, t1_signal2real(data))

    fig.show()
    signals = SignalBuffer(
        (len(lengths),),
        on_update=update,
    )
    with Schedule(signals, scope=context.scope) as sched:
        builder = TimelineBuilder(
            context.soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
        )
        delay = builder.add_sweep("delay", cfg.delay)
        if cfg.reset is not None:
            builder.emit(cfg.reset.build("reset"))
        builder.emit(Pulse("pi_pulse", cfg.pi_pulse))
        builder.timeline_inc(delay)
        builder.delay_by_timeline()
        builder.emit(Readout("readout", cfg.readout))
        prog = builder.build()
        sched.run_acquire(
            prog,
            context.soc,
            rounds=cfg.rounds,
            options=RunOptions(
                retry=retry,
                stop_condition=snr_checker(
                    signals.at(), cfg.earlystop_snr, t1_signal2real
                ),
            ),
        )
    return T1Result(times=lengths, signals=signals.array)


def _run_t1_decay(cfg: T1Cfg, *, context: RunContext, retry: int) -> T1Result:
    table = materialize_nonuniform_t1_delays(cfg.delay, soccfg=context.soccfg)
    lengths = table.times_us
    fig, ax = context.plots.subplots(num="measurement")
    ax.set(xlabel="Time (us)", ylabel="Amplitude")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.grid()
    viewer = live_line(ax)

    @context.plots.on_plot_thread
    def update(data):
        return viewer.set(lengths, t1_signal2real(data))

    fig.show()
    signals = SignalBuffer(
        (len(lengths),),
        on_update=update,
    )
    with Schedule(signals, scope=context.scope) as sched:
        builder = TimelineBuilder(
            context.soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
        )
        builder.add_loop("delay", len(lengths))
        builder.emit(
            LoadValue(
                "load_t1_delay",
                values=list(table.cycles),
                idx_reg="delay",
                val_reg="t1_delay_cycle",
                auto_compress=False,
            )
        )
        if cfg.reset is not None:
            builder.emit(cfg.reset.build("reset"))
        builder.emit(Pulse("pi_pulse", cfg.pi_pulse))
        builder.emit(DelayAuto("t1_delay", t="t1_delay_cycle"))
        builder.emit(Readout("readout", cfg.readout))
        prog = builder.build()
        sched.run_acquire(
            prog,
            context.soc,
            rounds=cfg.rounds,
            options=RunOptions(
                retry=retry,
                stop_condition=snr_checker(
                    signals.at(), cfg.earlystop_snr, t1_signal2real
                ),
            ),
        )
    return T1Result(times=lengths, signals=signals.array)


def run_t2echo(cfg: T2EchoCfg, *, context: RunContext, retry: int) -> T2EchoResult:
    """Measure refocused coherence with a Hahn-echo delay sweep.

    After optional reset, apply pi/2, half the total delay, pi, the other
    half delay, then pi/2 and readout. cfg supplies the pulses and total
    delay sweep in us. detune in MHz ramps the final pi/2 phase with total
    delay to add fringes; it does not change the drive frequency.
    Return hardware-rounded total delays, mean complex IQ and the realized
    phase-ramp frequency in MHz. context supplies acquisition and plots;
    retry is the workflow's transient retry count. cfg is unchanged.
    """
    # 1. Round both half-delays and report the realized phase-ramp frequency.
    cfg = deepcopy(cfg)
    soc, soccfg = context.soc, context.soccfg
    detune = cfg.detune

    lengths = sweep2array(cfg.delay, "time", {"soccfg": soccfg, "scaler": 0.5})

    true_detune = actual_detune(
        cfg.delay,
        detune=detune,
        gen_ch=cfg.pi2_pulse.ch,
        soccfg=soccfg,
        time_scaler=0.5,
    )

    # 2. Display the IQ projection while averaging the total-delay sweep.
    fig, ax = context.plots.subplots(num="measurement")
    ax.set(xlabel="Time (us)", ylabel="Amplitude")
    ax.set_title(f"T2 Echo (detune={true_detune:.3f}MHz)")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.grid()
    viewer = live_line(ax)

    @context.plots.on_plot_thread
    def update(data):
        return viewer.set(lengths, t2echo_signal2real(data))

    fig.show()
    signals_buffer = SignalBuffer(
        (len(lengths),),
        on_update=update,
    )
    with Schedule(signals_buffer, scope=context.scope) as sched:
        builder = TimelineBuilder(soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay)
        length_sweep = cfg.delay
        length_param = builder.add_sweep("delay", length_sweep)
        detune_param = 360 * detune * length_param

        # 3. Refocus at the midpoint, then project with the programmed phase ramp.

        if cfg.reset is not None:
            builder.emit(cfg.reset.build("reset"))
        builder.emit(Pulse("pi2_pulse1", cfg.pi2_pulse))
        builder.emit(Delay("t2e_delay1", delay=0.5 * length_param))
        builder.delay_by_timeline()
        builder.emit(Pulse("pi_pulse", cfg.pi_pulse))
        builder.emit(Delay("t2e_delay2", delay=0.5 * length_param))
        builder.delay_by_timeline()
        builder.emit(
            Pulse(
                name="pi2_pulse2",
                cfg=cfg.pi2_pulse.with_updates(
                    phase=cfg.pi2_pulse.phase + detune_param
                ),
            )
        )
        builder.emit(Readout("readout", cfg.readout))
        prog = builder.build()
        sched.run_acquire(
            prog,
            soc,
            rounds=cfg.rounds,
            options=RunOptions(
                retry=retry,
                stop_condition=snr_checker(
                    signals_buffer.at(), cfg.earlystop_snr, t2echo_signal2real
                ),
            ),
        )

    # Keep measured IQ separate from fitted coherence times and fringe frequency.
    return T2EchoResult(
        times=lengths,
        signals=signals_buffer.array,
        true_activate_detune=true_detune,
    )
