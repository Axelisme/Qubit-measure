"""Workflow-owned mux lifetime sequences and per-participant live IQ plots."""

from copy import deepcopy
from functools import partial

import numpy as np
from numpy.typing import NDArray
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.v2.runtime import RunOptions, Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.experiment.v2.utils.t1_sampling import materialize_nonuniform_t1_delays
from zcu_tools.program.v2 import DelayAuto, LoadValue, Reset, TimelineBuilder
from zcu_tools.program.v2.modules import MuxReadout, Pulse

from zcu_lab.v2.mux.t1.core import MuxT1Cfg, MuxT1Result
from zcu_lab.v2.mux.t2echo.core import MuxT2EchoCfg
from zcu_lab.v2.support import live_line
from zcu_lab.v2.support.acquire import mux_snr_checker
from zcu_lab.v2.support.mux import averaged_mux_signal
from zcu_lab.v2.support.mux_coherence import MuxCoherenceResult
from zcu_lab.v2.twotone.time_domain.detune import actual_detune


def run_mux_t1(cfg: MuxT1Cfg, *, context: RunContext, retry: int) -> MuxT1Result:
    """Acquire simultaneous pi pulses, shared wait and mux readout.

    cfg supplies full tones, ordered participants, fixed pulses and averaging.
    Linear sampling returns requested delays in us; decay sampling returns
    quantized nonuniform delays. Return float64 times (P,), int64 original tone
    indices (K,) and complex128 mean IQ (K,P). Missing rounds leave NaN.
    context supplies acquisition handles, plots and stop scope; retry counts
    transient retries. Inputs remain unchanged; acquisition errors propagate.
    """
    cfg = deepcopy(cfg)
    if cfg.sampling == "linear":
        return _run_t1_linear(cfg, context=context, retry=retry)
    return _run_t1_decay(cfg, context=context, retry=retry)


def _run_t1_linear(cfg: MuxT1Cfg, *, context: RunContext, retry: int) -> MuxT1Result:
    times = sweep2array(cfg.delay, "none")
    indices = list(cfg.participants)
    readout = cfg.readout.with_updates(tones=[cfg.readout.tones[i] for i in indices])
    viewers = []
    for i in indices:
        fig, ax = context.plots.subplots(num=f"measurement_{i}")
        ax.set(
            xlabel="Delay (us)",
            ylabel="Magnitude (a.u.)",
            title=f"{i}: {cfg.tones[i].name}",
        )
        viewers.append(live_line(ax))
        fig.show()

    @context.plots.on_plot_thread
    def update(data: NDArray[np.complex128]):
        return tuple(
            artist
            for row, viewer in enumerate(viewers)
            for artist in viewer.set(times, np.abs(data[:, row]))
        )

    signals = SignalBuffer((len(times), len(indices)), on_update=update)
    with Schedule(signals, scope=context.scope) as sched:
        builder = TimelineBuilder(
            context.soccfg,
            reps=cfg.reps,
            initial_delay=cfg.initial_delay,
            relax_delay=cfg.relax_delay,
        )
        delay = builder.add_sweep("delay", cfg.delay)
        if cfg.reset is not None:
            builder.emit(Reset("reset", cfg.reset))
        timelines = [f"drive_{index}" for index in indices]
        builder.fork("default", into=timelines)
        for index, timeline in zip(indices, timelines, strict=True):
            builder.emit(
                Pulse(f"drive_{index}", cfg.tones[index].require_drive()),
                timeline=timeline,
            )
        builder.join(*timelines, into="default")
        builder.timeline_inc(delay)
        builder.delay_by_timeline()
        readout_module = MuxReadout("readout", readout)
        builder.emit(readout_module)
        prog = builder.build()
        sched.run_acquire(
            prog,
            context.soc,
            rounds=cfg.rounds,
            options=RunOptions(
                converter=partial(averaged_mux_signal, prog, readout_module),
                retry=retry,
                stop_condition=mux_snr_checker(signals, cfg.earlystop_snr, tone_axis=1),
            ),
        )
    return MuxT1Result(
        times, np.asarray(indices, dtype=np.int64), signals.array.T.copy()
    )


def _run_t1_decay(cfg: MuxT1Cfg, *, context: RunContext, retry: int) -> MuxT1Result:
    table = materialize_nonuniform_t1_delays(cfg.delay, soccfg=context.soccfg)
    times = table.times_us
    indices = list(cfg.participants)
    readout = cfg.readout.with_updates(tones=[cfg.readout.tones[i] for i in indices])
    viewers = []
    for i in indices:
        fig, ax = context.plots.subplots(num=f"measurement_{i}")
        ax.set(
            xlabel="Delay (us)",
            ylabel="Magnitude (a.u.)",
            title=f"{i}: {cfg.tones[i].name}",
        )
        viewers.append(live_line(ax))
        fig.show()

    @context.plots.on_plot_thread
    def update(data: NDArray[np.complex128]):
        return tuple(
            artist
            for row, viewer in enumerate(viewers)
            for artist in viewer.set(times, np.abs(data[:, row]))
        )

    signals = SignalBuffer((len(times), len(indices)), on_update=update)
    with Schedule(signals, scope=context.scope) as sched:
        builder = TimelineBuilder(
            context.soccfg,
            reps=cfg.reps,
            initial_delay=cfg.initial_delay,
            relax_delay=cfg.relax_delay,
        )
        builder.add_loop("delay_idx", len(table.cycles))
        builder.emit(
            LoadValue(
                "load_t1_delay",
                values=list(table.cycles),
                idx_reg="delay_idx",
                val_reg="t1_delay_cycle",
                auto_compress=False,
            )
        )
        if cfg.reset is not None:
            builder.emit(Reset("reset", cfg.reset))
        timelines = [f"drive_{index}" for index in indices]
        builder.fork("default", into=timelines)
        for index, timeline in zip(indices, timelines, strict=True):
            builder.emit(
                Pulse(f"drive_{index}", cfg.tones[index].require_drive()),
                timeline=timeline,
            )
        builder.join(*timelines, into="default")
        # Start the register delay after the full pulse branches, including post.
        builder.delay_by_timeline()
        builder.emit(DelayAuto("t1_delay", t="t1_delay_cycle"))
        readout_module = MuxReadout("readout", readout)
        builder.emit(readout_module)
        prog = builder.build()
        sched.run_acquire(
            prog,
            context.soc,
            rounds=cfg.rounds,
            options=RunOptions(
                converter=partial(averaged_mux_signal, prog, readout_module),
                retry=retry,
                stop_condition=mux_snr_checker(signals, cfg.earlystop_snr, tone_axis=1),
            ),
        )
    return MuxT1Result(
        times, np.asarray(indices, dtype=np.int64), signals.array.T.copy()
    )


def run_mux_t2echo(
    cfg: MuxT2EchoCfg, *, context: RunContext, retry: int
) -> MuxCoherenceResult:
    """Measure x90 → half wait → x180 → half wait → x90 → mux readout.

    cfg supplies full tones, ordered participants and a total delay sweep
    in us. Each pulse stage starts together; the next wait or readout starts
    after its longest pulse. The final x90 phase ramps with each tone's MHz
    detune. Return requested total delays (P,), averaged complex IQ (K,P),
    original tone indices, actual detunes and source frequencies (K,).
    context supplies hardware and live plots; retry counts transient retries.
    Cancellation keeps acquired averages and NaN gaps. Inputs remain unchanged;
    acquisition and program errors propagate.
    """
    # 1. Keep participant order while selecting from the complete tone list.
    cfg = deepcopy(cfg)
    indices = list(cfg.participants)
    tones = [cfg.tones[index] for index in indices]
    times = sweep2array(cfg.delay, "none")
    # Both waits use the native half-delay grid, not a rounded total delay.
    _ = sweep2array(cfg.delay, "time", {"soccfg": context.soccfg, "scaler": 0.5})
    detunes = np.asarray(
        [
            actual_detune(
                cfg.delay,
                detune=tone.detune,
                gen_ch=tone.require_drive().ch,
                soccfg=context.soccfg,
                time_scaler=0.5,
            )
            for tone in tones
        ],
        dtype=np.float64,
    )
    frequencies = np.asarray(
        [float(tone.require_drive().freq) for tone in tones], dtype=np.float64
    )
    readout = cfg.readout.with_updates(
        tones=[cfg.readout.tones[index] for index in indices]
    )

    # 2. Show each selected tone on the common requested total-delay axis.
    viewers = []
    for index in indices:
        fig, ax = context.plots.subplots(num=f"measurement_{index}")
        ax.set(
            xlabel="Requested delay (us)",
            ylabel="Magnitude (a.u.)",
            title=f"{index}: {cfg.tones[index].name}",
        )
        viewers.append(live_line(ax))
        fig.show()

    @context.plots.on_plot_thread
    def update(data: NDArray[np.complex128]):
        return tuple(
            artist
            for row, viewer in enumerate(viewers)
            for artist in viewer.set(times, np.abs(data[row]))
        )

    signals = SignalBuffer((len(indices), times.size), on_update=update)
    with Schedule(signals, scope=context.scope) as sched:
        # Each stage starts all drives together and waits for its longest pulse.
        builder = TimelineBuilder(
            context.soccfg,
            reps=cfg.reps,
            initial_delay=cfg.initial_delay,
            relax_delay=cfg.relax_delay,
        )
        delay = builder.add_sweep("delay", cfg.delay)
        if cfg.reset is not None:
            builder.emit(Reset("reset", cfg.reset))
        timelines = [f"pi2_first_{index}" for index in indices]
        builder.fork("default", into=timelines)
        for index, tone, timeline in zip(indices, tones, timelines, strict=True):
            builder.emit(
                Pulse(
                    f"pi2_first_{index}",
                    tone.require_drive(),
                ),
                timeline=timeline,
            )
        builder.join(*timelines, into="default")
        builder.timeline_inc(0.5 * delay)
        builder.delay_by_timeline()
        timelines = [f"pi_{index}" for index in indices]
        builder.fork("default", into=timelines)
        for index, tone, timeline in zip(indices, tones, timelines, strict=True):
            builder.emit(
                Pulse(
                    f"pi_{index}",
                    tone.require_pi_pulse(),
                ),
                timeline=timeline,
            )
        builder.join(*timelines, into="default")
        builder.timeline_inc(0.5 * delay)
        builder.delay_by_timeline()
        timelines = [f"pi2_final_{index}" for index in indices]
        builder.fork("default", into=timelines)
        for index, tone, timeline in zip(indices, tones, timelines, strict=True):
            builder.emit(
                Pulse(
                    f"pi2_final_{index}",
                    tone.require_drive(),
                    phase=tone.require_drive().phase + 360 * tone.detune * delay,
                ),
                timeline=timeline,
            )
        builder.join(*timelines, into="default")
        readout_module = MuxReadout("readout", readout)
        builder.emit(readout_module)

        # 4. Average completed rounds into tone-major IQ, preserving NaN gaps.
        prog = builder.build()
        _ = sched.run_acquire(
            prog,
            context.soc,
            rounds=cfg.rounds,
            options=RunOptions(
                converter=lambda raw: averaged_mux_signal(prog, readout_module, raw).T,
                retry=retry,
                stop_condition=mux_snr_checker(signals, cfg.earlystop_snr, tone_axis=0),
            ),
        )
    return MuxCoherenceResult(
        np.asarray(indices, dtype=np.int64),
        times,
        signals.array.copy(),
        detunes,
        frequencies,
    )
