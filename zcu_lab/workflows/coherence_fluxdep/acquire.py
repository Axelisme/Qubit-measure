"""coherence_fluxdep acquire helpers for lab workflows."""

from copy import deepcopy

import numpy as np
from numpy.typing import NDArray
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.v2.runtime.run_acquire import RunOptions
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.experiment.v2.utils.snr import snr_checker
from zcu_tools.program.v2 import (
    Pulse,
    PulseReadout,
    Readout,
    Reset,
    TimelineBuilder,
)
from zcu_tools.utils.process import rotate2real

from zcu_lab.v2.singleshot.ge.core import GE_Cfg, GE_Result
from zcu_lab.v2.support import live_line
from zcu_lab.v2.support.singleshot.util import raw_shots_to_signal
from zcu_lab.v2.twotone.time_domain.detune import actual_detune
from zcu_lab.v2.twotone.time_domain.t2ramsey.core import (
    T2RamseyCfg,
    T2RamseyResult,
)


def t2ramsey_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    """Project complex Ramsey IQ onto its signal axis without changing the input."""
    return rotate2real(signals).real


def run_t2ramsey(
    cfg: T2RamseyCfg, *, context: RunContext, retry: int
) -> T2RamseyResult:
    """Measure Ramsey fringes to observe dephasing between two pi/2 pulses.

    Supply complete native cfg with scalar detune in MHz and delay sweep in
    microseconds. context supplies SoC, plots and the cancellation token.
    retry is the workflow's transient acquisition retry count.
    The second pi/2 phase advances by 360*detune*delay. Return rounded delays,
    complex IQ and actual activated detune in MHz, retaining partial data on
    stop. Schedule records acquisition failures on the token; other errors
    propagate. No setup, fit or save occurs, and cfg remains unchanged.
    """
    cfg = deepcopy(cfg)
    soc, soccfg = context.soc, context.soccfg
    times = sweep2array(cfg.delay, "time", {"soccfg": soccfg})
    fig, ax = context.plots.subplots(num="measurement")
    ax.set(xlabel="Time (us)", ylabel="Signal")
    plot = live_line(ax)

    @context.plots.on_plot_thread
    def update(samples):
        return plot.set(times, np.real(samples))

    fig.show()
    buffer = SignalBuffer((len(times),), on_update=update)
    # Execute the pulse sequence using locally constructed modules and sweeps.
    with Schedule(buffer, scope=context.scope) as sched:
        builder = TimelineBuilder(soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay)
        delay = builder.add_sweep("delay", cfg.delay)
        reset = Reset("reset", cfg.reset)
        if reset is not None:
            builder.emit(reset)
        builder.emit(Pulse("pi2_pulse", cfg.pi2_pulse))
        builder.timeline_inc(delay)
        builder.emit(
            Pulse(
                "pi2_pulse",
                cfg.pi2_pulse,
                phase=cfg.pi2_pulse.phase + 360 * cfg.detune * delay,
            )
        )
        builder.emit(Readout("readout", cfg.readout))
        sched.run_acquire(
            builder.build(),
            soc,
            rounds=cfg.rounds,
            options=RunOptions(
                retry=retry,
                stop_condition=snr_checker(
                    buffer.at(), cfg.earlystop_snr, t2ramsey_signal2real
                ),
            ),
        )
    return T2RamseyResult(
        times=times,
        signals=buffer.array,
        true_activate_detune=actual_detune(
            cfg.delay,
            detune=cfg.detune,
            gen_ch=cfg.pi2_pulse.ch,
            soccfg=soccfg,
            time_scaler=1.0,
        ),
    )


def run_ge(cfg: GE_Cfg, *, context: RunContext) -> GE_Result:
    """Collect probe-off/on IQ clouds for ground/excited readout calibration.

    Each shot applies optional reset and init, omits or applies the probe
    pulse, then reads out. Return complex IQ shaped (2, cfg.shots) and shot
    indices. Rows denote probe off/on, not guaranteed pure ground/excited
    states. Physical G/E assignment belongs to the caller's analysis.

    Use one round and cfg.shots reps on a copied cfg. Raw-shot conversion
    requires exactly one readout channel. context supplies acquisition and plots;
    retry stays zero. cfg is unchanged.
    """

    # Configure raw-shot acquisition rather than averaging the IQ clouds.
    soc, soccfg = context.soc, context.soccfg
    cfg = deepcopy(cfg)

    figure, axes = context.plots.subplots(num="measurement")

    @context.plots.on_plot_thread
    def update(clouds: NDArray[np.complex128]):
        axes.clear()
        for cloud, label, color in zip(
            clouds,
            ("g preparation (probe off)", "e preparation (probe on)"),
            ("tab:blue", "tab:red"),
            strict=True,
        ):
            axes.scatter(
                cloud.real, cloud.imag, s=8, alpha=0.5, color=color, label=label
            )
        axes.set(xlabel="I (a.u.)", ylabel="Q (a.u.)", title="GE IQ")
        axes.set_aspect("equal", adjustable="datalim")
        axes.grid()
        axes.legend()
        return (figure,)

    figure.show()

    signals_buffer = SignalBuffer((2, cfg.shots), on_update=update)
    with Schedule(signals_buffer, scope=context.scope) as sched:
        for with_probe, step in sched.scan("w/o probe pulse", [False, True]):
            probe_cfg = cfg.probe_pulse if with_probe else None

            # Compare identical preparations with the probe omitted/applied.
            builder = TimelineBuilder(
                soccfg,
                reps=cfg.shots,
                initial_delay=cfg.initial_delay,
                relax_delay=cfg.relax_delay,
            )
            if cfg.reset is not None:
                builder.emit(Reset("reset", cfg.reset))
            builder.emit(Pulse("init_pulse", cfg.init_pulse))
            builder.emit(Pulse("probe_pulse", probe_cfg))
            builder.emit(PulseReadout("readout", cfg.readout))
            program = builder.build()

            # Store each cloud independently, preserving every complex shot.
            shots = step.run_acquire_shots(
                program, soc, options=RunOptions(progress=True)
            )
            if shots is None:
                break
            signals_buffer[step].set(raw_shots_to_signal(program))
    signals = signals_buffer.array

    # Keep acquisition order; physical G/E mapping belongs to analysis.
    return GE_Result(
        signals=signals,
        shot_indices=np.arange(signals.shape[1], dtype=np.int64),
    )
