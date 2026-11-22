"""Measure mux Hahn Echo coherence with simultaneous pulses and shared waits.

Reset, join the x90 pulses, wait half the requested total delay, join the x180
pulses, wait the other half, join the final x90 pulses, then read out together.
Shorter pulses wait for the longest branch at each stage; native pre/post
delays stay unchanged. The final x90 phase includes each tone's detune.

Return original tone indices (K,), requested total delays (P,) in us and mean
complex IQ (K,P), plus actual detunes and source frequencies in MHz. Missing
rounds leave IQ NaN. Live and fit figures show projected IQ and same-direction
shot SEM from optional Native moments. Analyze each tone with decay or fringe
fits, omitting the
original first point. Zero source detune forces decay. Report T2E/error in us,
fringe rate/error in MHz, fit quality or a per-tone numerical failure.

Timing, detune and persistence details: zcu_lab/v2/mux/t2echo/guide.md.
RunContext, stop and plot ownership: lib/zcu_tools/experiment/v2/runtime/README.md.
RunRecord and save/load contracts: lib/zcu_tools/experiment/README.md.
"""

from __future__ import annotations

from typing import Annotated, ClassVar, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, StrictInt, model_validator
from pydantic.dataclasses import dataclass as validated_dataclass
from zcu_tools.analysis.failure import AnalysisFailed
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import Experiment
from zcu_tools.experiment.axes_spec import GroupedAxesSpec
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime import RunOptions, Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    Pulse,
    Reset,
    ResetCfg,
    SweepCfg,
    TimelineBuilder,
    mux_iq_sums,
    split_sums,
)
from zcu_tools.program.v2.modules import (
    MuxReadout,
    MuxReadoutCfg,
)

from zcu_lab.v2.mux._moments import project_iq
from zcu_lab.v2.support import live_errorbar
from zcu_lab.v2.support.acquire import mux_mean_snr_checker
from zcu_lab.v2.support.detune import actual_detune
from zcu_lab.v2.support.mux import MuxFitFailure
from zcu_lab.v2.support.mux_coherence import (
    MuxCoherenceResult,
    MuxCoherenceToneCfg,
    validate_coherence_result,
    validate_coherence_scan,
)
from zcu_lab.v2.support.mux_coherence_analysis import (
    MuxCoherenceFit,
    fit_coherence_trace,
    prepare_coherence_trace,
)
from zcu_lab.v2.support.mux_coherence_persistence import coherence_spec

_DELAY_LABEL = "Total free evolution time (µs)"
_IQ_LABEL = "Principal-axis mean IQ (a.u.)"


@validated_dataclass(frozen=True)
class T2EchoAnalyzeOptions:
    """Select decay or fringe fitting and optionally free the fringe phase."""

    fit_method: Annotated[Literal["fringe", "decay"], FieldMeta(label="Fit method")] = (
        "decay"
    )
    fit_phase: Annotated[bool, FieldMeta(label="Fit phase offset (fringe only)")] = (
        False
    )


class MuxEchoToneCfg(MuxCoherenceToneCfg):
    """One named qubit's x90 drive, MHz detune and x180 pi_pulse.

    Both pulses use the same generator, independent of the other qubits.
    Their frequency, gain, phase and waveform may differ. Nonparticipating
    tones may leave pulses None; name identifies the qubit in saved results.
    """

    pi_pulse: Annotated[
        Pulse.Cfg | None,
        FieldMeta(
            label="Refocusing π pulse",
            description="Enter the calibrated π pulse applied halfway through the total free evolution time.",
        ),
    ] = None

    def require_pi_pulse(self) -> Pulse.Cfg:
        """Return the active x180 pulse, or raise ValueError naming the tone."""
        if self.pi_pulse is None:
            raise ValueError(f"{self.name}: active tone requires pi_pulse")
        return self.pi_pulse


class MuxT2EchoCfg(ConfigBase):
    """Mux Hahn Echo pulses, participating qubits and a shared total-delay scan.

    modules holds optional reset and the complete mux readout. tones contains
    one uniquely named MuxEchoToneCfg per readout position. participants is a
    nonempty list of distinct original indices: [2, 0] gives result
    rows for original indices 2/0. All drives must use independent generators
    distinct from readout for selected tones only.
    delay is a finite nonnegative total wait in us, split into two native half
    waits. Each tone's detune ramps its final x90 phase. reps/rounds are positive;
    initial_delay/relax_delay are finite nonnegative us shared by all tones.
    """

    reset: Annotated[
        ResetCfg | None,
        FieldMeta(
            label="Reset",
            description="Choose the common preparation reset; leave blank to omit it.",
        ),
    ] = None
    readout: Annotated[
        MuxReadoutCfg,
        FieldMeta(
            label="Readout",
            description="Enter the calibrated multiplexed readout used after the final π/2 pulses.",
        ),
    ]
    tones: Annotated[list[MuxEchoToneCfg], FieldMeta(element_label="name")]
    participants: Annotated[
        list[StrictInt],
        FieldMeta(
            description="Enter original zero-based tone indices in measurement order; these select drives and readout tones."
        ),
    ]
    delay: Annotated[
        SweepCfg,
        FieldMeta(
            label="Delay",
            unit="µs",
            description="Enter the shared Hahn echo free evolution time range in µs.",
        ),
    ]

    reps: Annotated[
        int,
        FieldMeta(
            label="Reps", unit="count", description="Number of repetitions per round."
        ),
    ] = Field(gt=0)
    rounds: Annotated[
        int,
        FieldMeta(
            label="Rounds",
            unit="count",
            description="Number of software acquisition rounds.",
        ),
    ] = Field(default=1, gt=0)
    earlystop_snr: Annotated[
        float | None,
        FieldMeta(
            label="Early-stop SNR",
            description="Stop when every active tone reaches this trace SNR; None disables.",
        ),
    ] = Field(default=None, gt=0)
    initial_delay: Annotated[
        float,
        FieldMeta(
            label="Initial delay",
            unit="µs",
            description="Delay before the first repetition.",
        ),
    ] = Field(default=1.0, ge=0, allow_inf_nan=False)
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="µs",
            decimals=3,
            description="Relaxation delay after each repetition.",
        ),
    ] = Field(ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_scan(self) -> MuxT2EchoCfg:
        """Reject missing active pulses, swept values, shared channels or bad delays."""
        validate_coherence_scan(self, self.tones, self.delay, self.participants)
        validate_coherence_scan(
            self,
            [
                MuxCoherenceToneCfg(name=t.name, drive=t.pi_pulse, detune=t.detune)
                for t in self.tones
            ],
            self.delay,
            self.participants,
        )
        for index in self.participants:
            tone = self.tones[index]
            if tone.require_drive().ch != tone.require_pi_pulse().ch:
                raise ValueError(f"tones[{index}].pi_pulse.ch must equal drive.ch")
        return self


def _show_live_echo(plots: Plots, cfg: MuxT2EchoCfg, times: NDArray[np.float64]):
    viewers = []
    for index in cfg.participants:
        fig, ax = plots.subplots(num=f"measurement_{index}")
        fig.suptitle(f"Mux Hahn echo · {cfg.tones[index].name} · IQ contrast")
        ax.set(xlabel=_DELAY_LABEL, ylabel=_IQ_LABEL)
        progress = ax.set_title(f"Acquired delay points: 0 / {times.size}")
        viewer = live_errorbar(ax)
        ax.lines[0].set(marker="o", markersize=3, label="Measured IQ ± shot SEM")
        ax.legend(loc="best")
        viewers.append((viewer, progress))
        fig.show()

    @plots.on_plot_thread
    def update(data: NDArray[np.float64]):
        _, mean = split_sums(data)
        artists = []
        for row, (viewer, progress) in enumerate(viewers):
            measured = mean[:, row]
            acquired = np.isfinite(measured)
            # Show the full sweep, but choose the direction from points used in Analyze.
            projected, sem = project_iq(
                np.where(acquired, measured, np.nan), data[:, row], skip_for_axis=1
            )
            progress.set_text(
                f"Acquired delay points: {np.count_nonzero(acquired)} / {times.size}"
            )
            artists.extend((*viewer.set(times, projected, sem), progress))
        return artists

    return update


class MuxT2EchoExp(Experiment[MuxCoherenceResult, MuxT2EchoCfg]):
    """Measure and fit each participating qubit's Hahn Echo decay or fringes."""

    AXES_SPEC: ClassVar[GroupedAxesSpec[MuxCoherenceResult, MuxT2EchoCfg]] = (
        coherence_spec(MuxT2EchoCfg, tag="mux/time_domain/t2echo")
    )

    Options: ClassVar[type[T2EchoAnalyzeOptions]] = T2EchoAnalyzeOptions

    def run(self, config: MuxT2EchoCfg, *, context: RunContext) -> MuxCoherenceResult:
        """Measure x90 → half wait → x180 → half wait → x90 → mux readout.

        config supplies full tones, ordered participants and a total delay sweep
        in us. Each pulse stage starts together; the next wait or readout starts
        after its longest pulse. The final x90 phase ramps with each tone's MHz
        detune. Return requested total delays (P,), averaged complex IQ (K,P),
        original tone indices, actual detunes and source frequencies (K,).
        context supplies hardware and live plots. Cancellation keeps acquired
        averages and NaN gaps. Acquisition and program errors propagate.
        """
        # 1. Keep participant order while selecting from the complete tone list.
        cfg = config.model_copy(deep=True)
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
        update = _show_live_echo(context.plots, cfg, times)

        moments = SignalBuffer(
            (times.size, len(indices), 6), dtype=np.float64, on_update=update
        )
        with Schedule(moments, scope=context.scope) as sched:
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
            builder.emit(MuxReadout("readout", readout))

            # 4. Average completed rounds into tone-major IQ, preserving NaN gaps.
            prog = builder.build()
            _ = sched.run_acquire_summed(
                prog,
                context.soc,
                rounds=cfg.rounds,
                reducer=mux_iq_sums(prog, "readout"),
                options=RunOptions(
                    stop_condition=mux_mean_snr_checker(
                        lambda: split_sums(moments.array)[1],
                        cfg.earlystop_snr,
                        tone_axis=1,
                    )
                ),
            )
        moments.trigger_update(flush=True)
        tone_sums = np.swapaxes(moments.array, 0, 1).copy()
        _, mean = split_sums(tone_sums)
        return MuxCoherenceResult(
            np.asarray(indices, dtype=np.int64),
            times,
            mean,
            detunes,
            frequencies,
            tone_sums,
        )

    def analyze(
        self,
        source: RunRecord[MuxT2EchoCfg, MuxCoherenceResult],
        options: T2EchoAnalyzeOptions,
        *,
        plots: Plots,
    ) -> tuple[MuxCoherenceFit | MuxFitFailure, ...]:
        """Fit each source tone's Echo decay or fringes against requested delay.

        source is a Run/Load record, with optional cfg. options.fit_method selects
        decay or fringe; zero source detune forces decay. Gating uses cfg detune
        when present, otherwise finite saved actual_detunes. Unknown detune fails
        a requested fringe fit with a missing-source reason, not a guessed model.
        Explicit decay needs no detune. fit_phase frees fringe phase (otherwise zero).
        Omit the first point and missing IQ. Return per-tone T2E in us, fringe
        rate in MHz, uncertainties and fit quality, or a fit failure reason.
        Acquired delays, projected observations and predictions stay local
        to analysis. Draw only the primary curve in fit_i. Invalid sources or
        fit methods raise.
        """
        # 1. Omit the first requested point before removing missing IQ.
        validate_coherence_result(source.result)
        moments = source.result.iq_moments
        outcomes: list[MuxCoherenceFit | MuxFitFailure] = []
        for row, raw_index in enumerate(source.result.tone_indices):
            trace = prepare_coherence_trace(
                int(raw_index),
                source.result.times,
                source.result.signals[row],
                skip=1,
                iq_moments=None if moments is None else moments[row],
            )
            cfg = source.cfg
            tone = f"Tone {trace.index}" if cfg is None else cfg.tones[trace.index].name
            fig, ax = plots.subplots(num=f"fit_{trace.index}", figsize=(7, 5))
            ax.set(xlabel=_DELAY_LABEL, ylabel=_IQ_LABEL)
            fig.suptitle(f"Mux Hahn echo · {tone} · IQ contrast")
            real, sem = project_iq(trace.signal, trace.iq_moments)
            ax.plot(trace.times, real, "o", ms=3, label="Measured IQ")
            if sem is not None and np.any(np.isfinite(sem)):
                ax.errorbar(
                    trace.times,
                    real,
                    yerr=sem,
                    fmt="none",
                    ecolor="C0",
                    label="Shot SEM",
                )
            try:
                # Source detune, not a fitted oscillation, determines fringe eligibility.
                fit_fringe = options.fit_method == "fringe"
                if fit_fringe:
                    detune = (
                        source.cfg.tones[trace.index].detune
                        if source.cfg is not None
                        else float(source.result.actual_detunes[row])
                    )
                    if not np.isfinite(detune):
                        raise ValueError(
                            "Echo fringe fit requires source detune: cfg or saved finite actual_detune"
                        )
                    fit_fringe = detune != 0.0
                if len(trace.times) < (6 if fit_fringe else 4):
                    raise ValueError("Insufficient acquired points for coherence fit")

                # 2. Echo fringes start at zero phase unless phase is fitted freely.
                phase = None if options.fit_phase else 0.0
                curve_times = np.linspace(trace.times[0], trace.times[-1], 400)
                outcome, fitted = fit_coherence_trace(
                    trace,
                    real,
                    fit_fringe=fit_fringe,
                    phase=phase,
                    expected_frequency_mhz=None,
                    curve_times=curve_times,
                )
            except (
                AnalysisFailed,
                ValueError,
                RuntimeError,
                ArithmeticError,
                np.linalg.LinAlgError,
            ) as exc:
                ax.text(
                    0.02,
                    0.98,
                    f"Fit unavailable: {exc}",
                    transform=ax.transAxes,
                    va="top",
                    wrap=True,
                )
                outcomes.append(MuxFitFailure(trace.index, str(exc)))
            else:
                model = "Damped cosine" if outcome.fit_fringe else "Exponential decay"
                ax.plot(curve_times, fitted, label=model)
                if trace.times[0] <= outcome.t2 <= trace.times[-1]:
                    ax.axvline(outcome.t2, color="0.4", ls=":", label="T2 echo")
                result = f"T2 echo = {outcome.t2:.4g} ± {outcome.t2_err:.2g} µs"
                if outcome.fit_fringe:
                    result += f" · Fitted fringe rate = {outcome.detune:.4g} ± {outcome.detune_err:.2g} MHz"
                r2 = outcome.quality.r2
                rms = outcome.quality.normalized_residual_rms
                quality = f"R² = {r2:.3g}" if r2 is not None else "R² unavailable"
                quality += (
                    f" · Normalized residual RMS = {rms:.3g}"
                    if rms is not None
                    else " · Normalized residual RMS unavailable"
                )
                ax.set_title(f"{result}\n{quality}", fontsize=10)
                outcomes.append(outcome)

            # 3. Keep each tone's fit or failure visible without stopping other fits.
            ax.legend()
            fig.tight_layout()
        return tuple(outcomes)
