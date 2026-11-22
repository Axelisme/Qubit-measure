"""Measure mux Hahn Echo coherence with simultaneous pulses and shared waits.

Reset, join the x90 pulses, wait half the requested total delay, join the x180
pulses, wait the other half, join the final x90 pulses, then read out together.
Shorter pulses wait for the longest branch at each stage; native pre/post
delays stay unchanged. The final x90 phase includes each tone's detune.

Return original tone indices (K,), requested total delays (P,) in us and mean
complex IQ (K,P), plus actual detunes and source frequencies in MHz. Missing
rounds leave IQ NaN. Analyze each tone with decay or fringe fits, omitting the
original first point. Zero source detune forces decay. Report T2E/error in us,
fringe rate/error in MHz, fit quality or a per-tone numerical failure.

Timing, detune and persistence details: zcu_lab/v2/mux/t2echo/guide.md.
RunContext, stop and plot ownership: lib/zcu_tools/experiment/v2/runtime/README.md.
RunRecord and save/load contracts: lib/zcu_tools/experiment/README.md.
"""

from __future__ import annotations

from typing import Annotated, ClassVar, Literal

import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
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

from zcu_lab.v2.mux._moments import project_iq, validate_iq_moments
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

_IQ_LABEL = "Principal-axis mean IQ (a.u.)"
_DELAY_LABEL = "Requested total echo delay (µs)"


@validated_dataclass(frozen=True)
class T2EchoAnalyzeOptions:
    """Select decay or fringe fitting and optionally free the fringe phase."""

    fit_method: Annotated[
        Literal["fringe", "decay"],
        FieldMeta(
            label="Fit method",
            description="Choose a damped cosine or exponential decay; zero detune forces decay.",
        ),
    ] = "decay"
    fit_phase: Annotated[
        bool,
        FieldMeta(
            label="Fit phase offset (fringe only)",
            description="On fits the fringe phase offset; off fixes it at 0°.",
        ),
    ] = False


class MuxEchoToneCfg(MuxCoherenceToneCfg):
    """One named qubit's x90 drive, MHz detune and x180 pi_pulse.

    Both pulses use the same generator, independent of the other qubits.
    Their frequency, gain, phase and waveform may differ. Nonparticipating
    tones may leave pulses None; name identifies the qubit in saved results.
    """

    pi_pulse: Pulse.Cfg | None = None

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
            description="Choose the common preparation reset; leave blank to omit it."
        ),
    ] = None
    readout: Annotated[
        MuxReadoutCfg,
        FieldMeta(
            description="Enter the calibrated mux readout used after the final π/2 pulses."
        ),
    ]
    tones: Annotated[list[MuxEchoToneCfg], FieldMeta(element_label="name")]
    participants: Annotated[
        list[StrictInt],
        FieldMeta(
            description="Enter the original zero-based qubit positions in measurement order."
        ),
    ]
    delay: Annotated[
        SweepCfg,
        FieldMeta(
            label="Delay",
            unit="µs",
            description="Sweep the total wait split around the π pulse; Expts counts the delay points.",
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


def _show_live(plots: Plots, cfg: MuxT2EchoCfg, times: NDArray[np.float64]):
    """Show one live figure per cfg participant and return its shared updater.

    times contains requested total echo delays in µs. The updater takes completed-round
    sums shaped (delays, participants, 6) and returns the changed data, SEM and
    progress artists. Projection and plot update errors propagate.
    """
    viewers = []
    empty = times[:0]
    for index in cfg.participants:
        fig, ax = plots.subplots(num=f"measurement_{index}", layout="constrained")
        fig.suptitle(f"Mux Echo · {cfg.tones[index].name} · IQ contrast")
        ax.set(xlabel=_DELAY_LABEL, ylabel=_IQ_LABEL)
        progress = ax.set_title(
            f"Acquired points: 0 / {times.size} · 0 shots per point"
        )
        viewer = live_errorbar(ax)
        for artist in viewer.set(empty, empty, empty):
            if isinstance(artist, Line2D):
                artist.set(marker="o", markersize=4, label="Measured IQ ± shot SEM")
        fig.legend(loc="outside lower center")
        viewers.append((viewer, progress))
        fig.show()

    @plots.on_plot_thread
    def update(data: NDArray[np.float64]):
        counts, means = split_sums(data)
        artists = []
        for row, (viewer, progress) in enumerate(viewers):
            acquired = np.isfinite(means[:, row])
            real, sem = project_iq(means[:, row], data[:, row], skip_for_axis=1)
            shots = int(counts[acquired, row].min()) if acquired.any() else 0
            progress.set_text(
                f"Acquired points: {np.count_nonzero(acquired)} / {times.size} · "
                f"{shots:,} shots per point"
            )
            artists.extend((*viewer.set(times, real, sem), progress))
        return artists

    return update


def _plot_fit(
    ax: Axes,
    times: NDArray[np.float64],
    fitted: NDArray[np.float64],
    outcome: MuxCoherenceFit,
) -> None:
    """Add the fit curve, in-window T2 echo marker and fit facts to ax.

    times contains increasing requested delays in µs. fitted contains the model
    values on that grid; outcome supplies the model kind, estimates and quality.
    Update ax in place; return None without presenting or refreshing the figure.
    """
    model = "Damped cosine" if outcome.fit_fringe else "Exponential decay"
    ax.plot(times, fitted, color="C1", label=model)
    if times[0] <= outcome.t2 <= times[-1]:
        ax.axvline(outcome.t2, color="0.4", ls=":", label="T2 echo")
    estimate = f"T2 echo = {outcome.t2:.4g} ± {outcome.t2_err:.2g} µs"
    if outcome.fit_fringe:
        estimate += f" · Fitted fringe rate = {outcome.detune:.4g} ± {outcome.detune_err:.2g} MHz"
    r2 = outcome.quality.r2
    rms = outcome.quality.normalized_residual_rms
    r2_text = "unavailable" if r2 is None else f"{r2:.3g}"
    rms_text = "unavailable" if rms is None else f"{rms:.3g}"
    ax.set_title(
        f"{estimate}\nR² = {r2_text} · Normalized residual RMS = {rms_text}",
        fontsize=10,
    )


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
        original tone indices, actual detunes and source frequencies (K,), plus
        completed-round IQ sums (K,P,6) for this Run's plots.
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

        update = _show_live(context.plots, cfg, times)
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
            readout_module = MuxReadout("readout", readout)
            builder.emit(readout_module)

            # One summed acquisition supplies the IQ mean and shot SEM (and the Stop check).
            prog = builder.build()
            _ = sched.run_acquire_summed(
                prog,
                context.soc,
                reducer=mux_iq_sums(prog, "readout"),
                rounds=cfg.rounds,
                options=RunOptions(
                    stop_condition=mux_mean_snr_checker(
                        lambda: split_sums(moments.array)[1],
                        cfg.earlystop_snr,
                        tone_axis=1,
                    )
                ),
            )
        moments.trigger_update(flush=True)
        _, means = split_sums(moments.array)
        return MuxCoherenceResult(
            np.asarray(indices, dtype=np.int64),
            times,
            means.T,
            detunes,
            frequencies,
            moments.array.transpose(1, 0, 2).copy(),
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
        to analysis. Draw fit_i with any available projected IQ; on projection
        failure show the reason without data. Invalid sources or fit methods raise.
        """
        # 1. Omit the first requested point before removing missing IQ.
        data = source.result
        validate_coherence_result(data)
        validate_iq_moments(data.iq_moments, data.signals)
        outcomes: list[MuxCoherenceFit | MuxFitFailure] = []
        for row, raw_index in enumerate(data.tone_indices):
            trace = prepare_coherence_trace(
                int(raw_index), data.times, data.signals[row], skip=1
            )
            finite = np.isfinite(data.signals[row, 1:])
            moments = (
                None if data.iq_moments is None else data.iq_moments[row, 1:][finite]
            )
            name = (
                f"Tone {raw_index}"
                if source.cfg is None
                else source.cfg.tones[trace.index].name
            )
            fig, ax = plots.subplots(
                num=f"fit_{trace.index}", figsize=(8, 5), layout="constrained"
            )
            ax.set(xlabel=_DELAY_LABEL, ylabel=_IQ_LABEL)
            fig.suptitle(f"Mux Echo · {name} · IQ contrast")
            projected = None
            try:
                projected = project_iq(trace.signal, moments)
                real, _ = projected
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
                if trace.times.size < (6 if fit_fringe else 4):
                    raise ValueError("Insufficient acquired points for coherence fit")
                curve_times = np.linspace(trace.times.min(), trace.times.max(), 400)

                # 2. Echo fringes start at zero phase unless phase is fitted freely.
                phase = None if options.fit_phase else 0.0
                outcome, fitted = fit_coherence_trace(
                    trace,
                    real,
                    fit_fringe=fit_fringe,
                    phase=phase,
                    curve_times=curve_times,
                    expected_frequency_mhz=None,
                )
            except (
                AnalysisFailed,
                ValueError,
                RuntimeError,
                ArithmeticError,
                np.linalg.LinAlgError,
            ) as exc:
                fit_curve = None
                ax.set_title(f"Fit unavailable: {exc}", wrap=True)
                outcomes.append(MuxFitFailure(trace.index, str(exc)))
            else:
                fit_curve = (curve_times, fitted, outcome)
                outcomes.append(outcome)

            if projected is None:
                continue
            real, sem = projected
            ax.plot(
                trace.times,
                real,
                "o",
                ms=4,
                linestyle="-" if fit_curve is None else "none",
                label="Measured IQ",
            )
            if sem is not None and np.isfinite(sem).any():
                ax.errorbar(
                    trace.times,
                    real,
                    yerr=sem,
                    fmt="none",
                    ecolor="C0",
                    label="Shot SEM",
                )
            if fit_curve is not None:
                _plot_fit(ax, *fit_curve)
            fig.legend(loc="outside lower center", ncols=2)
        return tuple(outcomes)
