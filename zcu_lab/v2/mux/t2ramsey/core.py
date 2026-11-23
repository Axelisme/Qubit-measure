"""Mux Ramsey core: common requested delay, independent x90 pulses and detunes.

Technical reference for lab authors and framework developers.

`MuxT2RamseyExp.run(config, *, context)` measures selected qubits in one program
with independent x90 drives and a common mux readout. It has no Form, recipe,
catalog registration or writeback proposal.

## Configuration and timing

`participants` is a nonempty, unique list of original indices in the complete `tones` list.
`[2, 0]` selects Q3 then Q1, retaining that execution/data order.
Only participants require valid drives on independent generators. Inactive
drives may be None. Every tone retains its name and readout identity; count
and order match the fixed capture.

Import `MuxT2RamseyCfg`, `MuxT2RamseyExp` and `MuxCoherenceToneCfg` from
`zcu_lab.v2.mux.t2ramsey`. The cfg contains:

- `modules: MuxDriveModuleCfg`, with optional common reset and full mux readout.
- `tones: list[MuxCoherenceToneCfg]`, one element per original readout tone.
  Each element has a fixed `name`, required `drive` and `detune` in MHz.
  `drive` is the source x90 pulse, not x180. Only selected drives are required; inactive drives may be None.
- `delay: SweepCfg`, one common requested free-evolution delay in us.
- Shared `reps`, `rounds`, `initial_delay` and `relax_delay`, with times in us.

Participants must be nonempty. All drives must use independent generators,
different from the readout generator. Pulse parameters must be fixed and finite.
The delay bounds must be finite and nonnegative. Numeric cfg validation does not
resolve component identities; the caller supplies valid tone/qubit bindings.

The sequence is reset, parallel first x90 pulses, common delay, parallel final x90
pulses, then active mux readout. Each fork starts its branches at the same stage
origin. Native pulse pre/post delays remain unchanged. The Delay starts after the
longest first stage, not independently after each qubit's own pulse.
For zero pre/post delays, a shorter pulse waits an extra fixed
`max_len - len_i`. This can change fringe amplitude and constant phase.
The analysis still uses the common requested delay. Enable free fit phase when
a constant phase offset matters. Native clock differences of a few ticks remain;
the core does not end-align, pad, double-compile or save compiled actual delays.

Each final x90 phase uses its source phase plus its own nominal detune times the
shared delay parameter. Program/sweep owners perform native conversion.
`actual_detune(..., time_scaler=1.0)` owns the realized MHz detune estimate.
The estimate preserves that owner's phase-wrap convention, including for signed
detunes; do not treat it as a signed unwrapped frequency correction without
checking the acquisition range.

## Run and analysis

```python
from zcu_tools.experiment.records import RunRecord
from zcu_lab.v2.mux.t2ramsey import MuxT2RamseyExp

core = MuxT2RamseyExp()
result = core.run(cfg, context=context)
source = RunRecord(cfg=cfg, result=result)
outcomes = core.analyze(source, core.Options(fit_phase=True), plots=analysis_plots)
```

RunContext supplies compatible soc/soccfg, a fresh StopSignal and caller-owned
Plots. The caller finishes/releases plots and cleans up devices. Run does not
save files or cache a previous result.

`MuxCoherenceResult` contains original active `tone_indices (K,)`, common requested
`times (P,)` in us, `signals (K,P)` as ADC-length-normalized complex average IQ,
`actual_detunes (K,)` and `source_frequencies (K,)` in MHz. Source frequencies are
the supplied x90 carriers before the phase ramp; full pulse settings remain in cfg.
Actual detunes may be NaN when a nonzero nominal detune has no positive rounded
delay. The core preserves the owner's diagnostics and NaN, without substituting a
nominal rate. Save/load and analysis still accept this source.

The hardware sweeps delay and averages reps/rounds. Cancellation keeps the mean of
completed rounds across the whole sweep. With no completed round, IQ stays NaN.
Acquisition, build, validation and conversion errors propagate directly.
A stop request does not classify data completeness; authors decide which
completed-round averages are usable.

Analysis accepts explicit RunRecord, including cfg=None. `fit_fringe=True`
selects the existing damped-cosine fitter; False selects exponential decay.
`fit_phase=False` uses the single-qubit nearest-zero 0/180-degree phase policy.
True frees fringe phase. Decay ignores fit_phase. Missing IQ is filtered per tone;
a fringe fit needs six points and a decay fit four.

Each original index returns `MuxCoherenceFit` with T2* and its error in us,
fringe rate and its error in MHz, selected model and FitQuality. Numeric failures
return `MuxFitFailure(index, reason)` without losing other tones. Source contract
errors raise instead of becoming fit failures. Graphs are named
`measurement_i` and `fit_i`, so duplicate labels do not overwrite them.
A finite fit is not a calibration-quality guarantee.

## Native source persistence

`save_run(source, path, spec=MuxT2RamseyExp.AXES_SPEC)` validates source.metadata
against RunMetadata and stores full cfg, inactive tones, original
indices, shared requested delay, IQ, actual detunes and source frequencies.
Available shot moments are stored on the same tone and delay grid as IQ.
Native files without moments load as mean-only records and Analyze omits bars.
Time and frequency use seconds and Hz on disk, us and MHz in memory.
`save_run` writes only native data and returns None. `save(source, path)` is
the separate Labber export. The grouped axes spec declares the Labber
`independent` layout. Both formats preserve IQ on the delay/tone grid and detune
and source frequency on the tone-only grid. Labber does not export shot moments.
The Labber writer uses the declared grids without broadcasting per-tone values,
splitting files or falling back to another layout.
For multi-output saving and per-output SaveReport status/errors, see
`save_artifacts` in lib/zcu_tools/experiment/README.md, section "實驗介面與資料".
Recipe completion alone does not prove that every output saved.
`load_run(path)` restores an explicit source; changing a current cfg does not
retarget or alter its analysis. The caller sets complete/partial/stopped metadata
from the run outcome, not from the existence of IQ.

This core produces no proposal. A later calibration caller must require a usable
fringe fit, source frequency and actual detune before considering a Ramsey
frequency correction. NaN detune does not meet that requirement.
"""

from __future__ import annotations

from typing import Annotated, ClassVar

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
from zcu_tools.experiment.v2.runtime import Schedule, SignalBuffer
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
_DELAY_LABEL = "Requested free-evolution delay (µs)"


@validated_dataclass(frozen=True)
class T2RamseyAnalyzeOptions:
    """Select fringe fitting and optionally free its phase; decay ignores phase."""

    fit_fringe: Annotated[
        bool,
        FieldMeta(
            label="Fit fringe",
            description="On fits a damped cosine; off fits an exponential decay.",
        ),
    ] = True
    fit_phase: Annotated[
        bool,
        FieldMeta(
            label="Fit phase offset (fringe only)",
            description="On fits the fringe phase offset; off uses the signal at the delay closest to zero to choose 0° or 180°.",
        ),
    ] = True


class MuxT2RamseyCfg(ConfigBase):
    """Named x90 drives and a shared delay sweep for mux Ramsey fringes.

    modules holds the optional reset and full mux readout. tones follows the
    original readout order; every tone has a unique nonempty name, a nullable
    x90 drive, and a MHz detune for its final phase ramp. Only participants
    require drives on independent generators. Selected drives cannot share
    the readout generator. participants is a nonempty list of unique original indices
    in result-row order (Q3/Q1 selects original positions 2/0).
    delay is the common finite, nonnegative requested-us sweep. Explicit reps
    and rounds are positive; initial_delay and relax_delay are finite,
    nonnegative us before acquisition and between repetitions respectively.
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
    tones: Annotated[list[MuxCoherenceToneCfg], FieldMeta(element_label="name")]
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
            description="Sweep the wait between the π/2 pulses; Expts counts the delay points.",
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
    def validate_scan(self) -> MuxT2RamseyCfg:
        """Reject invalid full tones, fixed drives, channels and delay bounds."""
        validate_coherence_scan(self, self.tones, self.delay, self.participants)
        return self


def _show_live(plots: Plots, cfg: MuxT2RamseyCfg, times: NDArray[np.float64]):
    """Show one live figure per cfg participant and return its shared updater.

    times contains requested free-evolution delays in µs. The updater takes completed-round
    sums shaped (delays, participants, 6) and returns the changed data, SEM and
    progress artists. Projection and plot update errors propagate.
    """
    viewers = []
    empty = times[:0]
    for index in cfg.participants:
        fig, ax = plots.subplots(num=f"measurement_{index}", layout="constrained")
        fig.suptitle(f"Mux Ramsey · {cfg.tones[index].name} · IQ contrast")
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
            real, sem = project_iq(means[:, row], data[:, row])
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
    """Add the fit curve, in-window T2* marker and fit facts to ax.

    times contains increasing requested delays in µs. fitted contains the model
    values on that grid; outcome supplies the model kind, estimates and quality.
    Update ax in place; return None without presenting or refreshing the figure.
    """
    model = "Damped cosine" if outcome.fit_fringe else "Exponential decay"
    ax.plot(times, fitted, color="C1", label=model)
    if times[0] <= outcome.t2 <= times[-1]:
        ax.axvline(outcome.t2, color="0.4", ls=":", label="T2*")
    estimate = f"T2* = {outcome.t2:.4g} ± {outcome.t2_err:.2g} µs"
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


class MuxT2RamseyExp(Experiment[MuxCoherenceResult, MuxT2RamseyCfg]):
    """Measure mux Ramsey fringes and fit each participating qubit's T2*."""

    AXES_SPEC: ClassVar[GroupedAxesSpec[MuxCoherenceResult, MuxT2RamseyCfg]] = (
        coherence_spec(MuxT2RamseyCfg, tag="mux/time_domain/t2ramsey")
    )

    Options: ClassVar[type[T2RamseyAnalyzeOptions]] = T2RamseyAnalyzeOptions

    def run(self, config: MuxT2RamseyCfg, *, context: RunContext) -> MuxCoherenceResult:
        """Measure x90 → shared delay → phase-ramped x90 → mux readout.

        config supplies named x90 drives, MHz detunes and a requested-us delay
        sweep; it is copied before use. Each pulse stage starts together, and
        the delay begins after the longest first pulse. Return requested times
        (P,), average complex IQ (K,P), original tone indices, actual MHz detunes
        and source frequencies in participant order, plus completed-round IQ
        sums (K,P,6) for this Run's plots. context supplies hardware
        and plots. Cancellation keeps completed-round means and NaN for no data.
        Acquisition, validation, build and conversion errors propagate.
        """
        # 1. Keep participant order while selecting drives and mux readout tones.
        cfg = config.model_copy(deep=True)
        indices = list(cfg.participants)
        tones = [cfg.tones[index] for index in indices]
        drives = [tone.require_drive() for tone in tones]
        times = sweep2array(cfg.delay, "none")
        _ = sweep2array(cfg.delay, "time", {"soccfg": context.soccfg, "scaler": 1.0})
        detunes = np.asarray(
            [
                actual_detune(
                    cfg.delay,
                    detune=tone.detune,
                    gen_ch=drive.ch,
                    soccfg=context.soccfg,
                    time_scaler=1.0,
                )
                for tone, drive in zip(tones, drives, strict=True)
            ],
            dtype=np.float64,
        )
        frequencies = np.asarray(
            [float(drive.freq) for drive in drives], dtype=np.float64
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
            for index, drive, timeline in zip(indices, drives, timelines, strict=True):
                builder.emit(
                    Pulse(
                        f"pi2_first_{index}",
                        drive,
                    ),
                    timeline=timeline,
                )
            builder.join(*timelines, into="default")
            builder.timeline_inc(delay)
            builder.delay_by_timeline()
            timelines = [f"pi2_final_{index}" for index in indices]
            builder.fork("default", into=timelines)
            for index, tone, drive, timeline in zip(
                indices, tones, drives, timelines, strict=True
            ):
                builder.emit(
                    Pulse(
                        f"pi2_final_{index}",
                        drive,
                        phase=drive.phase + 360 * tone.detune * delay,
                    ),
                    timeline=timeline,
                )
            builder.join(*timelines, into="default")
            readout_module = MuxReadout("readout", readout)
            builder.emit(readout_module)

            # One summed acquisition supplies the IQ mean and shot SEM.
            prog = builder.build()
            _ = sched.run_acquire_summed(
                prog,
                context.soc,
                reducer=mux_iq_sums(prog, "readout"),
                rounds=cfg.rounds,
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
        source: RunRecord[MuxT2RamseyCfg, MuxCoherenceResult],
        options: T2RamseyAnalyzeOptions,
        *,
        plots: Plots,
    ) -> tuple[MuxCoherenceFit | MuxFitFailure, ...]:
        """Fit each qubit's Ramsey decay and fringe rate from source.

        source contains requested delays and measured IQ, with or without cfg.
        options.fit_fringe selects a damped cosine instead of a decay;
        fit_phase frees its phase instead of choosing 0/180 degrees from the
        nearest-zero signal. Return T2* (us), fringe rate (MHz), uncertainties
        and fit quality, or a failure reason, for each original tone index.
        Sampling quality uses acquired delays and finite actual_detunes as known
        alias-risk lower bounds. Unknown actual detune leaves a fitted-rate-only
        assessment, which cannot exclude aliasing.
        Draw fit_i with any available projected IQ; on projection failure show
        the reason without data. Invalid sources raise.
        """
        # 1. Validate the saved source before opening per-tone fit figures.
        data = source.result
        validate_coherence_result(data)
        validate_iq_moments(data.iq_moments, data.signals)
        outcomes: list[MuxCoherenceFit | MuxFitFailure] = []
        for row, raw_index in enumerate(data.tone_indices):
            trace = prepare_coherence_trace(
                int(raw_index), data.times, data.signals[row], skip=0
            )
            finite = np.isfinite(data.signals[row])
            moments = None if data.iq_moments is None else data.iq_moments[row][finite]
            name = (
                f"Tone {raw_index}"
                if source.cfg is None
                else source.cfg.tones[trace.index].name
            )
            fig, ax = plots.subplots(
                num=f"fit_{trace.index}", figsize=(8, 5), layout="constrained"
            )
            ax.set(xlabel=_DELAY_LABEL, ylabel=_IQ_LABEL)
            fig.suptitle(f"Mux Ramsey · {name} · IQ contrast")
            projected = None
            try:
                projected = project_iq(trace.signal, moments)
                real, _ = projected
                if trace.times.size < (6 if options.fit_fringe else 4):
                    raise ValueError("Insufficient acquired points for coherence fit")
                curve_times = np.linspace(trace.times.min(), trace.times.max(), 400)

                # 2. Fix Ramsey phase from the nearest-zero acquired delay unless free.
                phase = None
                if options.fit_fringe and not options.fit_phase:
                    zero = real[np.argmin(np.abs(trace.times))]
                    phase = 0.0 if zero > 0.5 * (np.max(real) + np.min(real)) else 180.0
                outcome, fitted = fit_coherence_trace(
                    trace,
                    real,
                    fit_fringe=options.fit_fringe,
                    phase=phase,
                    curve_times=curve_times,
                    expected_frequency_mhz=(
                        float(source.result.actual_detunes[row])
                        if np.isfinite(source.result.actual_detunes[row])
                        else None
                    ),
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
