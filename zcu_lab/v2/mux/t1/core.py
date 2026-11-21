"""Simultaneous T1 acquisition on one shared delay axis.

Technical reference for lab authors and framework developers.

## Sequence and coordinates

The complete `tones` list retains capture-generated `name` values and original
readout positions. `participants` is a nonempty, unique list of original indices.
`[2, 0]` selects Q3 then Q1, retaining that execution/data order.
Only participants require valid drives. Inactive drives may be None.

`MuxT1Exp` excites active qubits with fixed pi pulses on independent drive generators.
Timeline forks start all pulses together. Each shorter branch waits
for the longest branch to finish. One common delay follows the join, then one
active mux readout. Reset, reps, rounds, initial delay and relax delay are shared.
Pulse pre/post delays remain part of their native branches; no end-alignment
padding or second compilation is added.

`MuxT1Cfg.delay` defines a finite, nonnegative, strictly increasing start/stop
window and point count in microseconds. `sampling="linear"` uses its linear SweepCfg
axis in one hardware loop. The result and fit retain requested times; readout
can differ by native timing ticks. `sampling="decay"` uses the single-qubit T1
sampling policy to concentrate points in the decay region of the same window.
A table sweep and DelayAuto execute one shared delay table for all qubits. Results
and fits then retain quantized delays; duplicate/reordered quantized points
raise ValueError before acquisition.

A short pi pulse adds a fixed `max_len - len_i` wait before the common delay.
For an exponential decay this changes the amplitude, not T1. Pulse delays can
add their own fixed offsets. Calibrate the pulses first and choose a delay window
long enough to constrain the decay and baseline. This core does not implement
T1 with a probe tone or singleshot T1.

## Inputs and ownership

Import the leaf without loading a GUI:

```python
from zcu_lab.v2.support.mux_sweep_cfg import MuxT1ToneCfg
from zcu_lab.v2.mux.t1 import MuxT1Cfg, MuxT1Exp, MuxT1ToneCfg
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment import load_run, save_run
from zcu_tools.experiment.scope import Scope
from zcu_tools.plotting.plots import NonPresentingHost, Plots
from zcu_tools.program.v2 import SweepCfg

# env is a caller-owned compatible hardware or simulation environment.
# full_readout is MuxReadoutCfg; pi_0/pi_1/pi_2 are fixed PulseCfg values.
# Their generators must be independent of each other and readout.gen_ch.
cfg = MuxT1Cfg(
    readout=full_readout,
    tones=[
        MuxT1ToneCfg(name="Q1", drive=pi_0),
        MuxT1ToneCfg(name="Q2", drive=pi_1),
        MuxT1ToneCfg(name="Q3", drive=pi_2),
    ],
    participants=[2, 0],
    delay=SweepCfg(start=0.0, stop=8.0, expts=41, step=0.2),
    reps=128,
    rounds=2,
    initial_delay=1.0,
    relax_delay=40.0,
)
core = MuxT1Exp()
plots = Plots(NonPresentingHost())
scope = Scope("run")
try:
    result = core.run(
        cfg,
        context=RunContext(env.soc, env.soccfg, plots, {}, scope),
    )
    source = RunRecord(cfg=cfg, result=result, metadata=metadata.model_dump(mode="json"))
    fits = core.analyze(source, core.Options(skip=0, dual_exp=False), plots=plots)
    # scope.stop.is_set() reports a stop request, not data completeness.
    # metadata is caller-captured RunMetadata for this source and its completion.
    save_run(source, destination, spec=MuxT1Exp.AXES_SPEC)
    loaded, snapshot = load_run(destination, spec=MuxT1Exp.AXES_SPEC)
finally:
    plots.finish(present=False)
    plots.release()
# The environment owner closes its devices after all operations finish.
```

The full ordered tone list must match the full readout count. Only participants
require valid drives. The nonempty, unique participants list contains original
indices in its own order;
ADC channel order and duplicate labels do not change identity. Cfg validation
rejects missing drives, shared generators, invalid counts and invalid delay axes.
Native pulse and readout owners reject unsupported hardware settings.

The caller owns compatible soc/soccfg, devices, plots and a fresh stop signal.
This core does not initialize external devices, save automatically or close a
caller's environment. For an FTF offline run, configure `readout.mux` with
its own generator and `readout.mux.adc` receiver list. Bind the qubit drives
to independent generators and build FTF against the matching board profile. Small truncations demonstrate execution only and do not
establish physical convergence or calibration accuracy.

## Results, cancellation and analysis

`MuxT1Result.times` is float64 `(P,)` us, requested for uniform sampling and
quantized for nonuniform sampling. `tone_indices` is int64 `(K,)`
of original active indices. `signals` is complex128 `(K,P)`, ADC-length-normalized
average I+jQ. The program averages reps and rounds. Cancellation keeps
completed-round averages across the whole sweep. Before the first completed
round, signals remain NaN. Acquisition, admission and construction errors
propagate directly. Authors decide whether returned data meets their analysis
requirements; StopSignal carries only a stop request.

Live plots use `measurement_i` and fit plots use `fit_i`, where `i` is the original
index, not the label. Analysis reads only the explicit source, permits cfg=None,
and removes `skip` leading points before finite-IQ filtering. It reuses the
single-qubit T1 rotation, single/dual decay fitters and fit-quality calculation.
`MuxT1Fit(index, fit)` contains the existing `T1Analysis`, including microsecond
values/errors, optional second decay and quality under `fit`. A numerical failure
returns `MuxFitFailure(index, reason)` without hiding another tone's outcome.
Malformed source shapes or options raise instead of becoming fit failures.
Inspect quality and uncertainty before using a fitted value as calibration.

Native save/load preserve full cfg, original indices, complex IQ and the one
recorded delay axis. Time is seconds on disk and microseconds in memory. Always pass
the captured `RunRecord`; changing a current cfg does not change the saved source.
Canonical saving requires cfg. External `save_run`/`load_run` use native HDF5;
`save_labber(source, path, spec=MuxT1Exp.AXES_SPEC)` exports Labber separately.

The numerical core does not infer component identity from readout labels.
The adapter below owns component proposals using the captured binding.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Annotated, ClassVar, Literal, Self

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray
from pydantic import Field, StrictInt, model_validator
from pydantic.dataclasses import dataclass as validated_dataclass
from zcu_tools.analysis.failure import AnalysisFailed
from zcu_tools.analysis.fitting import (
    FitQuality,
    compute_fit_quality,
    decay_time_bounds,
    dual_expfunc,
    expfunc,
    fit_decay,
    fit_dual_decay,
)
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime import RunOptions, Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.experiment.v2.utils.t1_sampling import materialize_nonuniform_t1_delays
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    DelayAuto,
    Reset,
    ResetCfg,
    SweepCfg,
    TimelineBuilder,
)
from zcu_tools.program.v2.modules import (
    MuxReadout,
    MuxReadoutCfg,
    Pulse,
)
from zcu_tools.utils.process import rotate2real

from zcu_lab.v2.support import live_line
from zcu_lab.v2.support.acquire import mux_snr_checker
from zcu_lab.v2.support.mux import MuxFitFailure, averaged_mux_signal
from zcu_lab.v2.support.mux_sweep_cfg import MuxDriveToneCfg, validate_mux_tones


@dataclass(frozen=True)
class MuxT1Result:
    """Averaged complex IQ on a common delay axis.

    times is float64 (P,) us, requested for uniform sampling and quantized for
    nonuniform sampling. tone_indices is int64 (K,) of original full-tone indices
    in participants order.
    signals is complex128 (K,P), I+jQ in ADC-length-normalized a.u. Unwritten
    entries remain NaN; completed-round means survive cancellation. The caller
    must check the RunContext stop signal before declaring success.
    """

    times: NDArray[np.float64]
    tone_indices: NDArray[np.int64]
    signals: NDArray[np.complex128]


@validated_dataclass(frozen=True)
class T1AnalyzeOptions:
    """Fit one or two decays after skipping nonnegative leading delay points."""

    dual_exp: Annotated[bool, FieldMeta(label="Dual exponential")] = False
    skip: Annotated[int, FieldMeta(label="Skip leading points"), Field(ge=0)] = 0


@dataclass(frozen=True)
class T1Analysis:
    """Per-tone decay times and one-sigma errors in us, with fit diagnostics.

    t1b and t1b_err hold the second component only for a dual-exponential fit.
    fit_quality maps the selected fit to its numerical quality, or is absent.
    """

    t1: float
    t1_err: float
    t1b: float | None = None
    t1b_err: float | None = None
    fit_quality: Mapping[str, FitQuality] | None = None


@dataclass(frozen=True)
class MuxT1Fit:
    """Original full-tone index and existing single-qubit T1 numerical analysis.

    fit holds t1/t1_err in us, optional second decay component in us and the
    existing fit quality mapping under 'fit'. Delay coordinates, projected IQ
    and predictions stay local to analysis for plotting. It owns no figures
    or source cfg.
    """

    index: int
    fit: T1Analysis


class MuxT1ToneCfg(MuxDriveToneCfg):
    """One original qubit's nullable fixed pi pulse, required only when active."""


class MuxT1Cfg(ConfigBase):
    """Complete named tones, ordered participants and shared acquisition controls.

    reset and readout hold the common modules. tones contains one
    MuxT1ToneCfg per original readout position, with unique nonempty names.
    participants is a nonempty, duplicate-free list of original tone indices; its order
    determines execution/data rows (Q3/Q1 maps to original indices 2/0).
    Selected drives are required/validated on independent
    generators distinct from readout. delay is one common SweepCfg of
    strictly increasing, finite, nonnegative requested us. sampling defaults to
    linear; decay concentrates points within that window
    using a shared cycle table. Fits retain requested times for uniform sampling,
    quantized times otherwise.
    Explicit reps/rounds are positive; initial_delay/relax_delay are finite
    nonnegative us. No capture, bindings or writeback destinations are stored.
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
            description="Enter the calibrated multiplexed readout used after the qubit drive sequence.",
        ),
    ]
    tones: Annotated[list[MuxT1ToneCfg], FieldMeta(element_label="name")]
    participants: Annotated[
        list[StrictInt],
        FieldMeta(
            description="Enter the original zero-based tone indices in measurement order."
        ),
    ]
    delay: Annotated[
        SweepCfg,
        FieldMeta(
            label="Delay",
            unit="µs",
            description="Enter the wait after the π pulse in µs.",
        ),
    ]
    sampling: Annotated[
        Literal["linear", "decay"],
        FieldMeta(
            label="Delay spacing",
            description="Choose even spacing or denser early points to resolve a decay.",
        ),
    ] = "linear"

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
    def validate_scan(self) -> Self:
        """Reject invalid counts, channels and requested delay coordinates."""
        validate_mux_tones(self, self.tones, self.participants)
        times = sweep2array(self.delay, "none")
        if np.any(times < 0) or np.any(np.diff(times) <= 0):
            raise ValueError("delays must be nonnegative increasing µs")
        return self


def _validate_result(result: MuxT1Result) -> None:
    """Raise ValueError for invalid source axes, indices or IQ shape/dtype.

    result supplies float64 increasing nonnegative times (P,), unique int64
    original indices (K,) and complex128 IQ (K,P), with K,P>0. Return None
    without mutation when those axes match; partial NaN IQ remains allowed.
    """
    times, indices, signals = result.times, result.tone_indices, result.signals
    if (
        times.ndim != 1
        or times.dtype != np.dtype(np.float64)
        or not times.size
        or not np.all(np.isfinite(times))
        or np.any(times < 0)
        or np.any(np.diff(times) <= 0)
        or indices.ndim != 1
        or indices.dtype != np.dtype(np.int64)
        or not indices.size
        or np.any(indices < 0)
        or len(np.unique(indices)) != len(indices)
        or signals.dtype != np.dtype(np.complex128)
        or signals.shape != (len(indices), len(times))
    ):
        raise ValueError("invalid MuxT1Result axes or IQ shape/dtype")


_DELAY_LABEL = "Wait after π pulse (µs)"
_IQ_LABEL = "Principal-axis mean IQ (a.u.)"


def _show_live_t1(plots: Plots, cfg: MuxT1Cfg, times: NDArray[np.float64]):
    viewers = []
    for index in cfg.participants:
        fig, ax = plots.subplots(num=f"measurement_{index}")
        fig.suptitle(f"Mux T1 · {cfg.tones[index].name} · IQ contrast")
        ax.set(xlabel=_DELAY_LABEL, ylabel=_IQ_LABEL)
        progress = ax.set_title(f"Acquired delay points: 0 / {times.size}")
        viewer = live_line(ax)
        ax.lines[0].set(marker="o", markersize=3, label="Measured IQ")
        ax.legend()
        viewers.append((viewer, progress))
        fig.show()

    @plots.on_plot_thread
    def update(data: NDArray[np.complex128]):
        artists = []
        for row, (viewer, progress) in enumerate(viewers):
            measured = data[:, row]
            acquired = np.isfinite(measured)
            real = rotate2real(np.where(acquired, measured, np.nan)).real
            progress.set_text(
                f"Acquired delay points: {np.count_nonzero(acquired)} / {times.size}"
            )
            artists.extend((*viewer.set(times, real), progress))
        return artists

    return update


def _plot_t1_fit(
    ax: Axes,
    xs: NDArray[np.float64],
    real: NDArray[np.float64],
    params: Sequence[float],
    outcome: MuxT1Fit,
) -> None:
    t1, err = outcome.fit.t1, outcome.fit.t1_err
    t1b, errb = outcome.fit.t1b, outcome.fit.t1b_err
    quality = (
        None if outcome.fit.fit_quality is None else outcome.fit.fit_quality["fit"]
    )
    ax.plot(xs, real, "o", ms=3, label="Measured IQ")
    model = dual_expfunc if t1b is not None else expfunc
    label = "Double exponential" if t1b is not None else "Exponential decay"
    if xs.min() <= t1 <= xs.max():
        ax.axvline(t1, color="0.4", ls=":", label="T1")
    result_label = f"T1 = {t1:.4g} ± {err:.2g} µs"
    if t1b is not None and errb is not None:
        result_label += f" · Short decay = {t1b:.4g} ± {errb:.2g} µs"
    curve = np.linspace(xs.min(), xs.max(), 400)
    ax.plot(curve, model(curve, *params), color="C1", label=label)
    r2 = None if quality is None else quality.r2
    rms = None if quality is None else quality.normalized_residual_rms
    quality_label = f"R² = {r2:.3g}" if r2 is not None else "R² unavailable"
    quality_label += (
        f" · Normalized residual RMS = {rms:.3g}"
        if rms is not None
        else " · Normalized residual RMS unavailable"
    )
    ax.set_title(f"{result_label}\n{quality_label}", fontsize=10)


def fit_and_plot_t1(
    result: MuxT1Result,
    options: T1AnalyzeOptions,
    *,
    plots: Plots,
    tone_names: tuple[str, ...] | None = None,
) -> tuple[MuxT1Fit | MuxFitFailure, ...]:
    """Validate IQ axes, raise on invalid sources, and fit and plot each tone T1.

    Require float64 nonnegative strictly increasing times (P,), int64 unique
    nonnegative original indices (K,) and complex128 signals (K,P), with K,P>0.
    NaN IQ is allowed for partial acquisition. Invalid sources and negative or
    noninteger skip raise ValueError before fitting. Existing single/dual decay
    fitters and quality consume only acquired points. Numeric failures return
    indexed MuxFitFailure reasons; plotting and resource errors propagate.
    plots owns fit_i, independent of duplicate readout labels; caller
    finishes/releases figures. Draw only the selected primary model and return
    per-tone fit summaries. Acquired delays, projected observations and
    predictions remain local to plotting. tone_names supplies original-index
    display names when the saved configuration is available. No hardware is read.
    """
    _validate_result(result)
    times, indices, signals = result.times, result.tone_indices, result.signals
    outcomes: list[MuxT1Fit | MuxFitFailure] = []
    for row, raw_index in enumerate(indices):
        index = int(raw_index)
        signal = signals[row, options.skip :]
        finite = np.isfinite(signal)
        xs = times[options.skip :][finite]
        signal = signal[finite]
        tone = f"Tone {index}" if tone_names is None else tone_names[index]
        fig, ax = plots.subplots(num=f"fit_{index}", figsize=(7, 5))
        ax.set(xlabel=_DELAY_LABEL, ylabel=_IQ_LABEL)
        fig.suptitle(f"Mux T1 · {tone} · IQ contrast")
        try:
            minimum = 6 if options.dual_exp else 4
            if len(xs) < minimum:
                raise ValueError(f"T1 fit requires at least {minimum} acquired points")
            # Project acquired IQ onto its principal contrast axis.
            real = rotate2real(signal).real
            if options.dual_exp:
                fit = fit_dual_decay(xs, real)
                t1, err = fit.decay_time1, fit.decay_time1_err
                t1b, errb = fit.decay_time2, fit.decay_time2_err
                names = ("y0", "yscale1", "decay_time1", "yscale2", "decay_time2")
            else:
                fit = fit_decay(xs, real)
                t1, err = fit.decay_time, fit.decay_time_err
                names = ("y0", "yscale", "decay_time")
                t1b, errb = None, None
            fitted, params, covariance = fit.curve, fit.params, fit.cov
            if not np.all(
                np.isfinite(
                    np.asarray(
                        (t1, err, t1b, errb) if options.dual_exp else (t1, err),
                        dtype=np.float64,
                    )
                )
            ):
                raise ValueError("T1 fit returned nonfinite estimates")
            quality = compute_fit_quality(
                real,
                fitted,
                {name: float(value) for name, value in zip(names, params, strict=True)},
                covariance,
                decay_bounds=decay_time_bounds(xs),
            )
            outcome = MuxT1Fit(
                index,
                T1Analysis(float(t1), float(err), t1b, errb, {"fit": quality}),
            )
        except (
            AnalysisFailed,
            ValueError,
            RuntimeError,
            ArithmeticError,
            np.linalg.LinAlgError,
        ) as exc:
            outcomes.append(MuxFitFailure(index, str(exc)))
            ax.plot(xs, rotate2real(signal).real, "o", ms=3, label="Measured IQ")
            ax.text(
                0.02,
                0.98,
                f"Fit unavailable: {exc}",
                transform=ax.transAxes,
                va="top",
                wrap=True,
            )
        else:
            _plot_t1_fit(ax, xs, real, tuple(params), outcome)
            outcomes.append(outcome)
        ax.legend()
        fig.tight_layout()
    return tuple(outcomes)


class MuxT1Exp(Experiment[MuxT1Result, MuxT1Cfg]):
    """Measure selected qubits' energy relaxation on one shared delay axis.

    Each point applies simultaneous pi pulses, waits for the longest branch,
    then waits a shared delay before a joint mux readout. Saved results retain
    recorded delays, original tone indices, complex IQ and the full cfg.
    """

    AXES_SPEC: ClassVar[AxesSpec[MuxT1Result, MuxT1Cfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis("times", label="Delay", unit="s", scale=1e-6),
            AxesSpec.Axis("tone_indices", label="Tone", unit="", dtype=np.int64),
        ),
        z=AxesSpec.Z("signals", label="Signal", unit="a.u."),
        result_type=MuxT1Result,
        cfg_type=MuxT1Cfg,
        cfg_schema_version="4.0",
        tag="mux/t1",
    )

    Options: ClassVar[type[T1AnalyzeOptions]] = T1AnalyzeOptions

    def run(self, config: MuxT1Cfg, *, context: RunContext) -> MuxT1Result:
        """Acquire pi pulse -> shared wait -> mux readout for each delay in us.

        config supplies the full tone list, ordered participants, pi pulses,
        reset/readout and averaging controls; it is deep-copied before use.
        The join waits for the longest pi branch, including its pulse delays, before
        the shared wait. Return times (P,), original tone indices (K,) and
        averaged complex IQ (K,P), with one live figure per tone. sampling="linear"
        retains requested times; "decay" returns quantized nonuniform
        delays in the same window using the single-qubit T1 sampling policy.
        Both paths use one hardware loop. Nonuniform quantization collisions
        raise ValueError before acquisition. Cancellation retains
        completed-round averages across the sweep, or NaN before any round finishes.
        context supplies offline or hardware handles, plots and a stop signal;
        acquisition and program errors propagate.

        """
        # 1. Dispatch linear sampling to the shared hardware delay sweep.
        cfg = config.model_copy(deep=True)
        if cfg.sampling == "linear":
            return self._run_linear(cfg, context=context)
        # 2. Dispatch decay sampling to the quantized nonuniform hardware delay table.
        return self._run_decay(cfg, context=context)

    def _run_linear(self, cfg: MuxT1Cfg, *, context: RunContext) -> MuxT1Result:
        """Acquire the common linear delay sweep and return tone-major IQ."""
        # 1. Show each participant's decay magnitude on the common requested delay axis.
        times = sweep2array(cfg.delay, "none")
        indices = list(cfg.participants)
        readout = cfg.readout.with_updates(
            tones=[cfg.readout.tones[i] for i in indices]
        )
        update = _show_live_t1(context.plots, cfg, times)

        # 2. Sweep the shared wait after all pi branches finish, then acquire mux readout.
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
                    stop_condition=mux_snr_checker(
                        signals, cfg.earlystop_snr, tone_axis=1
                    ),
                ),
            )
        signals.trigger_update(flush=True)
        # 3. Keep completed-round IQ for every delay; NaN marks no completed round.
        return MuxT1Result(
            times, np.asarray(indices, dtype=np.int64), signals.array.T.copy()
        )

    def _run_decay(self, cfg: MuxT1Cfg, *, context: RunContext) -> MuxT1Result:
        """Acquire the nonuniform delay table and return tone-major IQ."""
        # 1. Show each participant's decay on the collision-checked, quantized nonuniform delays.
        table = materialize_nonuniform_t1_delays(cfg.delay, soccfg=context.soccfg)
        times = table.times_us
        indices = list(cfg.participants)
        readout = cfg.readout.with_updates(
            tones=[cfg.readout.tones[i] for i in indices]
        )
        update = _show_live_t1(context.plots, cfg, times)

        # 2. Step the hardware delay table after all pi branches finish, then acquire mux readout.
        signals = SignalBuffer((len(times), len(indices)), on_update=update)
        with Schedule(signals, scope=context.scope) as sched:
            builder = TimelineBuilder(
                context.soccfg,
                reps=cfg.reps,
                initial_delay=cfg.initial_delay,
                relax_delay=cfg.relax_delay,
            )
            delay = builder.add_table_sweep("delay_idx", table.times_us.tolist())
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
            # Start the shared delay after the full pulse branches, including post.
            builder.delay_by_timeline()
            builder.emit(DelayAuto("t1_delay", t=delay))
            readout_module = MuxReadout("readout", readout)
            builder.emit(readout_module)
            prog = builder.build()
            sched.run_acquire(
                prog,
                context.soc,
                rounds=cfg.rounds,
                options=RunOptions(
                    converter=partial(averaged_mux_signal, prog, readout_module),
                    stop_condition=mux_snr_checker(
                        signals, cfg.earlystop_snr, tone_axis=1
                    ),
                ),
            )
        signals.trigger_update(flush=True)
        # 3. Retain actual table delays and completed-round IQ without filling missing values.
        return MuxT1Result(
            times, np.asarray(indices, dtype=np.int64), signals.array.T.copy()
        )

    def analyze(
        self,
        source: RunRecord[MuxT1Cfg, MuxT1Result],
        options: T1AnalyzeOptions,
        *,
        plots: Plots,
    ) -> tuple[MuxT1Fit | MuxFitFailure, ...]:
        """Fit each saved IQ row against its recorded delays to estimate T1 in us.

        source is a Run/Load record; its cfg may be absent. options selects
        initial points to skip and single or dual exponential decay. Return
        per-tone fits with uncertainties and quality, or indexed fit failures;
        invalid axes/options raise. Acquired delays, projected observations
        and predictions stay local to analysis. Draw only the selected primary
        model as fit_i.

        """
        # 1. Fit saved acquired delays per tone using the requested skip and single/dual decay policy.
        names = (
            None
            if source.cfg is None
            else tuple(tone.name for tone in source.cfg.tones)
        )
        return fit_and_plot_t1(source.result, options, plots=plots, tone_names=names)
