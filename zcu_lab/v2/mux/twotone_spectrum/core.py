"""Acquisition, analysis and persistence for zipped mux spectroscopy.

Technical reference for lab authors and framework developers.
MuxFreqExp.run(config, *, context) drives selected qubits simultaneously.

Configuration and timing
------------------------
The full tones list matches the complete MuxReadoutCfg in count and order.
Each MuxFreqToneCfg has a fixed name. Only participants require and use a drive
and freq bounds. Participants is a nonempty, unique list of original indices.
Q3/Q1 selects original indices [2, 0] from Q1/Q2/Q3 and retains that execution/data
order. Cfg rejects active drives sharing the readout generator. Program build
rejects overlapping active drive pulses on the same generator. Inactive tones
retain drafts and are not subject to these active generator checks.
The caller supplies the complete qubit/resonator bindings; numeric cfg validation
does not resolve component identities.

Sweep ranges may differ; cfg.expts supplies one positive point count shared by
selected tones. Point p pairs each active tone's p-th value,
not a Cartesian product. One program sweeps drive DAC frequencies in a shared hardware loop.
Each point executes optional common reset, parallel emission of independent drive pulses,
then active fixed-frequency mux readout. Branches start together;
short branches wait for the longest branch at that point. Pre/post delays stay
unchanged. There is no end alignment, end padding or sweep-wide maximum padding.
All participants share reps, rounds, initial_delay and relax_delay.
Program modules perform native conversion and averaging.

Results and cancellation
------------------------
MuxSweepResult.tone_indices is int64 (K,) in participant order. Coordinates are
float64 (K,P) on each drive's own rounded grid. Signals are complex128 (K,P),
ADC-length-normalized average IQ in a.u. The measurement_i magnitude plots use
original tone indices and each tone's own coordinates, so duplicate labels retain
distinct plots. Cancellation preserves completed-round averages across the
whole sweep. Before the first completed round, signals remain NaN.

Frequency scan and analysis
--------------------------
MuxFreqCfg.reset is optional; MuxFreqCfg.readout is the full mux readout.
Sweep bounds and returned coordinates are in MHz. prepare_frequency_scan
copies cfg and constructs independent rounded drive-frequency axes in
participant order. Run resolves original positions from names, not a MuxGroup.

Analysis uses the single-qubit Lorentzian or sinc fitter. Each curve needs at
least six valid points. Each original index returns MuxFreqFit or a reason-bearing
MuxFitFailure and a fit_i figure. Frequency, FWHM and their errors are in MHz.
FitQuality describes that model fit rather than calibration approval.

Native persistence retains full cfg, original indices, independent coordinates
and complex IQ. Frequency coordinates use Hz on disk and MHz in memory.

Calling and ownership
---------------------
For context, plots, stop/error handling and record/persistence ownership, see
lib/zcu_tools/experiment/README.md, section "實驗介面與資料", and
lib/zcu_tools/experiment/v2/runtime/README.md, section "Stop 與 Retry".

Analysis uses the explicit historical source, not the current Form or library.
The numerical core leaves component bindings and writeback to its caller.

Example with validated cfg, RunContext and fresh analysis_plots::

    from zcu_tools.experiment.records import RunRecord
    from zcu_lab.v2.mux.twotone_spectrum import MuxFreqExp

    core = MuxFreqExp()
    result = core.run(cfg, context=context)
    source = RunRecord(cfg=cfg, result=result)
    outcomes = core.analyze(source, core.Options(), plots=analysis_plots)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Annotated, ClassVar, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, StrictInt, model_validator
from pydantic.dataclasses import dataclass as validated_dataclass
from qick import QickConfig
from zcu_tools.analysis.failure import AnalysisFailed
from zcu_tools.analysis.fitting import (
    FitQuality,
    compute_fit_quality,
    fit_qubit_freq,
    lorfunc,
    sincfunc,
)
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import Experiment
from zcu_tools.experiment.axes_spec import GroupedAxesSpec
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.utils.sweep import make_sweep
from zcu_tools.experiment.v2.runtime import RunOptions, Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import Reset, ResetCfg, TimelineBuilder, sweep2param
from zcu_tools.program.v2.modules import (
    MuxReadout,
    MuxReadoutCfg,
    Pulse,
)
from zcu_tools.utils.process import minus_background, rotate2real

from zcu_lab.v2.support import live_line
from zcu_lab.v2.support.mux import (
    MuxFitFailure,
    MuxSweepResult,
    averaged_mux_signal,
    validate_mux_result,
)
from zcu_lab.v2.support.mux_persistence import mux_sweep_spec
from zcu_lab.v2.support.mux_sweep_cfg import (
    MuxDriveToneCfg,
    ToneRangeCfg,
    validate_mux_scan,
)

_SWEEP_LABEL = "Qubit drive RF frequency (MHz)"
_IQ_LABEL = "|Background-subtracted mean IQ| (a.u.)"


@validated_dataclass(frozen=True)
class FreqAnalyzeOptions:
    """Line shape for each spectrum: lor magnitude or signed sinc projection."""

    model_type: Annotated[Literal["lor", "sinc"], FieldMeta(label="Model type")] = "lor"


@dataclass(frozen=True)
class MuxFreqFit:
    """Successful original tone index and MHz spectroscopy estimates.

    freq/fwhm are transition frequency and linewidth. freq_err/fwhm_err are
    their one-sigma uncertainties. quality reports residuals and fit reliability.
    """

    index: int
    freq: float
    freq_err: float
    fwhm: float
    fwhm_err: float
    quality: FitQuality


@dataclass(frozen=True)
class MuxFreqScan:
    """Selected tone positions and hardware-rounded frequency axes for a zip scan.

    indices is a nonempty sequence of unique original tone positions, in
    participant order. coordinates is float64 (K,P) in MHz, with K selected
    tones and P sweep points. Each row uses that drive's hardware grid;
    columns advance together by point index, not as a Cartesian product.
    This record contains no pulses or acquisition state. Treat values as read-only.
    """

    indices: Sequence[int]
    coordinates: NDArray[np.float64]


def qubfreq_signal2real(
    signals: NDArray[np.complex128], *, model_type: Literal["lor", "sinc"] = "lor"
) -> NDArray[np.float64]:
    """Convert complex IQ to contrast; raise ValueError for invalid spectra or model.

    Subtract component-wise median background without mutating signals. ``lor``
    returns the legacy absolute magnitude, also used by the Run live preview.
    ``sinc`` projects onto the IQ principal axis with the largest absolute
    excursion positive, retaining opposite-signed sidelobes instead of folding
    them. The axis has no absolute physical sign; this convention makes the
    dominant resonance positive at any fixed IQ rotation.

    Return the same shape, preserving NaNs in partial spectra. Constant or
    single-acquired-point spectra have zero contrast; all-NaN spectra remain
    NaN. Raise ValueError for an unknown model, empty/non-1-D input, non-complex128
    dtype or infinities. Principal-axis numerical failures propagate.
    """
    if model_type not in ("lor", "sinc"):
        raise ValueError(f"Invalid frequency model: {model_type!r}")
    if signals.ndim != 1 or signals.size == 0 or signals.dtype != np.complex128:
        raise ValueError("Expected a nonempty complex128 spectrum of shape (N,)")
    if np.any(np.isinf(signals)):
        raise ValueError("Frequency spectrum must not contain infinities")

    centered = minus_background(signals)
    if model_type == "lor":
        return np.abs(centered)

    projected = rotate2real(centered).real
    acquired = projected[~np.isnan(projected)]
    if acquired.size and acquired[np.argmax(np.abs(acquired))] < 0:
        projected = -projected
    return projected


class MuxFreqToneCfg(MuxDriveToneCfg):
    """One named drive and its independent freq bounds in MHz."""

    freq: Annotated[
        ToneRangeCfg | None,
        FieldMeta(
            label="Frequency",
            unit="MHz",
            description="Enter this tone's qubit drive RF frequency range in MHz.",
        ),
    ] = None

    def require_freq(self) -> ToneRangeCfg:
        """Return active freq bounds, or raise ValueError naming this tone."""
        if self.freq is None:
            raise ValueError(f"{self.name}: active tone requires freq")
        return self.freq


class MuxFreqCfg(ConfigBase):
    """Settings for simultaneous qubit frequency sweeps and mux readout.

    reset is optional and may be None. readout contains the full readout tone list.
    tones gives each original readout position a unique name.
    participants selects a nonempty, unique list of original indices in measurement order.
    Only participants require and use a drive and a MHz sweep.
    Cfg rejects active drives sharing the readout generator. Program build rejects
    overlapping active drive pulses on the same generator. Inactive tones retain
    drafts and are not subject to these active generator checks.
    expts is the positive hardware point count shared by all participants.
    Sweeps advance together by point, not as a Cartesian product. reps/rounds are positive averaging counts;
    initial_delay/relax_delay are finite nonnegative waits in us.
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
    tones: Annotated[list[MuxFreqToneCfg], FieldMeta(element_label="name")]
    expts: Annotated[int, FieldMeta(label="Expts", unit="count")] = Field(
        strict=True,
        gt=0,
        title="Expts",
        description="Hardware scan point count shared by all tones.",
    )
    participants: Annotated[
        list[StrictInt],
        FieldMeta(
            description="Enter the original zero-based tone indices in measurement order."
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
    def validate_scan(self) -> MuxFreqCfg:
        """Validate full tone identities, participant drives and shared point count.

        Reject mismatched full readout dimensions, invalid names or participants,
        and active drives sharing the readout generator. expts must be positive.

        """
        validate_mux_scan(self, self.tones, self.participants, self.expts)
        return self


def prepare_frequency_scan(cfg: MuxFreqCfg, *, soccfg: QickConfig) -> MuxFreqScan:
    """Prepare original indices and MHz grids; propagate board conversion errors.

    cfg must already be validated and is deep-copied before use.
    soccfg supplies the board frequency grids.
    Return MuxFreqScan with original indices and float64 (K,P) MHz coordinates
    in participant order. Coordinates use QICK's hardware start/span/step rounding.
    Sweeps advance together by point index. No per-point pulses are prepared.
    Frequency conversion errors propagate.
    This function does not acquire data or change cfg.
    """
    cfg = cfg.model_copy(deep=True)
    indices = list(cfg.participants)
    coordinates = []
    for index in indices:
        tone = cfg.tones[index]
        drive = tone.require_drive()
        actual = sweep2array(
            make_sweep(tone.require_freq().start, tone.require_freq().stop, cfg.expts),
            "freq",
            {"soccfg": soccfg, "gen_ch": drive.ch, "mixer_freq": drive.mixer_freq},
        )
        coordinates.append(actual)
    return MuxFreqScan(indices, np.asarray(coordinates, dtype=np.float64))


class MuxFreqExp(Experiment[MuxSweepResult, MuxFreqCfg]):
    """Measure qubit spectra with simultaneous drives and a shared mux readout.

    Each point optionally resets, starts selected drives together and reads out after
    the longest drive finishes. Analysis estimates each qubit's transition
    frequency and linewidth. Saved results retain the full settings, original
    tone positions, actual MHz axes and complex IQ.
    """

    AXES_SPEC: ClassVar[GroupedAxesSpec[MuxSweepResult, MuxFreqCfg]] = mux_sweep_spec(
        MuxFreqCfg,
        tag="mux/twotone_spectrum",
        coordinate_label="Drive frequency",
        unit="Hz",
        scale=1e6,
    )

    Options: ClassVar[type[FreqAnalyzeOptions]] = FreqAnalyzeOptions

    def run(
        self,
        config: MuxFreqCfg,
        *,
        context: RunContext,
        on_acquire_stop: Callable[[], None] | None = None,
    ) -> MuxSweepResult:
        """Sweep selected drive frequencies together, then measure mux IQ.

        config supplies independent MHz sweeps and shared averaging settings;
        it is copied before use. context supplies the board, plots and stop signal.
        Optional reset precedes each point. Drives start together, without end-alignment
        padding, and readout follows the longest drive. Return actual MHz and
        average complex IQ arrays of shape (participants, points), with original
        tone indices. Live plots show one spectrum per tone in measurement_i,
        titled with its original index and name.
        Drive frequencies share one hardware loop; readout frequencies stay fixed.
        Cancellation keeps completed-round averages across the sweep, or NaN
        when no round completed.
        Acquisition and program errors propagate. on_acquire_stop optionally
        observes external cancellation once, after acquire scope and progress
        cleanup, before returning its signal view. Pure data stop does not call
        it. Observer exceptions propagate without retry or outcome changes.

        """
        # 1. Prepare zipped frequencies and readout channels in participant order.
        cfg = config.model_copy(deep=True)
        scan = prepare_frequency_scan(cfg, soccfg=context.soccfg)
        readout = cfg.readout.with_updates(
            tones=[cfg.readout.tones[index] for index in scan.indices]
        )

        # 2. Keep each spectrum on its own hardware-rounded frequency axis.
        viewers = []
        for index in scan.indices:
            fig, ax = context.plots.subplots(num=f"measurement_{index}")
            fig.suptitle(
                f"Mux two-tone spectrum · {cfg.tones[index].name} · IQ contrast"
            )
            ax.set(xlabel=_SWEEP_LABEL, ylabel=_IQ_LABEL)
            progress = ax.set_title(f"Acquired frequency points: 0 / {cfg.expts}")
            viewer = live_line(ax)
            ax.lines[0].set(marker="o", markersize=3, label="Measured IQ")
            ax.legend()
            viewers.append((viewer, progress))
            fig.show()

        @context.plots.on_plot_thread
        def update(data: NDArray[np.complex128]):
            artists = []
            for row, (viewer, progress) in enumerate(viewers):
                measured = data[:, row]
                acquired = np.isfinite(measured)
                real = (
                    qubfreq_signal2real(np.where(acquired, measured, np.nan))
                    if acquired.any()
                    else measured.real
                )
                progress.set_text(
                    f"Acquired frequency points: {np.count_nonzero(acquired)} / {cfg.expts}"
                )
                artists.extend((*viewer.set(scan.coordinates[row], real), progress))
            return artists

        # 3. A shared DAC register loop advances the independent frequency ranges.
        signals = SignalBuffer(scan.coordinates.T.shape, on_update=update)
        with Schedule(signals, scope=context.scope) as sched:
            drives = []
            for index in scan.indices:
                tone = cfg.tones[index]
                freq = sweep2param(
                    "point",
                    make_sweep(
                        tone.require_freq().start, tone.require_freq().stop, cfg.expts
                    ),
                )
                drives.append(Pulse(f"drive_{index}", tone.require_drive(), freq=freq))

            # Reset -> simultaneous drives -> fixed mux readout after the longest drive.
            builder = TimelineBuilder(
                context.soccfg,
                reps=cfg.reps,
                initial_delay=cfg.initial_delay,
                relax_delay=cfg.relax_delay,
            )
            builder.add_loop("point", scan.coordinates.shape[1])
            if cfg.reset is not None:
                builder.emit(Reset("reset", cfg.reset))
            timelines = [f"drive_{index}" for index in scan.indices]
            builder.fork("default", into=timelines)
            for drive, timeline in zip(drives, timelines, strict=True):
                builder.emit(drive, timeline=timeline)
            builder.join(*timelines, into="default")
            readout_module = MuxReadout("readout", readout)
            builder.emit(readout_module)

            # 4. Each completed round supplies the full point/tone IQ matrix.
            prog = builder.build()
            _ = sched.run_acquire(
                prog,
                context.soc,
                options=RunOptions(
                    converter=partial(averaged_mux_signal, prog, readout_module),
                    on_stop=on_acquire_stop,
                ),
                rounds=cfg.rounds,
            )

        # Publish the final spectra and restore tone-major result storage.
        signals.trigger_update(flush=True)
        return MuxSweepResult(
            np.asarray(scan.indices, dtype=np.int64),
            scan.coordinates,
            signals.array.T.copy(),
        )

    def analyze(
        self,
        source: RunRecord[MuxFreqCfg, MuxSweepResult],
        options: FreqAnalyzeOptions,
        *,
        plots: Plots,
    ) -> tuple[MuxFreqFit | MuxFitFailure, ...]:
        """Fit each acquired spectrum for transition frequency and linewidth.

        source contains actual MHz axes and IQ; its cfg may be absent. options
        selects the line shape. plots receives a fit_i figure with a fitted
        overlay per original tone index. Return estimates or a per-tone failure reason.

        """
        # 1. Retain acquired IQ on each tone's actual frequency grid for independent line-shape fits.
        result = source.result
        validate_mux_result(result)

        # Each qubit keeps its own frequency grid and can fail independently.
        outcomes: list[MuxFreqFit | MuxFitFailure] = []
        for row, raw_index in enumerate(result.tone_indices):
            index = int(raw_index)
            finite = np.isfinite(result.signals[row])
            xs = result.coordinates[row, finite]
            signal = result.signals[row, finite]
            cfg = source.cfg
            tone = f"Tone {index}" if cfg is None else cfg.tones[index].name
            fig, ax = plots.subplots(num=f"fit_{index}", figsize=(7, 5))
            ax.set(xlabel=_SWEEP_LABEL, ylabel=_IQ_LABEL)
            fig.suptitle(f"Mux two-tone spectrum · {tone} · IQ contrast")

            # Fit only acquired points; six points are needed for the five-parameter model.
            try:
                if len(xs) < 6:
                    raise ValueError(
                        "Frequency fit requires at least six acquired points"
                    )
                real = qubfreq_signal2real(signal, model_type=options.model_type)
                freq, ferr, width, werr, fitted, (params, covariance) = fit_qubit_freq(
                    xs, real, options.model_type
                )
                if not np.all(np.isfinite((freq, ferr, width, werr))):
                    raise ValueError("Frequency fit returned nonfinite estimates")
                names = ("offset", "slope", "amplitude", "frequency", "width")
                quality = compute_fit_quality(
                    real,
                    fitted,
                    {
                        name: float(value)
                        for name, value in zip(names, params, strict=True)
                    },
                    covariance,
                )
                outcome = MuxFreqFit(index, freq, ferr, width, werr, quality)
            except (
                AnalysisFailed,
                ValueError,
                RuntimeError,
                ArithmeticError,
                np.linalg.LinAlgError,
            ) as exc:
                ax.plot(
                    xs,
                    qubfreq_signal2real(signal, model_type=options.model_type)
                    if signal.size
                    else signal.real,
                    "o",
                    ms=3,
                    label="Measured IQ",
                )
                ax.text(
                    0.02,
                    0.98,
                    f"Fit unavailable: {exc}",
                    transform=ax.transAxes,
                    va="top",
                    wrap=True,
                )
                outcomes.append(MuxFitFailure(index, str(exc)))
            else:
                ax.plot(xs, real, "o", ms=3, label="Measured IQ")
                model = lorfunc if options.model_type == "lor" else sincfunc
                label = "Lorentzian fit" if options.model_type == "lor" else "Sinc fit"
                ax.axvline(freq, color="0.4", ls=":", label="Transition frequency")
                result_label = f"Transition frequency = {freq:.7g} ± {ferr:.2g} MHz · FWHM = {width:.4g} ± {werr:.2g} MHz"
                curve = np.linspace(xs.min(), xs.max(), 400)
                ax.plot(curve, model(curve, *params), color="C1", label=label)
                r2 = quality.r2
                rms = quality.normalized_residual_rms
                quality_label = f"R² = {r2:.3g}" if r2 is not None else "R² unavailable"
                quality_label += (
                    f" · Normalized residual RMS = {rms:.3g}"
                    if rms is not None
                    else " · Normalized residual RMS unavailable"
                )
                ax.set_title(f"{result_label}\n{quality_label}", fontsize=10)
                outcomes.append(outcome)
            if options.model_type == "sinc":
                ax.set_ylabel("Principal-axis background-subtracted IQ (a.u.)")
            ax.legend()
            fig.tight_layout()
        # 2. Keep transition frequency, linewidth and quality, or a failure under the original tone identity.
        return tuple(outcomes)
