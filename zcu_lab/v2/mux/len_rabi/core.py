"""Mux length Rabi with simultaneous drives and per-qubit length sweeps.

Technical reference for lab authors and framework developers.
MuxLenRabiExp.run(config, *, context) sweeps selected drive lengths together.

Configuration and timing
------------------------
The full tones list matches the complete MuxReadoutCfg in count and order.
Each MuxLenRabiToneCfg has a fixed name, a required drive and length bounds. Participants is
a nonempty, unique list of original indices. Q3/Q1 selects original indices [2, 0]
from Q1/Q2/Q3 and retains that execution/data order. Every drive must be valid,
only for participants, on independent generators distinct from readout.
The caller supplies the complete qubit/resonator bindings; numeric cfg validation
does not resolve component identities.

Sweep ranges may differ; cfg.expts supplies one positive point count shared by
selected tones. Point p pairs each active tone's p-th value,
not a Cartesian product. All const/flat_top drives use one hardware point loop.
If any participant uses a sampled waveform, the host scans all participants,
compiles once per point and reuses each program across rounds. Each program runs
one round. Both paths apply optional common reset, simultaneous drives and mux readout.
Branches start together; short branches wait for the longest branch at that point. Pre/post delays stay
unchanged. There is no end alignment, end padding or sweep-wide maximum padding.
All participants share reps, rounds, initial_delay and relax_delay.
Program modules perform native conversion and averaging.

Results and cancellation
------------------------
MuxRabiResult.tone_indices is int64 (K,) in participant order. Coordinates are
float64 (K,P) on each drive's own rounded grid. Signals are complex128 (K,P),
corrected average IQ in a.u. Live and fit figures show projected IQ and
same-direction shot SEM for each original tone. Hardware acquisition updates
the full sweep after each round. Host acquisition combines all shots in completed
rounds for each point. Cancellation leaves unmeasured points NaN.

Length scan contract
--------------------
MuxLenRabiCfg.reset is optional; MuxLenRabiCfg.readout is the full mux readout.
Sweep bounds are positive requested lengths in us. Participants support const,
flat_top and sampled waveforms such as cosine, gauss and DRAG.
Const length is the entire pulse.
Flat_top length includes raise_waveform ramps; the waveform implementation
subtracts ramps to determine the plateau. Gain, frequency, phase and ramp
settings stay fixed. Arb assets have fixed length and are rejected at cfg
construction rather than stretched or cropped.

prepare_length_scan(cfg, soccfg=...) copies cfg and resolves original indices
in participant order. sweep2array(..., "time", gen_ch=...) creates each drive's
rounded requested-length grid for hardware tables or host points. Each grid retains its
hardware spacing and full point count, with no sorting or deduplication.
Nonpositive rounded lengths or overlapping adjacent points raise. Coordinates
are requested total lengths rather than compiled traces; flat_top ramp handling
and native rounding can change playback by a few ticks. Waveform/compiler
constraints, including ramps and minimum cycles, propagate as errors.
Nonzero pre_delay remains part of the pulse's native branch.

Analysis and persistence
------------------------
Analysis accepts an explicit RunRecord, including cfg=None. It returns
MuxLenRabiFit or MuxFitFailure per original index and draws fit_i. Each curve
uses its own length coordinates and needs six finite IQ points. Empty or
insufficient data fails per tone; malformed source shapes/dtypes raise.

Options reuse single-qubit Len Rabi. decay=True selects the default damped
cosine; it can be disabled. fit_phase=False fixes phase at zero with signed
amplitude; True fits free phase. The fitter computes pi/pi2 lengths and
covariance error propagation. Length and one-sigma errors are in us; Rabi
frequency and its error are in MHz. FitQuality uses finite observations,
model and covariance. Small-truncation or poor-quality curves can fit numerically
while remaining unsuitable for calibration.

Native persistence retains the full cfg, original indices, separate length axes
and complex IQ. Length coordinates use seconds on disk and us in memory.
The source drive/ramp stays in cfg for later proposals. Persistence does not
embed waveform asset bytes or fetch replacement pulses from a current library.

Calling and ownership
---------------------
For context, plots, stop/error handling and record/persistence ownership, see
lib/zcu_tools/experiment/README.md, section "實驗介面與資料", and
lib/zcu_tools/experiment/v2/runtime/README.md, section "Stop 與 Retry".

Analysis uses the explicit historical source, not the current Form or library.
The numerical core leaves component bindings and writeback to its caller.

Example with validated cfg, RunContext and fresh analysis_plots::

    from zcu_tools.experiment.records import RunRecord
    from zcu_lab.v2.mux.len_rabi import MuxLenRabiExp

    core = MuxLenRabiExp()
    result = core.run(cfg, context=context)
    source = RunRecord(cfg=cfg, result=result)
    outcomes = core.analyze(source, core.Options(), plots=analysis_plots)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray
from pydantic import Field, StrictInt, model_validator
from pydantic.dataclasses import dataclass as validated_dataclass
from qick import QickConfig
from zcu_tools.analysis.failure import AnalysisFailed
from zcu_tools.analysis.fitting import (
    FitQuality,
    compute_fit_quality,
    cosfunc,
    decaycos,
    fit_rabi,
)
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import Experiment
from zcu_tools.experiment.axes_spec import GroupedAxesSpec
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.utils.sweep import make_sweep
from zcu_tools.experiment.v2.runtime import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.experiment.v2.utils.snr import estimate_snr
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.acquisition import ShotReducer
from zcu_tools.program.v2 import (
    Reset,
    ResetCfg,
    TableParam,
    TimelineBuilder,
    TimelineProgramV2,
    mux_iq_sums,
    split_sums,
)
from zcu_tools.program.v2.modules import (
    MuxReadout,
    MuxReadoutCfg,
    Pulse,
)
from zcu_tools.utils.process import rotate2real

from zcu_lab.v2.mux._moments import project_iq, validate_iq_moments
from zcu_lab.v2.support import live_errorbar
from zcu_lab.v2.support.mux import (
    MuxFitFailure,
    MuxRabiResult,
    validate_mux_result,
)
from zcu_lab.v2.support.mux_persistence import mux_rabi_spec
from zcu_lab.v2.support.mux_sweep_cfg import (
    MuxDriveToneCfg,
    ToneRangeCfg,
    validate_mux_scan,
)

_SWEEP_LABEL = "Drive pulse length (µs)"
_IQ_LABEL = "Principal-axis mean IQ (a.u.)"


@validated_dataclass(frozen=True)
class LenRabiAnalyzeOptions:
    """Select a decay envelope and a free phase for each pulse-length fit."""

    decay: Annotated[bool, FieldMeta(label="Fit decay envelope")] = True
    fit_phase: Annotated[bool, FieldMeta(label="Fit phase offset")] = False


@dataclass(frozen=True)
class MuxLenRabiFit:
    """Fit for original tone index, with lengths in us and frequency in MHz.

    pi_len/pi2_len are fitted pi/pi2 total pulse lengths. Their *_err fields and
    rabi_f_err are one-sigma uncertainties from the owner's full covariance.
    rabi_f is the oscillation frequency in inverse us. quality describes the
    chosen cosine model and the finite acquired samples actually fitted.
    """

    index: int
    pi_len: float
    pi_len_err: float
    pi2_len: float
    pi2_len_err: float
    rabi_f: float
    rabi_f_err: float
    quality: FitQuality


@dataclass(frozen=True)
class MuxLengthScan:
    """Prepared length axes without acquisition state or per-point pulse copies.

    indices retains original tone indices in participant order. coordinates is
    float64 (K,P), containing each drive's rounded total lengths in us.
    """

    indices: Sequence[int]
    coordinates: NDArray[np.float64]


class MuxLenRabiToneCfg(MuxDriveToneCfg):
    """One named drive and its independent length bounds in us."""

    length: Annotated[
        ToneRangeCfg | None,
        FieldMeta(
            label="Length",
            unit="µs",
            description="Enter this tone's total drive pulse duration range in µs.",
        ),
    ] = None

    def require_length(self) -> ToneRangeCfg:
        """Return active length bounds, or raise ValueError naming this tone."""
        if self.length is None:
            raise ValueError(f"{self.name}: active tone requires length")
        return self.length


class MuxLenRabiCfg(ConfigBase):
    """Complete named tones, ordered participants and shared acquisition controls.

    reset and readout hold the common modules. tones contains one
    MuxLenRabiToneCfg per original readout position, with unique nonempty names.
    participants is a nonempty, duplicate-free list of original tone indices; its order
    determines execution/data rows (Q3/Q1 maps to original indices 2/0).
    Selected drives are required/validated on independent
    generators distinct from readout. Independent positive us length ranges
    use expts, the positive point count shared by all tones. Const/flat_top use
    hardware length tables; any sampled waveform selects the host path for all
    participants. Arb duration belongs to its asset and cannot be swept.
    earlystop_snr optionally stops the host scan after a full round when every
    participant reaches the threshold; None disables it. Hardware scans ignore it.
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
    tones: Annotated[list[MuxLenRabiToneCfg], FieldMeta(element_label="name")]
    expts: Annotated[int, FieldMeta(label="Expts", unit="count")] = Field(
        strict=True,
        gt=0,
        title="Expts",
        description="Scan point count shared by all tones.",
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
    earlystop_snr: Annotated[
        float | None,
        FieldMeta(
            label="Early-stop SNR",
            description="Host scan stops after a full round when every tone reaches this SNR; None disables.",
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
    def validate_scan(self) -> MuxLenRabiCfg:
        """Reject invalid full tones, active generators, counts or length policy."""
        validate_mux_scan(self, self.tones, self.participants, self.expts)
        for index in self.participants:
            tone = self.tones[index]
            drive = tone.require_drive()
            if drive.waveform.style == "arb":
                raise ValueError(
                    f"tones[{index}].drive.waveform: arb length is asset-derived "
                    "with no public length sweep"
                )
        return self


def prepare_length_scan(cfg: MuxLenRabiCfg, *, soccfg: QickConfig) -> MuxLengthScan:
    """Prepare pulse-length grids; raise for invalid or collapsed board lengths.

    cfg supplies complete tones and ordered participants; soccfg supplies the
    board clocks. Return original indices and float64 lengths (K,P) in us,
    pairing the qubits' own sweep points rather than taking their Cartesian
    product. No per-point pulses are constructed. Each length follows its drive generator's
    time grid and includes flat-top ramps. Playback may differ by clock ticks.

    Inputs stay unchanged. Invalid cfg, nonpositive rounded lengths, collapsed
    sweeps and waveform conversion errors raise before acquisition.
    """
    cfg = cfg.model_copy(deep=True)
    indices = list(cfg.participants)
    coordinates = []
    for index in indices:
        tone = cfg.tones[index]
        drive = tone.require_drive()
        lengths = sweep2array(
            make_sweep(
                tone.require_length().start, tone.require_length().stop, cfg.expts
            ),
            "time",
            {"soccfg": soccfg, "gen_ch": drive.ch},
        )
        if np.any(lengths <= 0):
            raise ValueError(
                f"tones[{index}].length quantized lengths must be positive µs"
            )
        coordinates.append(lengths)
    return MuxLengthScan(indices, np.asarray(coordinates, dtype=np.float64))


def _plot_len_fit(
    ax: Axes,
    xs: NDArray[np.float64],
    params: tuple[float, ...],
    outcome: MuxLenRabiFit,
    *,
    decay: bool,
) -> None:
    pi, pierr = outcome.pi_len, outcome.pi_len_err
    pi2, pi2err = outcome.pi2_len, outcome.pi2_len_err
    freq, freqerr = outcome.rabi_f, outcome.rabi_f_err
    model = decaycos if decay else cosfunc
    label = "Damped cosine" if decay else "Cosine fit"
    for value, marker, color in ((pi, "π length", "C2"), (pi2, "π/2 length", "C3")):
        if xs.min() <= value <= xs.max():
            ax.axvline(value, ls=":", color=color, label=marker)
    result_label = f"π length = {pi:.4g} ± {pierr:.2g} µs · π/2 length = {pi2:.4g} ± {pi2err:.2g} µs"
    result_label += f"\nRabi rate = {freq:.4g} ± {freqerr:.2g} MHz"
    curve = np.linspace(xs.min(), xs.max(), 400)
    ax.plot(curve, model(curve, *params), color="C1", label=label)
    r2 = outcome.quality.r2
    rms = outcome.quality.normalized_residual_rms
    quality_label = f"R² = {r2:.3g}" if r2 is not None else "R² unavailable"
    quality_label += (
        f" · Normalized residual RMS = {rms:.3g}"
        if rms is not None
        else " · Normalized residual RMS unavailable"
    )
    ax.set_title(f"{result_label}\n{quality_label}", fontsize=10)


class MuxLenRabiExp(Experiment[MuxRabiResult, MuxLenRabiCfg]):
    """Measure Rabi oscillations by sweeping simultaneous qubit pulse lengths.

    Each point resets, starts the selected drives together, waits for the longest
    drive, then reads their resonators together. Results retain each qubit's
    length axis in us and average complex IQ; fits estimate pi and pi/2 lengths.
    """

    AXES_SPEC: ClassVar[GroupedAxesSpec[MuxRabiResult, MuxLenRabiCfg]] = mux_rabi_spec(
        MuxLenRabiCfg,
        tag="mux/rabi/len",
        coordinate_label="Pulse length",
        unit="s",
        scale=1e-6,
    )

    Options: ClassVar[type[LenRabiAnalyzeOptions]] = LenRabiAnalyzeOptions

    def run(self, config: MuxLenRabiCfg, *, context: RunContext) -> MuxRabiResult:
        """Reset, sweep simultaneous drive lengths, and acquire mux readout IQ.

        config supplies full tones, ordered participants and shared averaging
        and delays. Each point pairs the participants' own quantized lengths;
        The fork starts them together and readout waits for the last drive, without
        end alignment or sweep-wide padding. Return original tone indices (K,),
        requested total lengths (K,P) in us and average complex IQ (K,P) in
        ADC-length-normalized a.u. Flat-top lengths include their ramps.

        All const/flat_top participants use hardware tables. Any sampled drive
        selects a host scan for all participants, with one compiled program per
        point reused across rounds. Host programs run rounds=1; the host adds
        shot moments from available rounds per point and checks earlystop_snr after each full scan.
        Native arb assets raise ValueError before coordinates or compilation.

        context supplies soc/soccfg, plots and a fresh stop signal. Cancellation
        retains acquired values and leaves unmeasured points NaN.
        Acquisition and cfg/build errors propagate. The caller owns plot and
        hardware cleanup and decides whether returned data can be analyzed.

        """
        # Prepare each qubit's length grid and select its original readout tone.
        # 1. Use hardware length tables only when every selected drive is constant or flat-top.
        cfg = config.model_copy(deep=True)
        scan = prepare_length_scan(cfg, soccfg=context.soccfg)
        if all(
            cfg.tones[index].require_drive().waveform.style in ("const", "flat_top")
            for index in scan.indices
        ):
            return self._run_for_flat(cfg, scan, context=context)
        # 2. Use cached host-point programs for sampled envelopes on any participant.
        return self._run_for_arb(cfg, scan, context=context)

    def _run_for_flat(
        self, cfg: MuxLenRabiCfg, scan: MuxLengthScan, *, context: RunContext
    ) -> MuxRabiResult:
        """Sweep all const/flat-top length tables in one hardware program."""
        readout = cfg.readout.with_updates(
            tones=[cfg.readout.tones[index] for index in scan.indices]
        )
        viewers = []
        for index in scan.indices:
            fig, ax = context.plots.subplots(num=f"measurement_{index}")
            fig.suptitle(f"Mux length Rabi · {cfg.tones[index].name} · IQ contrast")
            ax.set(xlabel=_SWEEP_LABEL, ylabel=_IQ_LABEL)
            progress = ax.set_title(f"Acquired length points: 0 / {cfg.expts}")
            viewer = live_errorbar(ax)
            ax.lines[0].set(marker="o", markersize=3, label="Measured IQ ± shot SEM")
            ax.legend()
            viewers.append((viewer, progress))
            fig.show()

        @context.plots.on_plot_thread
        def update(data: NDArray[np.float64]):
            _, mean = split_sums(data)
            artists = []
            for row, (viewer, progress) in enumerate(viewers):
                measured = mean[:, row]
                acquired = np.isfinite(measured)
                real, sem = project_iq(
                    np.where(acquired, measured, np.nan), data[:, row]
                )
                progress.set_text(
                    f"Acquired length points: {np.count_nonzero(acquired)} / {cfg.expts}"
                )
                artists.extend(
                    (*viewer.set(scan.coordinates[row], real, sem), progress)
                )
            return artists

        # 1. One hardware point loop selects all participants' length tables.
        moments = SignalBuffer(
            (*scan.coordinates.T.shape, 6), dtype=np.float64, on_update=update
        )
        with Schedule(moments, scope=context.scope) as sched:
            drives = [
                Pulse(
                    f"drive_{index}",
                    cfg.tones[index].require_drive(),
                    length=TableParam("point", scan.coordinates[row].tolist()),
                )
                for row, index in enumerate(scan.indices)
            ]
            # Reset -> simultaneous drives -> readout after this point's longest drive.
            builder = TimelineBuilder(
                context.soccfg,
                reps=cfg.reps,
                initial_delay=cfg.initial_delay,
                relax_delay=cfg.relax_delay,
            )
            builder.add_loop("point", cfg.expts)
            if cfg.reset is not None:
                builder.emit(Reset("reset", cfg.reset))
            timelines = [f"drive_{index}" for index in scan.indices]
            builder.fork("default", into=timelines)
            for drive, timeline in zip(drives, timelines, strict=True):
                builder.emit(drive, timeline=timeline)
            builder.join(*timelines, into="default")
            # Consume this point's common end before readout to release runtime registers.
            builder.delay_by_timeline()
            builder.emit(MuxReadout("readout", readout))
            prog = builder.build()

            # 2. Publish the full point/tone matrix after each completed round.
            _ = sched.run_acquire_summed(
                prog,
                context.soc,
                rounds=cfg.rounds,
                reducer=mux_iq_sums(prog, "readout"),
            )
        moments.trigger_update(flush=True)
        tone_sums = np.swapaxes(moments.array, 0, 1).copy()
        _, mean = split_sums(tone_sums)
        return MuxRabiResult(
            np.asarray(scan.indices, dtype=np.int64), scan.coordinates, mean, tone_sums
        )

    def _run_for_arb(
        self, cfg: MuxLenRabiCfg, scan: MuxLengthScan, *, context: RunContext
    ) -> MuxRabiResult:
        """Scan sampled waveforms on the host; native arb assets stay unsupported."""
        # 1. Show each participant's length response and average only measured round/point pairs.
        rounds = cfg.rounds
        cfg = cfg.model_copy(deep=True)
        cfg.rounds = 1
        readout = cfg.readout.with_updates(
            tones=[cfg.readout.tones[index] for index in scan.indices]
        )
        programs: dict[
            int, tuple[TimelineProgramV2, ShotReducer[NDArray[np.float64]]]
        ] = {}
        viewers = []
        for index in scan.indices:
            fig, ax = context.plots.subplots(num=f"measurement_{index}")
            fig.suptitle(f"Mux length Rabi · {cfg.tones[index].name} · IQ contrast")
            ax.set(xlabel=_SWEEP_LABEL, ylabel=_IQ_LABEL)
            progress = ax.set_title(f"Acquired length points: 0 / {cfg.expts}")
            viewer = live_errorbar(ax)
            ax.lines[0].set(marker="o", markersize=3, label="Measured IQ ± shot SEM")
            ax.legend()
            viewers.append((viewer, progress))
            fig.show()

        def sum_host_rounds(data: NDArray[np.float64]) -> NDArray[np.float64]:
            # Ignore unfinished round/point slots, without turning untouched points into zeros.
            sums = np.nansum(data, axis=0)
            sums[np.isnan(data[..., 0]).all(axis=0)] = np.nan
            return sums

        @context.plots.on_plot_thread
        def update(data: NDArray[np.float64]):
            sums = sum_host_rounds(data)
            _, mean = split_sums(sums)
            artists = []
            for row, (viewer, progress) in enumerate(viewers):
                measured = mean[:, row]
                acquired = np.isfinite(measured)
                real, sem = project_iq(
                    np.where(acquired, measured, np.nan), sums[:, row]
                )
                progress.set_text(
                    f"Acquired length points: {np.count_nonzero(acquired)} / {cfg.expts}"
                )
                artists.extend(
                    (*viewer.set(scan.coordinates[row], real, sem), progress)
                )
            return artists

        # 2. Reuse each point's reset/simultaneous-drive/readout program across host rounds.
        moments = SignalBuffer(
            (rounds, *scan.coordinates.T.shape, 6), dtype=np.float64, on_update=update
        )
        with Schedule(moments, scope=context.scope) as sched:
            for _, rep in sched.repeat("round", rounds):
                for point, step in rep.scan("point", list(range(cfg.expts))):
                    if point not in programs:
                        builder = TimelineBuilder(
                            context.soccfg,
                            reps=cfg.reps,
                            initial_delay=cfg.initial_delay,
                            relax_delay=cfg.relax_delay,
                        )
                        if cfg.reset is not None:
                            builder.emit(Reset("reset", cfg.reset))
                        timelines = [f"drive_{index}" for index in scan.indices]
                        builder.fork("default", into=timelines)
                        for row, (index, timeline) in enumerate(
                            zip(scan.indices, timelines, strict=True)
                        ):
                            builder.emit(
                                Pulse(
                                    f"drive_{index}",
                                    cfg.tones[index].require_drive(),
                                    length=float(scan.coordinates[row, point]),
                                ),
                                timeline=timeline,
                            )
                        builder.join(*timelines, into="default")
                        builder.emit(MuxReadout("readout", readout))
                        prog = builder.build()
                        programs[point] = prog, mux_iq_sums(prog, "readout")
                    prog, reducer = programs[point]
                    _ = step.run_acquire_summed(
                        prog, context.soc, reducer=reducer, rounds=1
                    )
                # Check every tone only after a complete host-side point scan.
                _, mean = split_sums(sum_host_rounds(moments.array))
                if (
                    cfg.earlystop_snr is not None
                    and np.all(np.isfinite(mean))
                    and all(
                        estimate_snr(rotate2real(trace).real) >= cfg.earlystop_snr
                        for trace in mean.T
                    )
                ):
                    break
        moments.trigger_update(flush=True)
        # 3. Keep available rounds on the planned grids; untouched points remain NaN.
        tone_sums = np.swapaxes(sum_host_rounds(moments.array), 0, 1).copy()
        _, mean = split_sums(tone_sums)
        return MuxRabiResult(
            np.asarray(scan.indices, dtype=np.int64), scan.coordinates, mean, tone_sums
        )

    def analyze(
        self,
        source: RunRecord[MuxLenRabiCfg, MuxRabiResult],
        options: LenRabiAnalyzeOptions,
        *,
        plots: Plots,
    ) -> tuple[MuxLenRabiFit | MuxFitFailure, ...]:
        """Fit Rabi oscillations from source, with or without its saved cfg.

        options.decay selects a damped cosine; fit_phase frees its phase instead
        of fixing it to zero. Return each original tone's pi/pi2 lengths and
        errors in us, Rabi frequency and error in MHz, and fit quality, or a
        numerical failure with its reason. Fresh caller-owned plots receive
        fit_i figures. Invalid source data raises; no live cfg or hardware is read.

        """
        # 1. Fit acquired length points per tone, with optional damping and a fixed or free phase.
        result = source.result
        validate_mux_result(result)
        moments = result.iq_moments
        validate_iq_moments(moments, result.signals)
        outcomes: list[MuxLenRabiFit | MuxFitFailure] = []
        # 2. Drop missing IQ per tone without substituting points from another length grid.
        for row, raw_index in enumerate(result.tone_indices):
            index = int(raw_index)
            signal = result.signals[row]
            finite = np.isfinite(signal)
            xs = result.coordinates[row, finite]
            signal = signal[finite]
            cfg = source.cfg
            tone = f"Tone {index}" if cfg is None else cfg.tones[index].name
            fig, ax = plots.subplots(
                num=f"fit_{index}", figsize=(7, 5), layout="constrained"
            )
            ax.set(xlabel=_SWEEP_LABEL, ylabel=_IQ_LABEL)
            fig.suptitle(f"Mux length Rabi · {tone} · IQ contrast")
            real, sem = project_iq(
                signal, None if moments is None else moments[row, finite]
            )
            ax.plot(xs, real, "o", ms=3, label="Measured IQ")
            if sem is not None and np.any(np.isfinite(sem)):
                ax.errorbar(
                    xs, real, yerr=sem, fmt="none", ecolor="C0", label="Shot SEM"
                )
            try:
                if len(xs) < 6:
                    raise ValueError(
                        "Len Rabi fit requires at least six acquired points"
                    )
                # 3. Project IQ before fitting the selected damped/undamped cosine and phase policy.
                fit = fit_rabi(
                    xs,
                    real,
                    decay=options.decay,
                    init_phase=None if options.fit_phase else 0.0,
                )
                pi, pierr = fit.pi_x, fit.pi_x_err
                pi2, pi2err = fit.pi2_x, fit.pi2_x_err
                freq, freqerr = fit.freq, fit.freq_err
                fitted, params, covariance = fit.curve, fit.params, fit.cov
                if not np.all(np.isfinite((pi, pierr, pi2, pi2err, freq, freqerr))):
                    raise ValueError("Len Rabi fit returned nonfinite estimates")
                names = ("y0", "yscale", "freq", "phase") + (
                    ("decay_time",) if options.decay else ()
                )
                quality = compute_fit_quality(
                    real,
                    fitted,
                    {
                        name: float(value)
                        for name, value in zip(names, params, strict=True)
                    },
                    covariance,
                )
                outcome = MuxLenRabiFit(
                    index, pi, pierr, pi2, pi2err, freq, freqerr, quality
                )
            except (
                AnalysisFailed,
                ValueError,
                RuntimeError,
                ArithmeticError,
                np.linalg.LinAlgError,
            ) as exc:
                # 4. Keep the failure reason without hiding acquired IQ or stopping other fits.
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
                _plot_len_fit(ax, xs, params, outcome, decay=options.decay)
                outcomes.append(outcome)
            fig.legend(loc="outside lower center", ncols=3)
        # 6. Preserve each tone's pi/pi2 lengths, frequency and quality, or its failure reason.
        return tuple(outcomes)
