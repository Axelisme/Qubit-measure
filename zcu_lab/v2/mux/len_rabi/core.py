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
MuxSweepResult.tone_indices is int64 (K,) in participant order. Coordinates are
float64 (K,P) on each drive's own rounded grid. Signals are complex128 (K,P),
ADC-length-normalized average IQ in a.u. The measurement_i magnitude plots use
original tone indices and each tone's own coordinates, so duplicate labels retain
distinct plots. Hardware acquisition updates the full sweep after each round.
Host acquisition updates each measured point and averages its available rounds.
Cancellation keeps acquired values and leaves unmeasured points NaN.

Length scan contract
--------------------
MuxLenRabiCfg.reset is optional; MuxLenRabiCfg.readout is the full mux readout.
Sweep bounds are positive requested lengths in us. Participants support const,
flat_top and sampled waveforms such as cosine, gauss and DRAG.
Const length is the entire pulse.
Flat_top length includes raise_waveform ramps; the waveform implementation
subtracts ramps to determine the plateau. Gain, frequency, phase and ramp
settings stay fixed. Arb assets have fixed length and are rejected by cfg and
run validation rather than stretched or cropped.

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
from functools import partial
from typing import Annotated, ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, StrictInt, model_validator
from qick import QickConfig
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import Experiment
from zcu_tools.experiment.axes_spec import GroupedAxesSpec
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.utils.sweep import make_sweep
from zcu_tools.experiment.v2.runtime import RunOptions, Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.experiment.v2.utils.snr import estimate_snr
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import Reset, ResetCfg, TimelineBuilder, TimelineProgramV2
from zcu_tools.program.v2.modules import (
    MuxReadout,
    MuxReadoutCfg,
    Pulse,
    TableLengthPulse,
)
from zcu_tools.utils.process import rotate2real

from zcu_lab.v2.support import live_line
from zcu_lab.v2.support.mux import MuxFitFailure, MuxSweepResult, averaged_mux_signal
from zcu_lab.v2.support.mux_persistence import mux_sweep_spec
from zcu_lab.v2.support.mux_sweep_cfg import (
    MuxDriveToneCfg,
    ToneRangeCfg,
    require_tone_value,
    validate_mux_scan,
)
from zcu_lab.v2.twotone.rabi.len_rabi.core import LenRabiAnalyzeOptions

from .analysis import MuxLenRabiFit, analyze_length


class MuxLenRabiToneCfg(MuxDriveToneCfg):
    """One named drive and its independent length bounds in us."""

    length: Annotated[ToneRangeCfg | None, FieldMeta(unit="us")] = None

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

    reset: ResetCfg | None = None
    readout: MuxReadoutCfg
    tones: Annotated[list[MuxLenRabiToneCfg], FieldMeta(element_label="name")]
    expts: int = Field(
        strict=True,
        gt=0,
        title="Expts",
        description="Scan point count shared by all tones.",
    )
    participants: list[StrictInt]

    reps: Annotated[
        int, FieldMeta(label="Reps", description="Number of repetitions per round.")
    ] = 1
    rounds: Annotated[
        int,
        FieldMeta(label="Rounds", description="Number of software acquisition rounds."),
    ] = 1
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
            unit="us",
            description="Delay before the first repetition.",
        ),
    ] = 1.0
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="us",
            decimals=3,
            description="Relaxation delay after each repetition.",
        ),
    ] = 1.0

    @model_validator(mode="after")
    def validate_scan(self) -> MuxLenRabiCfg:
        """Reject invalid full tones, active generators, counts or length policy."""
        validate_mux_scan(self, self.tones, self.participants, self.expts)
        for index in self.participants:
            tone = self.tones[index]
            require_tone_value(tone.length, ("tones", index, "length"))
            drive = tone.require_drive()
            if drive.waveform.style == "arb":
                raise ValueError(
                    f"tones[{index}].drive.waveform: arb length is asset-derived "
                    "with no public length sweep"
                )
            if min(tone.require_length().start, tone.require_length().stop) <= 0:
                raise ValueError(f"tones[{index}].length lengths must be positive us")
        return self


@dataclass(frozen=True)
class MuxLengthScan:
    """Prepared length axes without acquisition state or per-point pulse copies.

    indices retains original tone indices in participant order. coordinates is
    float64 (K,P), containing each drive's rounded total lengths in us.
    """

    indices: Sequence[int]
    coordinates: NDArray[np.float64]


def prepare_length_scan(cfg: MuxLenRabiCfg, *, soccfg: QickConfig) -> MuxLengthScan:
    """Prepare each selected qubit's pulse-length grid without acquiring data.

    cfg supplies complete tones and ordered participants; soccfg supplies the
    board clocks. Return original indices and float64 lengths (K,P) in us,
    pairing the qubits' own sweep points rather than taking their Cartesian
    product. No per-point pulses are constructed. Each length follows its drive generator's
    time grid and includes flat-top ramps. Playback may differ by clock ticks.

    Inputs stay unchanged. Invalid cfg, nonpositive rounded lengths, collapsed
    sweeps and waveform conversion errors raise before acquisition.
    """
    cfg = MuxLenRabiCfg.model_validate(cfg.model_dump())
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
                f"tones[{index}].length quantized lengths must be positive us"
            )
        coordinates.append(lengths)
    return MuxLengthScan(indices, np.asarray(coordinates, dtype=np.float64))


class MuxLenRabiExp(Experiment[MuxSweepResult, MuxLenRabiCfg]):
    """Measure Rabi oscillations by sweeping simultaneous qubit pulse lengths.

    Each point resets, starts the selected drives together, waits for the longest
    drive, then reads their resonators together. Results retain each qubit's
    length axis in us and average complex IQ; fits estimate pi and pi/2 lengths.
    """

    AXES_SPEC: ClassVar[GroupedAxesSpec[MuxSweepResult, MuxLenRabiCfg]] = (
        mux_sweep_spec(
            MuxLenRabiCfg,
            tag="mux/rabi/len",
            coordinate_label="Pulse length",
            unit="s",
            scale=1e-6,
        )
    )

    Options: ClassVar[type[LenRabiAnalyzeOptions]] = LenRabiAnalyzeOptions

    def run(self, config: MuxLenRabiCfg, *, context: RunContext) -> MuxSweepResult:
        """Reset, sweep simultaneous drive lengths, and acquire mux readout IQ.

        config supplies full tones, ordered participants and shared averaging
        and delays. Each point pairs the participants' own quantized lengths;
        The fork starts them together and readout waits for the last drive, without
        end alignment or sweep-wide padding. Return original tone indices (K,),
        requested total lengths (K,P) in us and average complex IQ (K,P) in
        ADC-length-normalized a.u. Flat-top lengths include their ramps.

        All const/flat_top participants use hardware tables. Any sampled drive
        selects a host scan for all participants, with one compiled program per
        point reused across rounds. Host programs run rounds=1; the host averages
        available rounds per point and checks earlystop_snr after each full scan.
        Native arb assets raise ValueError before coordinates or compilation.

        context supplies soc/soccfg, plots and a fresh stop signal. Cancellation
        retains acquired values and leaves unmeasured points NaN.
        Acquisition and cfg/build errors propagate. The caller owns plot and
        hardware cleanup and decides whether returned data can be analyzed.
        """
        # 1. Prepare each qubit's length grid and select its original readout tone.
        cfg = MuxLenRabiCfg.model_validate(config.model_dump())
        scan = prepare_length_scan(cfg, soccfg=context.soccfg)
        if all(
            cfg.tones[index].require_drive().waveform.style in ("const", "flat_top")
            for index in scan.indices
        ):
            return self._run_for_flat(cfg, scan, context=context)
        return self._run_for_arb(cfg, scan, context=context)

    def _run_for_flat(
        self, cfg: MuxLenRabiCfg, scan: MuxLengthScan, *, context: RunContext
    ) -> MuxSweepResult:
        """Sweep all const/flat-top length tables in one hardware program."""
        readout = cfg.readout.with_updates(
            tones=[cfg.readout.tones[index] for index in scan.indices]
        )
        viewers = []
        for index in scan.indices:
            fig, ax = context.plots.subplots(num=f"measurement_{index}")
            ax.set(
                xlabel="Pulse length (us)",
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
                for artist in viewer.set(scan.coordinates[row], np.abs(data[:, row]))
            )

        # 2. One hardware point loop selects all participants' length tables.
        signals = SignalBuffer(scan.coordinates.T.shape, on_update=update)
        with Schedule(signals, scope=context.scope) as sched:
            drives = [
                TableLengthPulse(
                    f"drive_{index}",
                    cfg.tones[index].require_drive(),
                    lengths=scan.coordinates[row],
                    idx_reg="point",
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
            readout_module = MuxReadout("readout", readout)
            builder.emit(readout_module)
            prog = builder.build()

            # 3. Publish the full point/tone matrix after each completed round.
            _ = sched.run_acquire(
                prog,
                context.soc,
                options=RunOptions(
                    converter=partial(averaged_mux_signal, prog, readout_module)
                ),
                rounds=cfg.rounds,
            )
        signals.trigger_update(flush=True)
        return MuxSweepResult(
            np.asarray(scan.indices, dtype=np.int64),
            scan.coordinates,
            signals.array.T.copy(),
        )

    def _run_for_arb(
        self, cfg: MuxLenRabiCfg, scan: MuxLengthScan, *, context: RunContext
    ) -> MuxSweepResult:
        """Scan sampled waveforms on the host; native arb assets stay unsupported."""
        rounds = cfg.rounds
        cfg = cfg.model_copy(deep=True)
        cfg.rounds = 1
        readout = cfg.readout.with_updates(
            tones=[cfg.readout.tones[index] for index in scan.indices]
        )
        programs: dict[int, TimelineProgramV2] = {}
        readouts: dict[int, MuxReadout] = {}
        viewers = []
        for index in scan.indices:
            fig, ax = context.plots.subplots(num=f"measurement_{index}")
            ax.set(
                xlabel="Pulse length (us)",
                ylabel="Magnitude (a.u.)",
                title=f"{index}: {cfg.tones[index].name}",
            )
            viewers.append(live_line(ax))
            fig.show()

        def average_round(data: NDArray[np.complex128]) -> NDArray[np.complex128]:
            # Average only measured entries; untouched point/tone pairs stay NaN.
            counts = np.sum(~np.isnan(data), axis=0)
            mean = np.full(data.shape[1:], np.nan, dtype=np.complex128)
            np.divide(np.nansum(data, axis=0), counts, out=mean, where=counts > 0)
            return mean

        @context.plots.on_plot_thread
        def update(data: NDArray[np.complex128]):
            mean = average_round(data)
            return tuple(
                artist
                for row, viewer in enumerate(viewers)
                for artist in viewer.set(scan.coordinates[row], np.abs(mean[:, row]))
            )

        signals = SignalBuffer((rounds, *scan.coordinates.T.shape), on_update=update)
        with Schedule(signals, scope=context.scope) as sched:
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
                        readout_module = MuxReadout("readout", readout)
                        builder.emit(readout_module)
                        programs[point] = builder.build()
                        readouts[point] = readout_module
                    prog = programs[point]
                    _ = step.run_acquire(
                        prog,
                        context.soc,
                        options=RunOptions(
                            converter=partial(
                                averaged_mux_signal, prog, readouts[point]
                            )
                        ),
                        rounds=1,
                    )
                # Check every tone only after a complete host-side point scan.
                mean = average_round(signals.array)
                if (
                    cfg.earlystop_snr is not None
                    and np.all(np.isfinite(mean))
                    and all(
                        estimate_snr(rotate2real(trace).real) >= cfg.earlystop_snr
                        for trace in mean.T
                    )
                ):
                    break
        signals.trigger_update(flush=True)
        return MuxSweepResult(
            np.asarray(scan.indices, dtype=np.int64),
            scan.coordinates,
            average_round(signals.array).T.copy(),
        )

    def analyze(
        self,
        source: RunRecord[MuxLenRabiCfg, MuxSweepResult],
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
        return analyze_length(source.result, options, plots=plots)
