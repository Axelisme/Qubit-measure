"""Mux length Rabi with simultaneous drives and per-qubit length sweeps.

Technical reference for lab authors and framework developers.
MuxLenRabiExp.run(config, *, context) sweeps selected drive lengths together.

Configuration and timing
------------------------
The full tones list matches the complete MuxReadoutCfg in count and order.
Each MuxToneCfg has a fixed name, a required drive and a sweep. Participants is
a nonempty, unique list of existing names. Q3/Q1 selects original indices [2, 0]
from Q1/Q2/Q3 and retains that execution/data order. Every drive must be valid,
including nonparticipants, on independent generators distinct from readout.
The caller supplies the complete qubit/resonator bindings; numeric cfg validation
does not resolve component identities.

Sweep ranges may differ; cfg.expts supplies one positive point count shared by
all tones, including nonparticipants. Point p pairs each active tone's p-th value,
not a Cartesian product. One program executes a shared hardware point loop with
optional common reset, Join of table-length drive pulses, then active mux readout.
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
distinct plots. Each completed round updates the full sweep. Cancellation keeps
completed-round averages across all points, or NaN when no round completed.

Length scan contract
--------------------
MuxLenRabiCfg.modules contains optional common reset and full mux readout.
Sweep bounds are positive requested lengths in us. Every tone must use const
or flat_top, including nonparticipants. Const length is the entire pulse.
Flat_top length includes raise_waveform ramps; the waveform implementation
subtracts ramps to determine the plateau. Gain, frequency, phase and ramp
settings stay fixed. Arb assets have fixed length and are rejected by cfg and
run validation rather than stretched or cropped.

prepare_length_scan(cfg, soccfg=...) copies cfg and resolves original indices
in participant order. sweep2array(..., "time", gen_ch=...) creates each drive's
rounded requested-length grid for TableLengthPulse. Each grid retains its
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
    context.scope.stop.raise_if_error()
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
from pydantic import Field, model_validator
from qick import QickConfig
from zcu_tools.cfg_model import FieldMeta
from zcu_tools.experiment import US_TO_S, PersistableExperiment
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.utils.sweep import make_sweep
from zcu_tools.experiment.v2.runtime import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import ModularProgramV2, ProgramV2Cfg
from zcu_tools.program.v2.modules import (
    ConstWaveformCfg,
    FlatTopWaveformCfg,
    Join,
    MuxReadout,
    TableLengthPulse,
)

from zcu_lab.v2._support.mux import MuxFitFailure, MuxSweepResult
from zcu_lab.v2._support.mux_persistence import mux_sweep_spec
from zcu_lab.v2._support.mux_sweep_cfg import (
    MuxDriveModuleCfg,
    MuxToneCfg,
    participant_indices,
    validate_mux_scan,
)
from zcu_lab.v2.twotone.rabi.len_rabi.core import LenRabiAnalyzeOptions

from .analysis import MuxLenRabiFit, analyze_length


def _averaged_signal(
    prog: ModularProgramV2, acquired: Sequence[NDArray[np.float64]]
) -> NDArray[np.complex128]:
    # Builder copies modules; collect only through the initialized readout owner.
    for module in prog.modules:
        if isinstance(module, MuxReadout) and module.name == "readout":
            iq = module.collect_iq(prog, acquired, read_index=0)
            return iq[..., 0] + 1j * iq[..., 1]
    raise ValueError("Mux scan program is missing its named MuxReadout")


class MuxLenRabiCfg(ProgramV2Cfg, ExpCfgModel):
    """Complete named tones, ordered participants and shared acquisition controls.

    modules holds optional reset and complete mux readout. tones contains one
    MuxToneCfg per original readout position, with unique nonempty names.
    participants is a nonempty, duplicate-free list of those names; its order
    determines execution/data rows (Q3/Q1 maps to original indices 2/0).
    Every drive is required/validated, including nonparticipants, on independent
    generators distinct from readout. Independent positive us length ranges
    use expts, the positive hardware point count shared by all tones. Every waveform
    must be const or flat_top; arb duration belongs to its asset, not this sweep.
    Inherited reps/rounds are positive; initial_delay/relax_delay are finite
    nonnegative us. No capture, bindings or writeback destinations are stored.
    """

    modules: MuxDriveModuleCfg
    tones: Annotated[list[MuxToneCfg], FieldMeta(element_label="name")]
    expts: int = Field(
        strict=True,
        gt=0,
        title="Expts",
        description="Hardware scan point count shared by all tones.",
    )
    participants: list[str]

    @model_validator(mode="after")
    def validate_scan(self) -> MuxLenRabiCfg:
        """Reject invalid full tones, active generators, counts or length policy."""
        validate_mux_scan(self, self.modules, self.tones, self.participants, self.expts)
        for index, tone in enumerate(self.tones):
            drive = tone.drive
            if not isinstance(drive.waveform, (ConstWaveformCfg, FlatTopWaveformCfg)):
                raise ValueError(
                    f"tones[{index}].drive.waveform: length sweep requires const or "
                    "flat_top; arb length is asset-derived with no public length sweep"
                )
            if min(tone.sweep.start, tone.sweep.stop) <= 0:
                raise ValueError(f"tones[{index}].sweep lengths must be positive us")
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
    indices = participant_indices([tone.name for tone in cfg.tones], cfg.participants)
    coordinates = []
    for index in indices:
        tone = cfg.tones[index]
        drive = tone.drive
        lengths = sweep2array(
            make_sweep(tone.sweep.start, tone.sweep.stop, cfg.expts),
            "time",
            {"soccfg": soccfg, "gen_ch": drive.ch},
        )
        if np.any(lengths <= 0):
            raise ValueError(
                f"tones[{index}].sweep quantized lengths must be positive us"
            )
        coordinates.append(lengths)
    return MuxLengthScan(indices, np.asarray(coordinates, dtype=np.float64))


class MuxLenRabiExp(PersistableExperiment[MuxSweepResult, MuxLenRabiCfg]):
    """Measure Rabi oscillations by sweeping simultaneous qubit pulse lengths.

    Each point resets, starts the selected drives together, waits for the longest
    drive, then reads their resonators together. Results retain each qubit's
    length axis in us and average complex IQ; fits estimate pi and pi/2 lengths.
    """

    AXES_SPEC = mux_sweep_spec(
        MuxLenRabiCfg,
        tag="mux/rabi/len",
        coordinate_label="Pulse length",
        unit="s",
        scale=US_TO_S,
    )

    Options: ClassVar[type[LenRabiAnalyzeOptions]] = LenRabiAnalyzeOptions

    def run(self, config: MuxLenRabiCfg, *, context: RunContext) -> MuxSweepResult:
        """Reset, sweep simultaneous drive lengths, and acquire mux readout IQ.

        config supplies full tones, ordered participants and shared averaging
        and delays. Each point pairs the participants' own quantized lengths;
        Join starts them together and readout waits for the last drive, without
        end alignment or sweep-wide padding. Return original tone indices (K,),
        requested total lengths (K,P) in us and average complex IQ (K,P) in
        ADC-length-normalized a.u. Flat-top lengths include their ramps.

        context supplies soc/soccfg, plots and a fresh stop signal. Cancellation
        retains completed-round averages and leaves unmeasured points NaN.
        Check context.scope.stop.raise_if_error() before reporting success;
        the caller owns plot and hardware cleanup. Invalid cfg/builds raise.
        """
        # 1. Prepare each qubit's length grid and select its original readout tone.
        cfg = MuxLenRabiCfg.model_validate(config.model_dump())
        scan = prepare_length_scan(cfg, soccfg=context.soccfg)
        readout = cfg.modules.readout.with_updates(
            tones=[cfg.modules.readout.tones[index] for index in scan.indices]
        )
        viewers = [
            context.plots.liveplot_1d(
                f"measurement_{index}",
                "Pulse length (us)",
                "Magnitude (a.u.)",
                title=f"{index}: {cfg.tones[index].name}",
            )
            for index in scan.indices
        ]

        def update(data: NDArray[np.complex128]) -> None:
            for row, viewer in enumerate(viewers):
                viewer.update(scan.coordinates[row], np.abs(data[:, row]))

        # 2. One hardware point loop selects all participants' length tables.
        signals = SignalBuffer(scan.coordinates.T.shape, on_update=update)
        with Schedule(cfg, signals, scope=context.scope) as sched:
            drives = [
                TableLengthPulse(
                    f"drive_{index}",
                    cfg.tones[index].drive,
                    lengths=scan.coordinates[row],
                    idx_reg="point",
                )
                for row, index in enumerate(scan.indices)
            ]
            # Reset -> simultaneous drives -> readout after this point's longest drive.
            builder = (
                sched.prog_builder(context.soc, context.soccfg)
                .add_reset("reset", cfg.modules.reset)
                .add(Join(*drives), MuxReadout("readout", readout))
                .declare_sweep("point", cfg.expts)
            )
            prog = builder.build()

            # 3. Publish the full point/tone matrix after each completed round.
            _ = builder.run_program(prog, raw2signal_fn=partial(_averaged_signal, prog))
        signals.trigger_update(flush=True)
        return MuxSweepResult(
            np.asarray(scan.indices, dtype=np.int64),
            scan.coordinates,
            signals.array.T.copy(),
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
