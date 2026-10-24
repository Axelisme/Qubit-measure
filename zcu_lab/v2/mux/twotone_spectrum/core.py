"""Acquisition, analysis and persistence for zipped mux spectroscopy.

Technical reference for lab authors and framework developers.
MuxFreqExp.run(config, *, context) drives selected qubits simultaneously.

Configuration and timing
------------------------
The full tones list matches the complete MuxReadoutCfg in count and order.
Each MuxToneCfg has a fixed name, a required drive and a sweep. Participants is
a nonempty, unique list of original indices. Q3/Q1 selects original indices [2, 0]
from Q1/Q2/Q3 and retains that execution/data order. Every drive must be valid,
only for participants, on independent generators distinct from readout.
The caller supplies the complete qubit/resonator bindings; numeric cfg validation
does not resolve component identities.

Sweep ranges may differ; cfg.expts supplies one positive point count shared by
selected tones. Point p pairs each active tone's p-th value,
not a Cartesian product. One program sweeps drive DAC frequencies in a shared hardware loop.
Each point executes optional common reset, Join of independent drive pulses,
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
MuxFreqCfg.modules contains optional common reset and full mux readout.
Sweep bounds and returned coordinates are in MHz. prepare_frequency_scan
validates cfg and constructs independent rounded drive-frequency axes in
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
from pydantic import Field, StrictInt, model_validator
from qick import QickConfig
from zcu_tools.cfg_model import FieldMeta
from zcu_tools.experiment import PersistableExperiment, ProgramExpCfg
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.utils.sweep import make_sweep
from zcu_tools.experiment.v2.runtime import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import ModularProgramV2, sweep2param
from zcu_tools.program.v2.modules import Join, MuxReadout, Pulse

from zcu_lab.v2._support.mux import MuxFitFailure, MuxSweepResult
from zcu_lab.v2._support.mux_persistence import mux_sweep_spec
from zcu_lab.v2._support.mux_sweep_cfg import (
    MuxDriveModuleCfg,
    MuxToneCfg,
    validate_mux_scan,
)
from zcu_lab.v2.twotone.freq.core import FreqAnalyzeOptions

from .analysis import MuxFreqFit, analyze_frequency


class MuxFreqCfg(ProgramExpCfg):
    """Settings for simultaneous qubit frequency sweeps and mux readout.

    modules supplies optional reset and the full readout tone list. tones gives
    each original readout position a unique name, a drive and a MHz sweep.
    participants selects a nonempty, unique list of original indices in measurement order.
    All tones need distinct drive generators, separate from readout.
    expts is the positive hardware point count shared by all tones.
    Sweeps advance together by point, not as a Cartesian product. reps/rounds are positive averaging counts;
    initial_delay/relax_delay are finite nonnegative waits in us.
    """

    modules: MuxDriveModuleCfg
    tones: Annotated[list[MuxToneCfg], FieldMeta(element_label="name")]
    expts: int = Field(
        strict=True,
        gt=0,
        title="Expts",
        description="Hardware scan point count shared by all tones.",
    )
    participants: list[StrictInt]

    @model_validator(mode="after")
    def validate_scan(self) -> MuxFreqCfg:
        """Reject mismatched tones, shared drive channels or an invalid point count."""
        validate_mux_scan(self, self.modules, self.tones, self.participants, self.expts)
        return self


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


def prepare_frequency_scan(cfg: MuxFreqCfg, *, soccfg: QickConfig) -> MuxFreqScan:
    """Prepare each selected qubit's original index and rounded frequency axis.

    cfg is copied and validated. soccfg supplies the board frequency grids.
    Return MuxFreqScan with original indices and float64 (K,P) MHz coordinates
    in participant order. Coordinates use QICK's hardware start/span/step rounding.
    Sweeps advance together by point index. No per-point pulses are prepared.
    Invalid settings raise ValueError; frequency conversion errors propagate.
    This function does not acquire data or change cfg.
    """
    cfg = MuxFreqCfg.model_validate(cfg.model_dump())
    indices = list(cfg.participants)
    coordinates = []
    for index in indices:
        tone = cfg.tones[index]
        drive = tone.require_drive()
        actual = sweep2array(
            make_sweep(
                tone.require_sweep().start, tone.require_sweep().stop, cfg.expts
            ),
            "freq",
            {"soccfg": soccfg, "gen_ch": drive.ch, "mixer_freq": drive.mixer_freq},
        )
        coordinates.append(actual)
    return MuxFreqScan(indices, np.asarray(coordinates, dtype=np.float64))


def _averaged_signal(
    prog: ModularProgramV2, acquired: Sequence[NDArray[np.float64]]
) -> NDArray[np.complex128]:
    # Builder copies modules; collect only through the initialized program owner.
    for module in prog.modules:
        if isinstance(module, MuxReadout) and module.name == "readout":
            iq = module.collect_iq(prog, acquired, read_index=0)
            return iq[..., 0] + 1j * iq[..., 1]
    raise ValueError("Mux scan program is missing its named MuxReadout")


class MuxFreqExp(PersistableExperiment[MuxSweepResult, MuxFreqCfg]):
    """Measure qubit spectra with simultaneous drives and a shared mux readout.

    Each point optionally resets, starts selected drives together and reads out after
    the longest drive finishes. Analysis estimates each qubit's transition
    frequency and linewidth. Saved results retain the full settings, original
    tone positions, actual MHz axes and complex IQ.
    """

    AXES_SPEC = mux_sweep_spec(
        MuxFreqCfg,
        tag="mux/twotone_spectrum",
        coordinate_label="Drive frequency",
        unit="Hz",
        scale=1e6,
    )

    Options: ClassVar[type[FreqAnalyzeOptions]] = FreqAnalyzeOptions

    def run(self, config: MuxFreqCfg, *, context: RunContext) -> MuxSweepResult:
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
        Call context.scope.stop.raise_if_error() before reporting success.
        """
        # 1. Prepare zipped frequencies and readout channels in participant order.
        cfg = MuxFreqCfg.model_validate(config.model_dump())
        scan = prepare_frequency_scan(cfg, soccfg=context.soccfg)
        readout = cfg.modules.readout.with_updates(
            tones=[cfg.modules.readout.tones[index] for index in scan.indices]
        )

        # 2. Keep each spectrum on its own hardware-rounded frequency axis.
        viewers = [
            context.plots.liveplot_1d(
                f"measurement_{index}",
                "Drive frequency (MHz)",
                "Magnitude (a.u.)",
                title=f"{index}: {cfg.tones[index].name}",
            )
            for index in scan.indices
        ]

        def update(data: NDArray[np.complex128]) -> None:
            for row, viewer in enumerate(viewers):
                viewer.update(scan.coordinates[row], np.abs(data[:, row]))

        # 3. A shared DAC register loop advances the independent frequency ranges.
        signals = SignalBuffer(scan.coordinates.T.shape, on_update=update)
        with Schedule(cfg, signals, scope=context.scope) as sched:
            drives = []
            for index in scan.indices:
                tone = cfg.tones[index]
                drive = tone.require_drive().model_copy(deep=True)
                drive.set_param(
                    "freq",
                    sweep2param(
                        "point",
                        make_sweep(
                            tone.require_sweep().start,
                            tone.require_sweep().stop,
                            cfg.expts,
                        ),
                    ),
                )
                drives.append(Pulse(f"drive_{index}", drive))

            # Reset -> simultaneous drives -> fixed mux readout after the longest drive.
            builder = (
                sched.prog_builder(context.soc, context.soccfg)
                .add_reset("reset", cfg.modules.reset)
                .add(Join(*drives), MuxReadout("readout", readout))
                .declare_sweep("point", scan.coordinates.shape[1])
            )

            # 4. Each completed round supplies the full point/tone IQ matrix.
            prog = builder.build()
            _ = builder.run_program(prog, raw2signal_fn=partial(_averaged_signal, prog))

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
        selects the line shape and fit overlay. plots receives a fit_i figure
        per original tone index. Return estimates or a per-tone failure reason.
        """
        return analyze_frequency(source.result, options, plots=plots)
