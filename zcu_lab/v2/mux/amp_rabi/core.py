"""Acquisition, analysis and persistence for zipped mux amplitude Rabi.

Technical reference for lab authors and framework developers.
MuxAmpRabiExp.run(config, *, context) sweeps participant gains simultaneously.

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
not a Cartesian product. One program advances all gains in a shared hardware loop.
Each point executes optional common reset, Join of independent drive pulses,
then active mux readout. Branches start together; short branches wait for the
longest branch at that point. Pre/post delays stay
unchanged. There is no end alignment, end padding or sweep-wide maximum padding.
All participants share reps, rounds, initial_delay and relax_delay.
Program modules perform native conversion and averaging.

Results and cancellation
------------------------
MuxSweepResult.tone_indices is int64 (K,) in participant order. Coordinates are
float64 (K,P) on each drive's actual hardware-quantized gain grid. Signals are
complex128 (K,P), ADC-length-normalized average IQ in a.u. The measurement_i magnitude plots use
original tone indices and each tone's own coordinates, so duplicate labels retain
distinct plots. Cancellation preserves completed-round averages across the full
sweep. If no round completes, all signals remain NaN.

Amplitude scan and analysis
--------------------------
MuxAmpRabiCfg.modules contains optional common reset and full mux readout.
Sweep bounds are normalized gains within [-1, 1]. prepare_amplitude_scan copies
cfg and builds independently rounded scalar gain grids. run publishes the
compiled integer-step gain grids instead.

Analysis uses the single-qubit undamped Rabi fitter and zero-gain phase policy.
Options.skip removes leading zipped points before finite-IQ filtering. Each
curve needs six remaining points. Each original index returns MuxAmpRabiFit or
MuxFitFailure with a reason, and a fit_i figure. Pi/pi2 gains and errors use
normalized gain units. FitQuality describes the numerical fit; a finite fit
alone does not establish calibration quality.

Native persistence retains the full cfg, original indices, independent gain axes
and complex IQ. The source pulse and its scalar length remain available to a
calibration caller for pulse-copy and pi-product proposals.

Calling and ownership
---------------------
For context, plots, stop/error handling and record/persistence ownership, see
lib/zcu_tools/experiment/README.md, section "實驗介面與資料", and
lib/zcu_tools/experiment/v2/runtime/README.md, section "Stop 與 Retry".

Analysis uses the explicit historical source, not the current Form or library.
The numerical core leaves component bindings and writeback to its caller.

Example with validated cfg, RunContext and fresh analysis_plots::

    from zcu_tools.experiment.records import RunRecord
    from zcu_lab.v2.mux.amp_rabi import MuxAmpRabiExp

    core = MuxAmpRabiExp()
    result = core.run(cfg, context=context)
    context.scope.stop.raise_if_error()
    source = RunRecord(cfg=cfg, result=result)
    outcomes = core.analyze(source, core.Options(), plots=analysis_plots)
"""

from __future__ import annotations

from collections.abc import Sequence
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
from zcu_tools.experiment.v2.utils.round_zcu import round_zcu_gain, sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import ModularProgramV2
from zcu_tools.program.v2.modules import Join, MuxReadout, Pulse
from zcu_tools.program.v2.utils import sweep2param

from zcu_lab.v2._support.mux import MuxFitFailure, MuxPulseScan, MuxSweepResult
from zcu_lab.v2._support.mux_persistence import mux_sweep_spec
from zcu_lab.v2._support.mux_sweep_cfg import (
    MuxDriveModuleCfg,
    MuxToneCfg,
    validate_mux_scan,
)
from zcu_lab.v2.twotone.rabi.amp_rabi.core import AmpRabiAnalyzeOptions

from .analysis import MuxAmpRabiFit, analyze_amplitude


class MuxAmpRabiCfg(ProgramExpCfg):
    """Complete named tones, ordered participants and shared acquisition controls.

    modules holds optional reset and complete mux readout. tones contains one
    MuxToneCfg per original readout position, with unique nonempty names.
    participants is a nonempty, duplicate-free list of original tone indices; its order
    determines execution/data rows (Q3/Q1 maps to original indices 2/0).
    Selected drives are required/validated on independent
    generators distinct from readout. Independent normalized-gain ranges stay
    in [-1,1]; expts is the positive hardware point count shared by all tones.
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
    participants: list[StrictInt]

    @model_validator(mode="after")
    def validate_scan(self) -> MuxAmpRabiCfg:
        """Reject invalid full length, drive channels, counts or gain ranges."""
        validate_mux_scan(self, self.modules, self.tones, self.participants, self.expts)
        for index in self.participants:
            tone = self.tones[index]
            if max(abs(tone.require_sweep().start), abs(tone.require_sweep().stop)) > 1:
                raise ValueError(f"tones[{index}].sweep gains must be in [-1,1]")
        return self


def prepare_amplitude_scan(cfg: MuxAmpRabiCfg, *, soccfg: QickConfig) -> MuxPulseScan:
    """Prepare simultaneous gain sweeps on the board's actual gain grids.

    cfg supplies the complete tones and ordered participants; it is not mutated.
    soccfg describes the target board. Each participant gets its own normalized
    gain coordinates and one pulse per point, rounded as independent scalar
    pulses. Sweeps advance together by point index, not as a Cartesian product.
    Pulses retain nominal gains for compilation. run replaces these preliminary
    coordinates with the compiled hardware sweep's integer-step gain grid.
    Invalid cfg raises ValueError; board conversion errors propagate.
    """
    cfg = MuxAmpRabiCfg.model_validate(cfg.model_dump())
    indices = list(cfg.participants)
    coordinates = []
    points = [[] for _ in range(cfg.expts)]
    for index in indices:
        tone = cfg.tones[index]
        drive = tone.require_drive()
        nominal = sweep2array(
            make_sweep(
                tone.require_sweep().start, tone.require_sweep().stop, cfg.expts
            ),
            "none",
        )
        actual = round_zcu_gain(nominal, soccfg, drive.ch)
        coordinates.append(actual)
        for point, value in enumerate(nominal):
            points[point].append(drive.with_updates(gain=float(value)))
    return MuxPulseScan(indices, np.asarray(coordinates, dtype=np.float64), points)


class MuxAmpRabiExp(PersistableExperiment[MuxSweepResult, MuxAmpRabiCfg]):
    """Measure each participant's Rabi oscillation versus drive gain.

    Reset precedes simultaneous drives and a shared mux readout at each point.
    Fits give pi and pi/2 gains. Results retain each tone's actual gain grid,
    complex IQ and original index; saved data also retains the complete cfg.
    """

    AXES_SPEC = mux_sweep_spec(
        MuxAmpRabiCfg,
        tag="mux/rabi/amp",
        coordinate_label="Pulse gain",
        unit="",
        scale=1.0,
    )

    Options: ClassVar[type[AmpRabiAnalyzeOptions]] = AmpRabiAnalyzeOptions

    def run(self, config: MuxAmpRabiCfg, *, context: RunContext) -> MuxSweepResult:
        """Sweep participant gains together and measure their averaged complex IQ.

        config supplies reset, readout, drives, gain sweeps and averaging controls;
        it is copied before use. After reset, all drives start together. Readout
        waits for the longest drive at that point, without end-alignment padding.
        context supplies the board, plots and stop signal. Coordinates and IQ have
        shape (participants, points), in participant order, with actual hardware-
        quantized normalized gains. All gains advance in one shared hardware loop.
        Each completed round updates the full sweep's live plots. Cancellation
        keeps completed-round averages, or NaN if no round completes.
        Call context.scope.stop.raise_if_error() before reporting success;
        the caller also finishes/releases plots. Invalid cfg/build errors propagate.
        """
        # Prepare each gain grid and select readout tones in participant order.
        cfg = MuxAmpRabiCfg.model_validate(config.model_dump())
        scan = prepare_amplitude_scan(cfg, soccfg=context.soccfg)
        readout = cfg.modules.readout.with_updates(
            tones=[cfg.modules.readout.tones[index] for index in scan.indices]
        )

        # Keep live traces on each drive's own quantized gain axis.
        viewers = [
            context.plots.liveplot_1d(
                f"measurement_{index}",
                "Pulse gain",
                "Magnitude (a.u.)",
                title=f"{index}: {cfg.tones[index].name}",
            )
            for index in scan.indices
        ]

        def update(data: NDArray[np.complex128]) -> None:
            for row, viewer in enumerate(viewers):
                viewer.update(scan.coordinates[row], np.abs(data[:, row]))

        # Normalize averaged ADC I/Q using the compiled readout windows.
        def averaged_signal(
            prog: ModularProgramV2, acquired: Sequence[NDArray[np.float64]]
        ) -> NDArray[np.complex128]:
            # Builder copies modules; read IQ through the initialized program.
            for module in prog.modules:
                if isinstance(module, MuxReadout) and module.name == "readout":
                    iq = module.collect_iq(prog, acquired, read_index=0)
                    return iq[..., 0] + 1j * iq[..., 1]
            raise ValueError("Mux scan program is missing its named MuxReadout")

        # Advance each drive gain in the same tProc point loop.
        signals = SignalBuffer(scan.coordinates.T.shape, on_update=update)
        with Schedule(cfg, signals, scope=context.scope) as sched:
            drives = []
            for index in scan.indices:
                tone = cfg.tones[index]
                drive = tone.require_drive().model_copy(deep=True)
                drive.set_param(
                    "gain",
                    sweep2param(
                        "point",
                        make_sweep(
                            tone.require_sweep().start,
                            tone.require_sweep().stop,
                            cfg.expts,
                        ),
                    ),
                )
                drives.append(Pulse(f"drive_{index}", drive, pulse_id=f"drive_{index}"))

            # Reset -> simultaneous drives -> fixed readout after the longest drive.
            builder = (
                sched.prog_builder(context.soc, context.soccfg)
                .add_reset("reset", cfg.modules.reset)
                .add(Join(*drives), MuxReadout("readout", readout))
                .declare_sweep("point", cfg.expts)
            )

            # Integer register steps can differ from independently rounded points.
            # Keep live plots, analysis and persistence on the compiled gain grid.
            prog = builder.build()
            for row, index in enumerate(scan.indices):
                scan.coordinates[row] = prog.get_pulse_param(
                    f"drive_{index}", "gain", as_array=True
                )
            _ = builder.run_program(prog, raw2signal_fn=partial(averaged_signal, prog))

        # Publish final plots and return tone-major data, including unmeasured NaNs.
        signals.trigger_update(flush=True)
        return MuxSweepResult(
            np.asarray(scan.indices, dtype=np.int64),
            scan.coordinates,
            signals.array.T.copy(),
        )

    def analyze(
        self,
        source: RunRecord[MuxAmpRabiCfg, MuxSweepResult],
        options: AmpRabiAnalyzeOptions,
        *,
        plots: Plots,
    ) -> tuple[MuxAmpRabiFit | MuxFitFailure, ...]:
        """Fit each tone's IQ oscillation to obtain pi and pi/2 gains.

        source contains the acquired gain grids and IQ; its cfg may be None.
        options.skip omits leading points. Return gains, errors and fit quality,
        or a failure reason per original tone index. plots receives fit_i figures.
        """
        return analyze_amplitude(source.result, options, plots=plots)
