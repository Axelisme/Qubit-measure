"""Acquisition, analysis and persistence for zipped mux amplitude Rabi."""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from typing import Annotated, ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import model_validator
from qick import QickConfig
from zcu_tools.cfg_model import FieldMeta
from zcu_tools.experiment import PersistableExperiment
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import round_zcu_gain, sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import ModularProgramV2, ProgramV2Cfg
from zcu_tools.program.v2.modules import Join, MuxReadout, Pulse

from zcu_lab.v2._support.mux import MuxFitFailure, MuxPulseScan, MuxSweepResult
from zcu_lab.v2._support.mux_liveplot import draw_mux_magnitudes
from zcu_lab.v2._support.mux_persistence import mux_sweep_spec
from zcu_lab.v2._support.mux_sweep_cfg import (
    MuxDriveModuleCfg,
    MuxToneCfg,
    participant_indices,
    validate_mux_scan,
)
from zcu_lab.v2.twotone.rabi.amp_rabi.core import AmpRabiAnalyzeOptions

from .analysis import MuxAmpRabiFit, analyze_amplitude


class MuxAmpRabiCfg(ProgramV2Cfg, ExpCfgModel):
    """Complete named tones, ordered participants and shared acquisition controls.

    modules holds optional reset and complete mux readout. tones contains one
    MuxToneCfg per original readout position, with unique nonempty names.
    participants is a nonempty, duplicate-free list of those names; its order
    determines execution/data rows (Q3/Q1 maps to original indices 2/0).
    Every drive is required/validated, including nonparticipants, on independent
    generators distinct from readout. Independent normalized-gain sweeps stay in [-1,1], with equal point counts.
    Inherited reps/rounds are positive; initial_delay/relax_delay are finite
    nonnegative us. No capture, bindings or writeback destinations are stored.
    """

    modules: MuxDriveModuleCfg
    tones: Annotated[list[MuxToneCfg], FieldMeta(element_label="name")]
    participants: list[str]

    @model_validator(mode="after")
    def validate_scan(self) -> MuxAmpRabiCfg:
        """Reject invalid full length, drive channels, counts or gain ranges."""
        validate_mux_scan(self, self.modules, self.tones, self.participants)
        for index, tone in enumerate(self.tones):
            if max(abs(tone.sweep.start), abs(tone.sweep.stop)) > 1:
                raise ValueError(f"tones[{index}].sweep gains must be in [-1,1]")
        return self


def prepare_amplitude_scan(cfg: MuxAmpRabiCfg, *, soccfg: QickConfig) -> MuxPulseScan:
    """Prepare simultaneous gain sweeps on the board's actual gain grids.

    cfg supplies the complete tones and ordered participants; it is not mutated.
    soccfg describes the target board. Each participant gets its own normalized
    gain coordinates and one pulse per point. Sweeps advance together by point
    index, not as a Cartesian product. Pulses retain nominal gains for compilation.
    Invalid cfg raises ValueError; board conversion errors propagate.
    """
    cfg = MuxAmpRabiCfg.model_validate(cfg.model_dump())
    indices = participant_indices([tone.name for tone in cfg.tones], cfg.participants)
    coordinates = []
    points = [[] for _ in range(cfg.tones[indices[0]].sweep.expts)]
    for index in indices:
        tone = cfg.tones[index]
        drive = tone.drive
        nominal = sweep2array(tone.sweep, "none")
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
        shape (participants, points), in participant order, with normalized gains.
        Cancellation keeps completed averages and leaves unmeasured points NaN.
        Call context.cancel_signal.raise_if_error() before reporting success;
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
                f"measurement_{index}", "Pulse gain", "Magnitude (a.u.)"
            )
            for index in scan.indices
        ]

        def update(data: NDArray[np.complex128]) -> None:
            for row, viewer in enumerate(viewers):
                viewer.update(scan.coordinates[row], np.abs(data[:, row]))
            draw_mux_magnitudes(
                context.plots,
                scan.coordinates,
                data.T,
                [f"{index}: {cfg.tones[index].name}" for index in scan.indices],
                "Pulse gain",
            )

        # Normalize averaged ADC I/Q using the compiled readout windows.
        def averaged_signal(
            prog: ModularProgramV2, acquired: Sequence[NDArray[np.float64]]
        ) -> NDArray[np.complex128]:
            # Builder copies modules; read IQ through the initialized program.
            for module in prog.modules:
                if isinstance(module, MuxReadout) and module.name == "readout":
                    iq = module.collect_iq(prog, acquired, read_index=0)
                    return iq[:, 0] + 1j * iq[:, 1]
            raise ValueError("Mux scan program is missing its named MuxReadout")

        # Store points first so each acquisition writes one complete tone vector.
        signals = SignalBuffer(scan.coordinates.T.shape, on_update=update)
        with Schedule(cfg, signals, stop=context.cancel_signal) as sched:
            for pulses, step in sched.scan("point", scan.point_pulses):
                # Reset, start all drives together, then read after the longest.
                builder = (
                    step.prog_builder(context.soc, context.soccfg)
                    .add_reset("reset", cfg.modules.reset)
                    .add(
                        Join(
                            *[
                                Pulse(f"drive_{index}", pulse)
                                for index, pulse in zip(
                                    scan.indices, pulses, strict=True
                                )
                            ]
                        ),
                        MuxReadout("readout", readout),
                    )
                )

                # Compile once per point; preserve completed-round partial averages.
                prog = builder.build()
                _ = builder.run_program(
                    prog, raw2signal_fn=partial(averaged_signal, prog)
                )

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
