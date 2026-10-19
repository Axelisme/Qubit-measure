"""Acquisition, analysis and persistence for zipped mux spectroscopy."""

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
from zcu_tools.experiment.v2.utils.round_zcu import round_zcu_freq, sweep2array
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
from zcu_lab.v2.twotone.freq.core import FreqAnalyzeOptions

from .analysis import MuxFreqFit, analyze_frequency


class MuxFreqCfg(ProgramV2Cfg, ExpCfgModel):
    """Settings for simultaneous qubit frequency sweeps and mux readout.

    modules supplies optional reset and the full readout tone list. tones gives
    each original readout position a unique name, a drive and a MHz sweep.
    participants selects a nonempty, unique list of names in measurement order.
    All tones need distinct drive generators, separate from readout, and equal
    sweep point counts, even when not selected. Sweeps advance together by point,
    not as a Cartesian product. reps/rounds are positive averaging counts;
    initial_delay/relax_delay are finite nonnegative waits in us.
    """

    modules: MuxDriveModuleCfg
    tones: Annotated[list[MuxToneCfg], FieldMeta(element_label="name")]
    participants: list[str]

    @model_validator(mode="after")
    def validate_scan(self) -> MuxFreqCfg:
        """Reject mismatched tones, shared drive channels or unequal sweep counts."""
        validate_mux_scan(self, self.modules, self.tones, self.participants)
        return self


def prepare_frequency_scan(cfg: MuxFreqCfg, *, soccfg: QickConfig) -> MuxPulseScan:
    """Prepare each selected qubit's frequency axis and per-point drive pulses.

    cfg is copied and validated. soccfg supplies the board frequency grids.
    Returned coordinates are actual MHz in participant order; point_pulses keeps
    nominal frequencies for compilation. Sweeps advance together by point index.
    Invalid settings raise ValueError; frequency conversion errors propagate.
    This function does not acquire data or change cfg.
    """
    cfg = MuxFreqCfg.model_validate(cfg.model_dump())
    indices = participant_indices([tone.name for tone in cfg.tones], cfg.participants)
    coordinates = []
    points = [[] for _ in range(cfg.tones[indices[0]].sweep.expts)]
    for index in indices:
        tone = cfg.tones[index]
        drive = tone.drive
        nominal = sweep2array(tone.sweep, "none")
        actual = round_zcu_freq(nominal, soccfg, drive.ch, mixer_freq=drive.mixer_freq)
        coordinates.append(actual)
        for point, value in enumerate(nominal):
            points[point].append(drive.with_updates(freq=float(value)))
    return MuxPulseScan(indices, np.asarray(coordinates, dtype=np.float64), points)


def _averaged_signal(
    prog: ModularProgramV2, acquired: Sequence[NDArray[np.float64]]
) -> NDArray[np.complex128]:
    # Builder copies modules; collect only through the initialized program owner.
    for module in prog.modules:
        if isinstance(module, MuxReadout) and module.name == "readout":
            iq = module.collect_iq(prog, acquired, read_index=0)
            return iq[:, 0] + 1j * iq[:, 1]
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
        tag="mux/freq",
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
        tone indices. Live plots show each spectrum and their combined magnitudes.
        Cancellation keeps acquired averages and leaves missing points NaN;
        call context.cancel_signal.raise_if_error() before reporting success.
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
                f"measurement_{index}", "Drive frequency (MHz)", "Magnitude (a.u.)"
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
                "Drive frequency (MHz)",
            )

        # Point-major storage lets each scan step write a whole tone vector.
        signals = SignalBuffer(scan.coordinates.T.shape, on_update=update)
        with Schedule(cfg, signals, stop=context.cancel_signal) as sched:
            for pulses, step in sched.scan("point", scan.point_pulses):
                # 3. Reset -> simultaneous drives -> readout after the longest drive.
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

                # 4. Average acquired rounds, preserving partial data on cancellation.
                prog = builder.build()
                _ = builder.run_program(
                    prog, raw2signal_fn=partial(_averaged_signal, prog)
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
        selects the line shape and fit overlay. plots receives a fit_i figure
        per original tone index. Return estimates or a per-tone failure reason.
        """
        return analyze_frequency(source.result, options, plots=plots)
