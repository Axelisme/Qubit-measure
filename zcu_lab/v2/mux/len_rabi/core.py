"""Mux length Rabi with simultaneous drives and per-qubit length sweeps."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from functools import partial
from typing import Annotated, ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import model_validator
from qick import QickConfig
from zcu_tools.cfg_model import FieldMeta
from zcu_tools.experiment import US_TO_S, PersistableExperiment
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import ModularProgramV2, ProgramV2Cfg
from zcu_tools.program.v2.modules import (
    ConstWaveformCfg,
    FlatTopWaveformCfg,
    Join,
    MuxReadout,
    Pulse,
)

from zcu_lab.v2._support.mux import MuxFitFailure, MuxPulseScan, MuxSweepResult
from zcu_lab.v2._support.mux_liveplot import draw_mux_magnitudes
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
            return iq[:, 0] + 1j * iq[:, 1]
    raise ValueError("Mux scan program is missing its named MuxReadout")


class MuxLenRabiCfg(ProgramV2Cfg, ExpCfgModel):
    """Complete named tones, ordered participants and shared acquisition controls.

    modules holds optional reset and complete mux readout. tones contains one
    MuxToneCfg per original readout position, with unique nonempty names.
    participants is a nonempty, duplicate-free list of those names; its order
    determines execution/data rows (Q3/Q1 maps to original indices 2/0).
    Every drive is required/validated, including nonparticipants, on independent
    generators distinct from readout. Independent positive us length sweeps
    have equal point counts. Every waveform
    must be const or flat_top; arb duration belongs to its asset, not this sweep.
    Inherited reps/rounds are positive; initial_delay/relax_delay are finite
    nonnegative us. No capture, bindings or writeback destinations are stored.
    """

    modules: MuxDriveModuleCfg
    tones: Annotated[list[MuxToneCfg], FieldMeta(element_label="name")]
    participants: list[str]

    @model_validator(mode="after")
    def validate_scan(self) -> MuxLenRabiCfg:
        """Reject invalid full tones, active generators, counts or length policy."""
        validate_mux_scan(self, self.modules, self.tones, self.participants)
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


def prepare_length_scan(cfg: MuxLenRabiCfg, *, soccfg: QickConfig) -> MuxPulseScan:
    """Prepare each selected qubit's pulse-length grid without acquiring data.

    cfg supplies complete tones and ordered participants; soccfg supplies the
    board clocks. Return original indices, float64 lengths (K,P) in us and
    copied pulses (P,K), pairing the qubits' own sweep points rather than
    taking their Cartesian product. Each length follows its drive generator's
    time grid and includes flat-top ramps. Playback may differ by clock ticks.

    Inputs stay unchanged. Invalid cfg, nonpositive rounded lengths, collapsed
    sweeps and waveform conversion errors raise before acquisition.
    """
    cfg = MuxLenRabiCfg.model_validate(cfg.model_dump())
    indices = participant_indices([tone.name for tone in cfg.tones], cfg.participants)
    coordinates = []
    points = [[] for _ in range(cfg.tones[indices[0]].sweep.expts)]
    for index in indices:
        tone = cfg.tones[index]
        drive = tone.drive
        lengths = sweep2array(
            tone.sweep, "time", {"soccfg": soccfg, "gen_ch": drive.ch}
        )
        if np.any(lengths <= 0):
            raise ValueError(
                f"tones[{index}].sweep quantized lengths must be positive us"
            )
        coordinates.append(lengths)
        for point, length in enumerate(lengths):
            pulse = deepcopy(drive)
            pulse.set_param("length", float(length))
            points[point].append(pulse)
    return MuxPulseScan(indices, np.asarray(coordinates, dtype=np.float64), points)


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
        Check context.cancel_signal.raise_if_error() before reporting success;
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
                f"measurement_{index}", "Pulse length (us)", "Magnitude (a.u.)"
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
                "Pulse length (us)",
            )

        # Point-major storage lets each scan step write one whole tone vector.
        signals = SignalBuffer(scan.coordinates.T.shape, on_update=update)
        with Schedule(cfg, signals, stop=context.cancel_signal) as sched:
            for pulses, step in sched.scan("point", scan.point_pulses):
                # 2. Reset, start drives together, then read after the longest drive.
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
                prog = builder.build()

                # 3. Average IQ in participant order, including completed rounds on stop.
                _ = builder.run_program(
                    prog, raw2signal_fn=partial(_averaged_signal, prog)
                )
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
