"""Simultaneous probe-off/on raw mux acquisition and explicit-source persistence."""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import StrictInt, model_validator
from qick.asm_v2 import QickParam
from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import AxesSpec, Axis, PersistableExperiment, RunRecord, ZSpec
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.v2.runtime import Schedule, SignalBuffer
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import ModularProgramV2, ProgramV2Cfg, PulseCfg
from zcu_tools.program.v2.modules import (
    ConstWaveformCfg,
    FlatTopWaveformCfg,
    Join,
    MuxReadout,
    Pulse,
)

from zcu_lab.v2._support.mux import MuxFitFailure
from zcu_lab.v2._support.mux_sweep_cfg import MuxDriveModuleCfg, participant_indices

from .analysis import (
    MuxGEAnalysis,
    MuxGEAnalyzeOptions,
    MuxGEPostAnalysis,
    MuxGEPostOptions,
    analyze_ge,
    post_analyze_ge,
)


def _validate_fixed_pulse(pulse: PulseCfg) -> None:
    for name in ("freq", "gain", "phase", "pre_delay", "post_delay"):
        value = getattr(pulse, name)
        if isinstance(value, QickParam) or not math.isfinite(value):
            raise ValueError(f"{name} must be fixed and finite")
    if pulse.pre_delay < 0 or pulse.post_delay < 0:
        raise ValueError("pulse delays must be nonnegative us")
    if isinstance(
        pulse.waveform, (ConstWaveformCfg, FlatTopWaveformCfg)
    ) and isinstance(pulse.waveform.length, QickParam):
        raise ValueError("pulse length must be fixed")


class MuxGEToneCfg(ConfigBase):
    """One named original tone's required probe drive and optional init pulse.

    name is the nonempty capture-generated qubit identity. drive is probe-on;
    init_pulse, when present, runs before it in both preparations on the same
    generator. All pulses are fixed native PulseCfg (MHz, us, degrees, gain),
    validated even for nonparticipants. Nonparticipant settings do not run.
    """

    name: str
    drive: PulseCfg
    init_pulse: PulseCfg | None = None

    @model_validator(mode="after")
    def validate_pulses(self) -> MuxGEToneCfg:
        """Reject swept values or shared-branch generators for every element."""
        _validate_fixed_pulse(self.drive)
        if self.init_pulse is not None:
            _validate_fixed_pulse(self.init_pulse)
            if self.init_pulse.ch != self.drive.ch:
                raise ValueError("init_pulse must use the tone's drive generator")
        return self


class MuxGECfg(ProgramV2Cfg, ExpCfgModel):
    """Complete named tones, ordered participants and shared acquisition controls.

    modules holds optional reset and complete mux readout. tones contains one
    MuxGEToneCfg per original readout position, with unique nonempty names.
    participants is a nonempty, duplicate-free list of those names; its order
    determines execution/data rows (Q3/Q1 maps to original indices 2/0).
    Every drive is required/validated, including nonparticipants, on independent
    generators distinct from readout. shots is positive. Run warns and forces rounds=1 and reps=shots; the
    original requested reps/rounds remain in RunRecord.cfg.
    Inherited reps/rounds are positive; initial_delay/relax_delay are finite
    nonnegative us. No capture, bindings or writeback destinations are stored.
    """

    modules: MuxDriveModuleCfg
    tones: list[MuxGEToneCfg]
    participants: list[str]
    shots: StrictInt

    @model_validator(mode="after")
    def validate_acquisition(self) -> MuxGECfg:
        """Reject invalid counts, full-list length or active channel conflicts."""
        if self.shots <= 0 or self.reps <= 0 or self.rounds <= 0:
            raise ValueError("shots, reps and rounds must be positive")
        if any(
            not math.isfinite(v) or v < 0
            for v in (self.initial_delay, self.relax_delay)
        ):
            raise ValueError(
                "initial_delay and relax_delay must be finite nonnegative us"
            )
        if len(self.tones) != len(self.modules.readout.tones):
            raise ValueError("tones must match the complete readout tone count")
        participant_indices([tone.name for tone in self.tones], self.participants)
        channels: set[int] = set()
        for index, tone in enumerate(self.tones):
            if (
                tone.drive.ch in channels
                or tone.drive.ch == self.modules.readout.gen_ch
            ):
                raise ValueError(
                    f"tones[{index}].drive requires an independent generator"
                )
            channels.add(tone.drive.ch)
        return self


@dataclass(frozen=True)
class MuxGEResult:
    """Paired ADC-length-normalized raw complex IQ, not averaged or classified.

    signals is complex128 (K,2,S), original active tone order, probe off/on,
    then shot. shot_indices is int64 arange(S), prepared_states is int64 [0,1]
    (probe off/on, not physical pure-state labels), tone_indices is int64 (K,)
    of original full-readout indices. All tones at a preparation/shot position
    share the same acquisition shot. The two preparations are separate acquires,
    not a four-state joint assignment calibration. Exactly one completed round
    per preparation is retained; missing preparations remain NaN in every tone.
    """

    signals: NDArray[np.complex128]
    shot_indices: NDArray[np.int64]
    prepared_states: NDArray[np.int64]
    tone_indices: NDArray[np.int64]

    def __post_init__(self) -> None:
        validate_ge_result(self)


def validate_ge_result(result: MuxGEResult) -> None:
    """Reject malformed source axes/dtypes or broken cross-tone shot availability.

    Require (K,2,S) complex128, K,S>0, unique original nonnegative
    int64 indices, [0,1] preparations and arange(S) shots. Missing preparation
    rows must be wholly NaN across all tones/shots; finite rows stay unfiltered.
    Raises ValueError; no cfg, fitting, mutation or hardware access occurs.
    """
    indices = result.tone_indices
    if (
        indices.ndim != 1
        or indices.dtype != np.int64
        or not indices.size
        or np.any(indices < 0)
        or len(np.unique(indices)) != len(indices)
    ):
        raise ValueError(
            "tone_indices must be unique nonnegative int64 in participant order"
        )
    signals = result.signals
    if (
        signals.dtype != np.complex128
        or signals.ndim != 3
        or signals.shape[:2] != (indices.size, 2)
        or signals.shape[2] == 0
    ):
        raise ValueError("signals must be complex128 (K,2,S), K,S>0")
    if (
        result.shot_indices.dtype != np.int64
        or not np.array_equal(result.shot_indices, np.arange(signals.shape[2]))
        or result.prepared_states.dtype != np.int64
        or not np.array_equal(result.prepared_states, [0, 1])
    ):
        raise ValueError("shot/preparation axes must preserve acquisition order")
    for preparation in range(2):
        row = signals[:, preparation, :]
        if np.isinf(row).any() or not (np.isfinite(row).all() or np.isnan(row).all()):
            raise ValueError("each preparation must be wholly finite or wholly missing")


def _raw_signal(
    prog: ModularProgramV2, _acquired: Sequence[NDArray[np.float64]]
) -> NDArray[np.complex128]:
    # The builder copies modules: use only the initialized owner from this acquire.
    for module in prog.modules:
        if isinstance(module, MuxReadout) and module.name == "readout":
            iq = module.collect_raw_iq(prog, read_index=0)
            return iq[..., 0] + 1j * iq[..., 1]
    raise ValueError("GE program is missing its named MuxReadout")


def _validate_source(source: RunRecord[MuxGECfg, MuxGEResult]) -> None:
    validate_ge_result(source.result)
    if source.cfg is not None:
        cfg = MuxGECfg.model_validate(source.cfg.model_dump())
        indices = participant_indices(
            [tone.name for tone in cfg.tones], cfg.participants
        )
        if not np.array_equal(indices, source.result.tone_indices) or cfg.shots != len(
            source.result.shot_indices
        ):
            raise ValueError("source cfg active indices/shots do not match raw result")


class MuxGEExp(PersistableExperiment[MuxGEResult, MuxGECfg]):
    """Stateless simultaneous GE acquisition; no GUI registration or proposals."""

    Options: ClassVar[type[MuxGEAnalyzeOptions]] = MuxGEAnalyzeOptions
    PostOptions: ClassVar[type[MuxGEPostOptions]] = MuxGEPostOptions

    def analyze(
        self,
        source: RunRecord[MuxGECfg, MuxGEResult],
        options: MuxGEAnalyzeOptions,
        *,
        plots: Plots,
    ) -> tuple[MuxGEAnalysis | MuxFitFailure, ...]:
        """Fit explicit raw source with GE policy and per-original-index options.

        Validate shape/axes and any source cfg before fitting. cfg=None works.
        Return indexed primary calibration/quality or numeric failure reason;
        incomplete preparation pairs never fit. Publish fit_i to caller-owned
        plots; caller finishes/releases. Invalid source/options raise. No current
        cfg, hardware, GUI, bindings or proposals are accessed.
        """
        _validate_source(source)
        return analyze_ge(source, options, plots=plots)

    def post_analyze(
        self,
        source: RunRecord[MuxGECfg, MuxGEResult],
        primary: tuple[MuxGEAnalysis | MuxFitFailure, ...],
        options: MuxGEPostOptions,
        *,
        plots: Plots,
    ) -> tuple[MuxGEPostAnalysis | MuxFitFailure, ...]:
        """Apply same-source adopted primary, producing post_i without refitting.

        source may have cfg=None; validate result and any cfg association.
        primary must be the source's ordered indexed outcomes. Return each GE
        3x3 diagnostic or retained/numeric failure. Options radii use normalized
        ADC units. Contract errors raise; caller finishes/releases plots. No
        hardware, current settings or four-state joint assignment is involved.
        """
        _validate_source(source)
        return post_analyze_ge(source, primary, options, plots=plots)

    AXES_SPEC = AxesSpec(
        axes=(
            Axis("shot_indices", "Shot Index", "", dtype=np.int64),
            Axis("prepared_states", "Probe On", "", dtype=np.int64),
            Axis("tone_indices", "Tone", "", dtype=np.int64),
        ),
        z=ZSpec("signals", "Signal", "a.u.", dtype=np.complex128),
        result_type=MuxGEResult,
        cfg_type=MuxGECfg,
        cfg_schema_version="1.0",
        tag="mux/singleshot/ge",
    )

    def run(self, config: MuxGECfg, *, context: RunContext) -> MuxGEResult:
        """Acquire raw (K,2,shots) IQ from simultaneous independent drive branches.

        Copy/validate config; lower active indices once. Warn before overriding
        rounds to 1 and reps to shots, as in single-qubit GE. In each preparation,
        common reset precedes Join of each tone's optional init and probe pulse
        (probe only in row 1), then one common mux readout. Short branches wait
        for the longest; native conversion/timing belongs to program modules.
        Read 0 from the initialized MuxReadout of that acquire, including every
        shot without per-tone sorting/filtering. No QICK raw parser or averaging.

        context owns hardware, devices, fresh StopSignal and Plots. Publish
        measurement_i (original index) magnitude traces. Completed preparations
        survive stop/error, missing rows remain NaN, no incomplete round is
        adopted. Errors stay on cancel_signal: call raise_if_error() before
        reporting success, then is_set() to distinguish cancellation. Build and
        validation errors propagate. Caller finishes/releases plots, retains
        completion metadata, pairs the original cfg with Result and closes devices.
        """
        cfg = MuxGECfg.model_validate(config.model_dump())
        if cfg.rounds != 1:
            warnings.warn(
                "rounds will be overwritten to 1 for singleshot measurement",
                stacklevel=2,
            )
        if cfg.reps != 1:
            warnings.warn(
                "reps will be overwritten by singleshot measurement shots", stacklevel=2
            )
        cfg.rounds = 1
        cfg.reps = cfg.shots
        indices = participant_indices(
            [tone.name for tone in cfg.tones], cfg.participants
        )
        readout = cfg.modules.readout.with_updates(
            tones=[cfg.modules.readout.tones[i] for i in indices]
        )
        shots = np.arange(cfg.shots, dtype=np.int64)
        viewers = [
            context.plots.liveplot_1d(
                f"measurement_{i}", "Shot Index", "Magnitude (a.u.)", num_lines=2
            )
            for i in indices
        ]

        def update(data: NDArray[np.complex128]) -> None:
            for row, viewer in enumerate(viewers):
                viewer.update(shots.astype(np.float64), np.abs(data[:, :, row]))

        buffer = SignalBuffer((2, cfg.shots, len(indices)), on_update=update)
        with Schedule(cfg, buffer, stop=context.cancel_signal) as sched:
            for with_probe, step in sched.scan("probe on", [False, True]):
                branches = [
                    [
                        Pulse(f"init_{i}", cfg.tones[i].init_pulse),
                        Pulse(f"probe_{i}", cfg.tones[i].drive if with_probe else None),
                    ]
                    for i in indices
                ]
                builder = (
                    step.prog_builder(context.soc, context.soccfg)
                    .add_reset("reset", cfg.modules.reset)
                    .add(Join(*branches), MuxReadout("readout", readout))
                )
                prog = builder.build()
                _ = builder.run_program(prog, raw2signal_fn=partial(_raw_signal, prog))
        buffer.trigger_update(flush=True)
        return MuxGEResult(
            buffer.array.transpose(2, 0, 1).copy(),
            shots,
            np.array([0, 1], dtype=np.int64),
            np.asarray(indices, dtype=np.int64),
        )
