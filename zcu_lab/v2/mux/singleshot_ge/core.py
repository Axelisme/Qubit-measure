"""Record simultaneous mux IQ shots with each qubit's probe pulse off and on.

For each preparation, reset, join the active init/probe branches, then read out
after the longest branch. Initial pulses run in both preparations; probe pulses
run only in the second. Acquire one round with reps=shots, preserving same-shot
IQ across tones. Probe-off/on rows are not guaranteed pure ground/excited states.

Return complex signals (K,2,shots), original tone indices, prepared_states [0,1]
and shot indices. IQ is normalized by each ADC integration length. Missing
preparations remain NaN. These two acquires are not four-state joint assignment
calibration. Primary analysis fits each tone's GE clouds; post uses that same
source's primary without refitting to derive a 3x3 g/e/other diagnostic.
Missing preparations and GE numerical failures are isolated per tone; plotting
and other errors propagate. Neither analysis approves a calibration.

Analysis options, simulation limits and source requirements:
zcu_lab/v2/mux/singleshot_ge/guide.md.
Mux raw conversion: lib/zcu_tools/program/v2/README.md.
RunContext, stop and plot ownership: lib/zcu_tools/experiment/v2/runtime/README.md.
RunRecord and save/load contracts: lib/zcu_tools/experiment/README.md.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Annotated, ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import StrictInt, model_validator
from qick.asm_v2 import QickParam
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment, RunRecord
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.v2.runtime import Schedule, SignalBuffer
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import ModularProgramV2, ResetCfg
from zcu_tools.program.v2.modules import (
    ConstWaveform,
    FlatTopWaveform,
    Join,
    MuxReadout,
    MuxReadoutCfg,
    Pulse,
)

from zcu_lab.v2.support.mux import MuxFitFailure
from zcu_lab.v2.support.mux_sweep_cfg import (
    MuxDriveToneCfg,
    require_tone_value,
    validate_participants,
)
from zcu_lab.v2.support.singleshot.plot import draw_ge_scatter

from .analysis import (
    MuxGEAnalysis,
    MuxGEAnalyzeOptions,
    MuxGEPostAnalysis,
    MuxGEPostOptions,
    analyze_ge,
    post_analyze_ge,
)


def _validate_fixed_pulse(pulse: Pulse.Cfg) -> None:
    for name in ("freq", "gain", "phase", "pre_delay", "post_delay"):
        value = getattr(pulse, name)
        if isinstance(value, QickParam) or not math.isfinite(value):
            raise ValueError(f"{name} must be fixed and finite")
    if pulse.pre_delay < 0 or pulse.post_delay < 0:
        raise ValueError("pulse delays must be nonnegative us")
    if isinstance(
        pulse.waveform, (ConstWaveform.Cfg, FlatTopWaveform.Cfg)
    ) and isinstance(pulse.waveform.length, QickParam):
        raise ValueError("pulse length must be fixed")


class MuxGEToneCfg(MuxDriveToneCfg):
    """One named original tone's required probe drive and optional init pulse.

    name is the nonempty capture-generated qubit identity. drive is probe-on;
    init_pulse, when present, runs before it in both preparations on the same
    generator. All pulses are fixed native PulseCfg (MHz, us, degrees, gain),
    validated only when selected. Nonparticipant settings do not run.
    """

    init_pulse: Pulse.Cfg | None = None

    def validate_pulses(self) -> MuxGEToneCfg:
        """Reject swept values or shared-branch generators for every element."""
        _validate_fixed_pulse(self.require_drive())
        if self.init_pulse is not None:
            _validate_fixed_pulse(self.init_pulse)
            if self.init_pulse.ch != self.require_drive().ch:
                raise ValueError("init_pulse must use the tone's drive generator")
        return self


class MuxGECfg(ConfigBase):
    """Complete named tones, ordered participants and shared acquisition controls.

    reset and readout hold the common modules. tones contains one
    MuxGEToneCfg per original readout position, with unique nonempty names.
    participants is a nonempty, duplicate-free list of original tone indices; its order
    determines execution/data rows (Q3/Q1 maps to original indices 2/0).
    Selected drives are required/validated on independent
    generators distinct from readout. shots is positive. Run sets program rounds=1
    and reps=shots; these derived runtime counts are not experiment inputs.
    initial_delay/relax_delay are finite
    nonnegative us. No capture, bindings or writeback destinations are stored.
    """

    initial_delay: Annotated[
        float,
        FieldMeta(
            label="Initial delay", unit="us", description="Delay before the first shot."
        ),
    ] = 1.0
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="us",
            decimals=3,
            description="Delay after each shot.",
        ),
    ] = 1.0
    reset: ResetCfg | None = None
    readout: MuxReadoutCfg
    tones: Annotated[list[MuxGEToneCfg], FieldMeta(element_label="name")]
    participants: list[StrictInt]
    shots: StrictInt

    @model_validator(mode="after")
    def validate_acquisition(self) -> MuxGECfg:
        """Reject invalid counts, full-list length or active channel conflicts."""
        if self.shots <= 0:
            raise ValueError("shots must be positive")
        if any(
            not math.isfinite(v) or v < 0
            for v in (self.initial_delay, self.relax_delay)
        ):
            raise ValueError(
                "initial_delay and relax_delay must be finite nonnegative us"
            )
        if len(self.tones) != len(self.readout.tones):
            raise ValueError("tones must match the complete readout tone count")
        validate_participants([tone.name for tone in self.tones], self.participants)
        channels: set[int] = set()
        for index in self.participants:
            tone = self.tones[index]
            require_tone_value(tone.drive, ("tones", index, "drive"))
            tone.validate_pulses()
            if (
                tone.require_drive().ch in channels
                or tone.require_drive().ch == self.readout.gen_ch
            ):
                raise ValueError(
                    f"tones[{index}].drive requires an independent generator"
                )
            channels.add(tone.require_drive().ch)
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
        indices = list(cfg.participants)
        if not np.array_equal(indices, source.result.tone_indices) or cfg.shots != len(
            source.result.shot_indices
        ):
            raise ValueError("source cfg active indices/shots do not match raw result")


class MuxGEExp(Experiment[MuxGEResult, MuxGECfg]):
    """Record paired mux IQ clouds and fit each participating qubit's GE separation."""

    Options: ClassVar[type[MuxGEAnalyzeOptions]] = MuxGEAnalyzeOptions
    PostOptions: ClassVar[type[MuxGEPostOptions]] = MuxGEPostOptions

    def analyze(
        self,
        source: RunRecord[MuxGECfg, MuxGEResult],
        options: MuxGEAnalyzeOptions,
        *,
        plots: Plots,
    ) -> tuple[MuxGEAnalysis | MuxFitFailure, ...]:
        """Fit each tone's probe-off/on IQ clouds from the supplied raw source.

        options selects GE fitting settings by original tone index. Return each
        tone's calibration and fit quality, or its failure reason, and draw fit_i
        in plots. Incomplete pairs do not fit. Source cfg may be absent; malformed
        axes, mismatched cfg or invalid options raise before fitting.
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
        """Estimate per-tone radius and 3x3 confusion diagnostics without refitting.

        primary contains the ordered GE fits from source; options selects radii
        in normalized ADC units. Return diagnostics or each tone's failure reason
        and draw post_i in plots. These are single-qubit diagnostics, not a joint
        assignment matrix. Source cfg may be absent; invalid source, primary or
        options raise.
        """
        _validate_source(source)
        return post_analyze_ge(source, primary, options, plots=plots)

    AXES_SPEC: ClassVar[AxesSpec[MuxGEResult, MuxGECfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis("shot_indices", "Shot Index", "", dtype=np.int64),
            AxesSpec.Axis("prepared_states", "Probe On", "", dtype=np.int64),
            AxesSpec.Axis("tone_indices", "Tone", "", dtype=np.int64),
        ),
        z=AxesSpec.Z("signals", "Signal", "a.u.", dtype=np.complex128),
        result_type=MuxGEResult,
        cfg_type=MuxGECfg,
        cfg_schema_version="3.0",
        tag="mux/singleshot/ge",
    )

    def run(self, config: MuxGECfg, *, context: RunContext) -> MuxGEResult:
        """Measure paired probe-off/on IQ shots for config's ordered participants.

        Each preparation runs common reset, simultaneous optional init pulses
        followed by probes (on only for the second preparation), then mux readout
        after the longest branch. Return normalized raw IQ shaped (K,2,shots),
        without averaging, sorting or classification. K follows participants;
        the two rows mean probe off/on, not guaranteed pure ground/excited states.
        Use one round with reps=shots, leaving config unchanged.
        context supplies hardware, stop signal and live IQ scatter plots. Missing
        preparations remain NaN; check context.scope.stop.raise_if_error()
        before reporting success. Invalid config or program builds raise.
        """
        # 1. Use one raw-shot round and preserve the requested participant order.
        cfg = MuxGECfg.model_validate(config.model_dump())

        indices = list(cfg.participants)
        readout = cfg.readout.with_updates(
            tones=[cfg.readout.tones[i] for i in indices]
        )
        shots = np.arange(cfg.shots, dtype=np.int64)

        # 2. Show both preparations without averaging away their IQ spread.
        def update(data: NDArray[np.complex128]) -> None:
            for row, index in enumerate(indices):
                draw_ge_scatter(
                    context.plots,
                    f"measurement_{index}",
                    data[:, :, row],
                    title=f"{index}: {cfg.tones[index].name}",
                )

        buffer = SignalBuffer((2, cfg.shots, len(indices)), on_update=update)
        with Schedule(cfg, buffer, scope=context.scope) as sched:
            for with_probe, step in sched.scan("probe on", [False, True]):
                # 3. Start tone branches together; read out after the longest one.
                # Init runs in both preparations, probe only in the second.
                branches = [
                    [
                        Pulse(f"init_{i}", cfg.tones[i].init_pulse),
                        Pulse(
                            f"probe_{i}",
                            cfg.tones[i].require_drive() if with_probe else None,
                        ),
                    ]
                    for i in indices
                ]
                builder = (
                    step.prog_builder(
                        context.soc,
                        context.soccfg,
                        reps=cfg.shots,
                        initial_delay=cfg.initial_delay,
                        relax_delay=cfg.relax_delay,
                    )
                    .add_reset("reset", cfg.reset)
                    .add(Join(*branches), MuxReadout("readout", readout))
                )
                prog = builder.build()
                # Read normalized shots from this program's initialized readout.
                _ = builder.run_program(
                    prog, raw2signal_fn=partial(_raw_signal, prog), rounds=1
                )

        # 4. Keep completed preparations in participant/preparation/shot order.
        buffer.trigger_update(flush=True)
        return MuxGEResult(
            buffer.array.transpose(2, 0, 1).copy(),
            shots,
            np.array([0, 1], dtype=np.int64),
            np.asarray(indices, dtype=np.int64),
        )
