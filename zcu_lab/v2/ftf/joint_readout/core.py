"""Thermal I/X preparation and same-shot, full-tone mux readout."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Annotated, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, field_serializer, model_validator
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import IDENTITY, AxesSpec, Axis, PersistableExperiment, ZSpec
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import ModularProgramV2, ProgramV2Cfg, PulseCfg
from zcu_tools.program.v2.modules import MuxReadout, MuxReadoutCfg

BRANCH_LABELS = ("00", "01", "10", "11")


class JointRoles(ConfigBase):
    """Fixed captured component names and original full-readout tone positions.

    f1/f2 name distinct fluxonium components, coupler names their transmon,
    and mux names the MuxGroup. f1_tone/f2_tone are their original zero-based
    positions in 0..tone_count-1. Form derives them from MuxGroup.qubits, never
    labels or ADCs. Core
    callers supply that same resolved mapping and tone_count, the complete
    captured group size in 1..8. Names persist with the run; no
    live alias/library lookup occurs during acquisition or offline analysis.
    """

    f1: str = Field(min_length=1)
    f2: str = Field(min_length=1)
    coupler: str = Field(min_length=1)
    mux: str = Field(min_length=1)
    f1_tone: int = Field(ge=0)
    f2_tone: int = Field(ge=0)
    tone_count: int = Field(ge=1, le=8)

    @model_validator(mode="after")
    def validate_distinct(self) -> Self:
        """Reject identical F1/F2 components or tone positions."""
        if self.f1 == self.f2 or self.f1_tone == self.f2_tone:
            raise ValueError("F1 and F2 must be distinct components and tones")
        return self


class GECenters(ConfigBase):
    """Existing GE centers in normalized raw ADC units, with no radius cut.

    Centers must be finite and distinct. Equidistant shots classify as G. This
    nearest-center partition retains every acquired shot, including thermal and
    higher-level contributions; it cannot identify leakage.
    """

    g: Annotated[complex, FieldMeta(label="Ground center")]
    e: Annotated[complex, FieldMeta(label="Excited center")]

    @field_serializer("g", "e")
    def serialize_center(self, value: complex) -> str:
        """Encode complex centers in cfg JSON without losing quadratures."""
        return str(value)

    @model_validator(mode="after")
    def validate_centers(self) -> Self:
        """Reject absent (by required fields), nonfinite or coincident centers."""
        if not np.isfinite(self.g) or not np.isfinite(self.e) or self.g == self.e:
            raise ValueError("GE centers must be finite and distinct")
        return self


class JointReadoutModules(ConfigBase):
    """Calibrated Xπ pulses on independent generators and the complete mux."""

    x180_f1: Annotated[PulseCfg, FieldMeta(label="F1 Xπ")]
    x180_f2: Annotated[PulseCfg, FieldMeta(label="F2 Xπ")]
    readout: MuxReadoutCfg


class JointReadoutCfg(ProgramV2Cfg, ExpCfgModel):
    """Four thermal preparation branches, with reps shots per software round.

    B00/B01/B10/B11 apply I/Xπ to F1/F2 then acquire one mux read. There is no
    herald, reset, post-selection or pure-state initialization. All tones are
    acquired in their original order. roles and GE centers come from the fixed
    Open/Reset capture; pulses/readout remain editable calibrated inputs.
    Delays are µs, pulse frequency MHz and phase degrees. No DC device is driven.
    Keep this original cfg with Result: only the runner copy sets rounds=1.
    """

    modules: JointReadoutModules
    roles: JointRoles
    f1_centers: GECenters
    f2_centers: GECenters

    @model_validator(mode="after")
    def validate_pair(self) -> Self:
        """Require independent charge generators and in-range original tones."""
        if self.modules.x180_f1.ch == self.modules.x180_f2.ch:
            raise ValueError("F1/F2 Xπ pulses require different charge generators")
        if self.roles.tone_count != len(self.modules.readout.tones):
            raise ValueError("Readout must retain every tone of the captured mux group")
        if max(self.roles.f1_tone, self.roles.f2_tone) >= len(
            self.modules.readout.tones
        ):
            raise ValueError("Captured F1/F2 tones exceed the complete mux readout")
        return self


@dataclass(frozen=True)
class JointReadoutResult:
    """Raw complex128 IQ (4, rounds*reps, N), all original N tones.

    branches is int64 [0,1,2,3] (F1 bit first), shot_indices is int64 in
    round-major then original program-loop/reps order, and tone_indices is int64
    [0..N-1]. Normalization is exactly collect_raw_iq's ADC units/sample, with DC
    offsets retained. Each program has only a reps loop and one read. Unwritten
    whole rounds remain complex NaN after cancellation. Acquisition rejects
    nonfinite acquired IQ; a partial sentinel never represents a measured shot.
    Consumers must check the RunContext stop signal for execution failures.
    """

    branches: NDArray[np.int64]
    shot_indices: NDArray[np.int64]
    tone_indices: NDArray[np.int64]
    signals: NDArray[np.complex128]


@dataclass(frozen=True)
class JointReadoutAnalysis:
    """Same-shot counts and row-normalized matrix, both (4,4) float64.

    Rows are nominal preparations, columns joint outcomes 00/01/10/11. Counts
    are integer-valued where complete; unanalysed rows are NaN, not measured
    zeros. acquired_shots includes retained partial raw; analyzed_shots is zero
    for incomplete branches. complete identifies analysed rows. No radius cut,
    matrix inversion, independent-error product or quality threshold is used.
    """

    counts: NDArray[np.float64]
    matrix: NDArray[np.float64]
    acquired_shots: NDArray[np.int64]
    analyzed_shots: NDArray[np.int64]
    complete: NDArray[np.bool_]


def collect_joint_shots(
    prog: ModularProgramV2, acquired: Sequence[NDArray[np.float64]]
) -> NDArray[np.complex128]:
    """Convert this initialized program's completed round, not averaged IQ.

    acquired is the Schedule conversion callback input; it is intentionally not
    interpreted. Read raw solely through the named MuxReadout's public contract.
    Return (reps,N) in unchanged loop/tone order; reject nonfinite acquired IQ.
    This has no accumulation side effect, so Schedule's final conversion simply
    overwrites the same round slot instead of appending duplicate shots.
    """
    del acquired
    for module in prog.modules:
        if isinstance(module, MuxReadout) and module.name == "readout":
            iq = module.collect_raw_iq(prog, read_index=0)
            shots = iq[..., 0] + 1j * iq[..., 1]
            if shots.shape != (prog.cfg_model.reps, len(module.cfg.tones)):
                raise ValueError(
                    "Joint readout requires only the reps loop and one mux read"
                )
            if not np.all(np.isfinite(shots)):
                raise ValueError(
                    "Acquired joint IQ must be finite; NaN is only an unwritten sentinel"
                )
            return shots
    raise ValueError("Joint program is missing its named readout MuxReadout")


def _source_shots(
    source: RunRecord[JointReadoutCfg, JointReadoutResult],
) -> tuple[JointReadoutCfg, NDArray[np.bool_]]:
    cfg, data = source.cfg, source.result
    if cfg is None:
        raise ValueError(
            "Joint analysis requires saved cfg and GE centers; current Form/library is not a substitute"
        )
    ntones, nshots = len(cfg.modules.readout.tones), cfg.reps * cfg.rounds
    if (
        not np.array_equal(data.branches, np.arange(4))
        or not np.array_equal(data.tone_indices, np.arange(ntones))
        or not np.array_equal(data.shot_indices, np.arange(nshots))
        or data.signals.shape != (4, nshots, ntones)
    ):
        raise ValueError(
            "Joint raw axes/shape must match saved cfg: (4, rounds*reps, full tones)"
        )
    rounds = data.signals.reshape(4, cfg.rounds, cfg.reps, ntones)
    acquired = np.all(np.isfinite(rounds), axis=(2, 3))
    missing = np.all(
        np.isnan(rounds) & ~np.isinf(rounds.real) & ~np.isinf(rounds.imag), axis=(2, 3)
    )
    if not np.all(acquired | missing):
        raise ValueError(
            "Joint raw contains nonfinite acquired IQ or a partially missing round"
        )
    return cfg, acquired


class JointReadoutExp(PersistableExperiment[JointReadoutResult, JointReadoutCfg]):
    """Stateless four-branch acquisition, native persistence and offline analysis."""

    AXES_SPEC = AxesSpec(
        axes=(
            Axis("tone_indices", "Tone", "", scale=IDENTITY, dtype=np.int64),
            Axis("shot_indices", "Shot", "", scale=IDENTITY, dtype=np.int64),
            Axis("branches", "Nominal preparation", "", scale=IDENTITY, dtype=np.int64),
        ),
        z=ZSpec("signals", "Raw IQ", "a.u."),
        result_type=JointReadoutResult,
        cfg_type=JointReadoutCfg,
        cfg_schema_version="1.0",
        tag="ftf/joint_readout",
    )

    def run(
        self, config: JointReadoutCfg, *, context: RunContext
    ) -> JointReadoutResult:
        """Apply thermal I/X branches and retain each completed raw round.

        Schedule owns cancellation/error propagation. Host rounds each run a
        single-round program, writing explicit (branch,round) slots on every
        completed hook. The final callback overwrites, never appends. Unacquired
        slots remain NaN. Caller checks cancel_signal.raise_if_error(); cfg is
        not mutated, saved, or repaired from hardware/library. No simulator
        truth or synthetic samples are used by this core.
        """
        cfg = config.model_copy(deep=True, update={"rounds": 1})
        ntones = len(cfg.modules.readout.tones)
        viewer = context.plots.liveplot_1d(
            "measurement",
            "Nominal preparation (00, 01, 10, 11)",
            "Acquired shots",
            num_lines=1,
        )

        def update_view(raw: NDArray[np.complex128]) -> None:
            acquired = np.all(np.isfinite(raw), axis=-1).sum(axis=(1, 2))
            viewer.update(np.arange(4, dtype=np.float64), acquired)

        buffer = SignalBuffer(
            (4, config.rounds, cfg.reps, ntones), on_update=update_view
        )
        with Schedule(cfg, buffer, stop=context.cancel_signal) as sched:
            for branch, step in sched.scan("branch", list(range(4))):
                builder = step.prog_builder(context.soc, context.soccfg)
                if branch & 2:
                    builder.add_pulse("x180_f1", cfg.modules.x180_f1)
                if branch & 1:
                    builder.add_pulse("x180_f2", cfg.modules.x180_f2)
                builder.add(MuxReadout("readout", cfg.modules.readout))
                prog = builder.build()
                for _, round_step in step.repeat("round", config.rounds):
                    _ = round_step.prog_builder(
                        context.soc, context.soccfg
                    ).run_program(
                        prog, raw2signal_fn=partial(collect_joint_shots, prog)
                    )
        buffer.trigger_update(flush=True)
        return JointReadoutResult(
            np.arange(4, dtype=np.int64),
            np.arange(config.rounds * cfg.reps, dtype=np.int64),
            np.arange(ntones, dtype=np.int64),
            buffer.array.reshape(4, config.rounds * cfg.reps, ntones),
        )

    def analyze(
        self,
        source: RunRecord[JointReadoutCfg, JointReadoutResult],
        options: None,
        *,
        plots: Plots,
    ) -> JointReadoutAnalysis:
        """Count every shot of complete branches using only saved mapping/centers.

        A complete branch requires every expected round. Partial raw survives,
        but counts/matrix rows stay NaN, with acquired/analyzed sample counts
        distinguished on the plot and numeric result. Missing cfg, malformed
        coordinates or invalid acquired data raise ValueError. Thermal content,
        preparation errors and readout effects remain inseparable in this matrix.
        """
        del options
        cfg, acquired = _source_shots(source)
        complete = np.all(acquired, axis=1)
        counts = np.full((4, 4), np.nan)
        matrix = np.full((4, 4), np.nan)
        acquired_shots = acquired.sum(axis=1).astype(np.int64) * cfg.reps
        for branch in np.flatnonzero(complete):
            shots = source.result.signals[branch]
            outcomes = np.zeros(shots.shape[0], dtype=np.int64)
            for tone, centers, weight in (
                (cfg.roles.f1_tone, cfg.f1_centers, 2),
                (cfg.roles.f2_tone, cfg.f2_centers, 1),
            ):
                values = shots[:, tone]
                outcomes += weight * (
                    np.abs(values - centers.e) < np.abs(values - centers.g)
                )
            counts[branch] = np.bincount(outcomes, minlength=4)
            matrix[branch] = counts[branch] / shots.shape[0]
        answer = JointReadoutAnalysis(
            counts,
            matrix,
            acquired_shots,
            np.where(complete, acquired_shots, 0),
            complete,
        )
        _plot_matrix(answer, plots)
        return answer


def _plot_matrix(answer: JointReadoutAnalysis, plots: Plots) -> None:
    fig, axes = plots.subplots("matrix", figsize=(9, 6))
    image = axes.imshow(
        np.ma.masked_invalid(answer.matrix), vmin=0, vmax=1, cmap="Blues"
    )
    for row in range(4):
        for col in range(4):
            label = (
                f"{answer.matrix[row, col]:.3f}\nn={int(answer.counts[row, col])}"
                if answer.complete[row]
                else "Not analysed"
            )
            axes.text(col, row, label, ha="center", va="center", fontsize=9)
    axes.set(
        xticks=range(4),
        xticklabels=BRANCH_LABELS,
        yticks=range(4),
        yticklabels=[
            f"B{label} ({int(n)} acquired)"
            for label, n in zip(BRANCH_LABELS, answer.acquired_shots, strict=True)
        ],
        xlabel="Observed joint outcome (F1, F2)",
        ylabel="Nominal I/Xπ preparation",
    )
    axes.set_title(
        "Nominal preparation → observed outcome, includes thermal population\nIncomplete branches are not analysed; no heralding or inversion"
    )
    fig.colorbar(image, ax=axes, label="Row-normalized probability")
    fig.tight_layout()
