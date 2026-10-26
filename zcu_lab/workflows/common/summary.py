"""Mechanical run totals from point outcomes and delivered operation facts."""

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from statistics import fmean, pstdev
from typing import Literal

from pydantic import BaseModel, Field
from zcu_tools.experiment.operations import Completed, Failed, OperationFacts

from zcu_lab.workflows.common.calibration_results import StageStatus

type StageName = Literal[
    "frequency",
    "Rabi",
    "amp Rabi",
    "readout",
    "singleshot",
    "T1",
    "Ramsey",
    "Echo",
    "MIST",
]


class DurationSummary(BaseModel):
    """Mean and maximum seconds of delivered operations per observed point/stage."""

    mean_s: float
    max_s: float


class StageSummary(BaseModel):
    """Outcome counts, exact rejection/skip/failure reasons and operation timing.

    Counts describe point-stage outcomes (per participant for mux lifetimes).
    Reasons preserve the original status and message. Timing sums same-stage operations within each point
    before taking mean/max, without multiplying mux time by participant count.
    """

    stage: StageName
    accepted: int = 0
    rejected: int = 0
    skipped: int = 0
    failed: int = 0
    rejected_reasons: dict[str, int] = Field(default_factory=dict)
    skipped_reasons: dict[str, int] = Field(default_factory=dict)
    failed_reasons: dict[str, int] = Field(default_factory=dict)
    duration: DurationSummary | None = Field(
        default=None, exclude_if=lambda v: v is None
    )


class FluxRange(BaseModel):
    """Minimum/maximum committed setpoints in the plan's explicit flux unit."""

    minimum: float
    maximum: float
    unit: str


class LifetimeSummary(BaseModel):
    """Accepted count and mean/population standard deviation in microseconds.

    Empty samples omit mean/std; one accepted sample has standard deviation zero.
    """

    accepted: int
    mean_us: float | None = Field(default=None, exclude_if=lambda v: v is None)
    std_us: float | None = Field(default=None, exclude_if=lambda v: v is None)


class QubitSummary(BaseModel):
    """Captured qubit name and accepted T1/Echo statistics, without refitting.

    Single-qubit overnight uses None if the snapshot has no single qubit role;
    the report never guesses an identity from a drive channel.
    """

    name: str | None
    t1: LifetimeSummary
    echo: LifetimeSummary


class RunSummary(BaseModel):
    """Normal-return report; cancelled/raised runs do not return this model.

    Completed points have status=completed, including rejected/failed stages.
    Committed points also include skipped points. Recovery counts attempted
    physical refits. Stage durations use delivered operation facts only.
    Mux has no flux range because it does not set flux.
    """

    planned_points: int
    committed_points: int
    completed_points: int
    skipped_points: int
    flux_range: FluxRange | None = Field(default=None, exclude_if=lambda v: v is None)
    physical_recoveries: int
    stages: tuple[StageSummary, ...]
    qubits: tuple[QubitSummary, ...] = ()


class StageOutcome(BaseModel):
    """One recorded stage status and its exact diagnostic, with a canonical name."""

    stage: StageName
    status: StageStatus
    reason: str | None = None


class QubitLifetimes(BaseModel):
    """Accepted lifetime values and captured identity; unknown name/values stay None."""

    name: str | None
    t1_us: float | None
    echo_us: float | None


class SummaryPoint(BaseModel):
    """Point payload projected onto summary inputs; no raw data or fit decisions."""

    status: Literal["collecting", "completed", "skipped"]
    flux: float | None = None
    stages: tuple[StageOutcome, ...]
    physical_recoveries: int = 0
    qubits: tuple[QubitLifetimes, ...] = ()


_OPERATION_STAGES: dict[str, StageName] = {
    "frequency": "frequency",
    "Rabi": "Rabi",
    "AmpRabi": "amp Rabi",
    "readout": "readout",
    "singleshot GE": "singleshot",
    "T1": "T1",
    "Ramsey": "Ramsey",
    "Echo": "Echo",
    "Mux T1": "T1",
    "Mux Echo": "Echo",
    "MIST excited": "MIST",
    "MIST ground": "MIST",
}


def _stage_totals(points: Iterable[SummaryPoint]) -> dict[StageName, StageSummary]:
    totals: dict[StageName, StageSummary] = {}
    for point in points:
        for outcome in point.stages:
            total = totals.setdefault(outcome.stage, StageSummary(stage=outcome.stage))
            match outcome.status:
                case "accepted":
                    total.accepted += 1
                case "rejected":
                    total.rejected += 1
                case "skipped":
                    total.skipped += 1
                case "failed":
                    total.failed += 1
            if outcome.reason is not None and outcome.status != "accepted":
                reasons = {
                    "rejected": total.rejected_reasons,
                    "skipped": total.skipped_reasons,
                    "failed": total.failed_reasons,
                }[outcome.status]
                reasons[outcome.reason] = reasons.get(outcome.reason, 0) + 1
    return totals


def _add_durations(
    totals: dict[StageName, StageSummary], operations: Iterable[OperationFacts]
) -> None:
    per_point: dict[tuple[int, StageName], float] = defaultdict(float)
    for facts in operations:
        stage = _OPERATION_STAGES.get(facts.name or "")
        if stage is not None:
            per_point[facts.iteration, stage] += (
                facts.finished_at - facts.started_at
            ).total_seconds()
    durations: dict[StageName, list[float]] = defaultdict(list)
    for (_, stage), seconds in per_point.items():
        durations[stage].append(seconds)
    for stage, seconds in durations.items():
        total = totals.setdefault(stage, StageSummary(stage=stage))
        total.duration = DurationSummary(mean_s=fmean(seconds), max_s=max(seconds))


def _lifetime_summary(values: list[float]) -> LifetimeSummary:
    if not values:
        return LifetimeSummary(accepted=0)
    return LifetimeSummary(
        accepted=len(values), mean_us=fmean(values), std_us=pstdev(values)
    )


def _qubit_totals(points: Iterable[SummaryPoint]) -> tuple[QubitSummary, ...]:
    values: dict[str | None, tuple[list[float], list[float]]] = {}
    for point in points:
        for qubit in point.qubits:
            t1, echo = values.setdefault(qubit.name, ([], []))
            if qubit.t1_us is not None:
                t1.append(qubit.t1_us)
            if qubit.echo_us is not None:
                echo.append(qubit.echo_us)
    return tuple(
        QubitSummary(name=name, t1=_lifetime_summary(t1), echo=_lifetime_summary(echo))
        for name, (t1, echo) in values.items()
    )


@dataclass
class SummaryAccumulator:
    """Keep run-local point reports and delivered facts; never read saved files.

    planned_points is the fixed plan size; flux_unit is None for a mux run.
    Append one SummaryPoint after each commit. Helpers call observe immediately
    after an operation returns; Failed has no facts and contributes no timing.
    finish returns mechanical totals without changing scientific acceptance.
    """

    planned_points: int
    flux_unit: str | None = None
    points: list[SummaryPoint] = field(default_factory=list)
    operations: list[OperationFacts] = field(default_factory=list)

    def observe[T](self, outcome: Completed[T] | Failed) -> None:
        """Retain only delivered Completed facts, including partial results."""
        if isinstance(outcome, Completed):
            self.operations.append(outcome.facts)

    def finish(self) -> RunSummary:
        """Return point totals, stage durations and population statistics."""
        stages = _stage_totals(self.points)
        _add_durations(stages, self.operations)
        fluxes = [point.flux for point in self.points if point.flux is not None]
        flux_range = None
        if fluxes and self.flux_unit is not None:
            flux_range = FluxRange(
                minimum=min(fluxes), maximum=max(fluxes), unit=self.flux_unit
            )
        return RunSummary(
            planned_points=self.planned_points,
            committed_points=len(self.points),
            completed_points=sum(p.status == "completed" for p in self.points),
            skipped_points=sum(p.status == "skipped" for p in self.points),
            flux_range=flux_range,
            physical_recoveries=sum(p.physical_recoveries for p in self.points),
            stages=tuple(stages.values()),
            qubits=_qubit_totals(self.points),
        )


def observe_operation[T](
    summary: SummaryAccumulator | None, outcome: Completed[T] | Failed
) -> None:
    """Pass delivered facts to an optional run accumulator without altering the outcome."""
    if summary is not None:
        summary.observe(outcome)
