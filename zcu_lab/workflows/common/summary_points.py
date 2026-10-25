"""Project authored point reports onto shared summary inputs, without refitting."""

from __future__ import annotations

from typing import TYPE_CHECKING

from zcu_lab.workflows.common.calibration_results import (
    FrequencyResult,
    RabiResult,
    ReadoutResult,
)
from zcu_lab.workflows.common.summary import (
    QubitLifetimes,
    StageName,
    StageOutcome,
    SummaryPoint,
)

if TYPE_CHECKING:
    from zcu_lab.workflows.coherence_fluxdep.workflow import (
        CoherencePlan,
        CoherencePoint,
    )
    from zcu_lab.workflows.mist_fluxdep.workflow import MISTPoint
    from zcu_lab.workflows.overnight.workflow import OvernightPoint
    from zcu_lab.workflows.overnight_mux.workflow import OvernightMuxPoint


def _calibration_stages(point: CoherencePoint | MISTPoint) -> tuple[StageOutcome, ...]:
    stages: tuple[
        tuple[StageName, FrequencyResult | RabiResult | ReadoutResult | None], ...
    ] = (
        ("frequency", point.frequency),
        ("Rabi", point.rabi),
        ("readout", point.readout),
    )
    outcomes = []
    for name, result in stages:
        outcomes.append(
            StageOutcome(
                stage=name,
                status=result.status if result else "skipped",
                reason=result.reason if result else "not reached",
            )
        )
        if name == "Rabi" and point.rabi is not None and point.rabi.amp is not None:
            amp = point.rabi.amp
            outcomes.append(
                StageOutcome(stage="amp Rabi", status=amp.status, reason=amp.reason)
            )
    return tuple(outcomes)


def coherence_summary_point(point: CoherencePoint, plan: CoherencePlan) -> SummaryPoint:
    """Count calibration and enabled lifetime/GE stages, marking unvisited ones skipped."""
    stages = list(_calibration_stages(point))
    if plan.measure_singleshot:
        result = point.singleshot
        stages.append(
            StageOutcome(
                stage="singleshot",
                status=(
                    "skipped"
                    if result is None
                    else "accepted"
                    if result.status == "measured"
                    else result.status
                ),
                reason=result.reason if result else "not reached",
            )
        )
    results = {"t1": point.t1, "t2ramsey": point.t2ramsey, "t2echo": point.t2echo}
    enabled_stages: tuple[tuple[str, StageName, bool], ...] = (
        ("t1", "T1", plan.measure_t1),
        ("t2ramsey", "Ramsey", plan.measure_t2ramsey),
        ("t2echo", "Echo", plan.measure_t2echo),
    )
    for kind, name, enabled in enabled_stages:
        if enabled:
            item = results.get(kind)
            stages.append(
                StageOutcome(
                    stage=name,
                    status=item.status if item else "skipped",
                    reason=item.reason if item else "not reached",
                )
            )
    return SummaryPoint(
        status=point.status,
        flux=point.flux,
        stages=tuple(stages),
        physical_recoveries=int(
            point.frequency is not None and point.frequency.recovery is not None
        ),
    )


def mist_summary_point(point: MISTPoint) -> SummaryPoint:
    """Count each requested preparation separately, sharing its stage timing."""
    stages = _calibration_stages(point) + tuple(
        StageOutcome(
            stage="MIST",
            status=trace.status,
            reason=trace.reason
            if trace.reason is not None
            else "not reached"
            if trace.status == "skipped"
            else None,
        )
        for trace in point.traces.values()
    )
    return SummaryPoint(
        status=point.status,
        flux=point.flux,
        stages=stages,
        physical_recoveries=int(
            point.frequency is not None and point.frequency.recovery is not None
        ),
    )


def overnight_summary_point(
    point: OvernightPoint, qubit_name: str | None
) -> SummaryPoint:
    """Use recorded member statuses and captured qubit identity, not reason heuristics."""
    return SummaryPoint(
        status=point.status,
        flux=point.flux,
        stages=(
            StageOutcome(
                stage="T1",
                status=point.t1_status,
                reason=point.t1_reason
                or ("not reached" if point.t1_status == "skipped" else None),
            ),
            StageOutcome(
                stage="Echo",
                status=point.t2_status,
                reason=point.t2_reason
                or ("not reached" if point.t2_status == "skipped" else None),
            ),
        ),
        qubits=(
            QubitLifetimes(name=qubit_name, t1_us=point.t1_us, echo_us=point.t2_us),
        ),
    )


def mux_summary_point(point: OvernightMuxPoint) -> SummaryPoint:
    """Count each participant's T1/Echo status and accepted values independently."""
    stages: list[StageOutcome] = []
    qubits: list[QubitLifetimes] = []
    for qubit in point.qubits:
        stages.extend(
            (
                StageOutcome(
                    stage="T1",
                    status=qubit.t1_status,
                    reason=qubit.t1_reason
                    or ("not reached" if qubit.t1_status == "skipped" else None),
                ),
                StageOutcome(
                    stage="Echo",
                    status=qubit.t2_status,
                    reason=qubit.t2_reason
                    or ("not reached" if qubit.t2_status == "skipped" else None),
                ),
            )
        )
        qubits.append(
            QubitLifetimes(name=qubit.name, t1_us=qubit.t1_us, echo_us=qubit.t2_us)
        )
    return SummaryPoint(status=point.status, stages=tuple(stages), qubits=tuple(qubits))
