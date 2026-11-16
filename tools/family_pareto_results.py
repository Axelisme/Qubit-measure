"""Measurement result schema and qualified timing comparisons, without execution.

The measurement runner, saved report renderer and scenario summaries share
these records. Comparison rules never construct a source or read a clock.
"""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from zcu_tools.simulate.evolution import FamilyRouting

ALL_FAMILIES = (
    "t4",
    "adiabatic_map",
    "flux_dop853",
    "slow_envelope",
    "dyson",
    "single_tone_rwa",
    "floquet_period",
    "t6",
    "single_tone_gate",
    "shaped_sambe",
    "gate_cache",
    "period_table",
    "common_frame",
    "multitone_gate",
    "multitone_rwa",
    "floquet",
    "dop853",
)


def density_error(
    actual: NDArray[np.complex128], expected: NDArray[np.complex128]
) -> float:
    """Return max(abs(actual-expected)) over the complete complex density.

    Arrays must have identical nonempty (...,M,M) shapes and finite values.
    Invalid shapes or nonfinite entries raise ValueError. No population-only
    projection or truncated density block participates in qualification.
    """
    if (
        actual.shape != expected.shape
        or actual.ndim < 2
        or not actual.size
        or actual.shape[-1] != actual.shape[-2]
    ):
        raise ValueError("densities must have identical nonempty (...,M,M) shapes")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("densities must be finite")
    return float(np.max(np.abs(actual - expected)))


@dataclass(frozen=True)
class Sample:
    """One timed evolve, inspected after the clocks stop.

    wall_seconds and cpu_seconds are elapsed wall/process CPU seconds.
    density_error is the maximum absolute complex density difference to oracle.
    None marks missing accuracy data; available measured rows cannot use it in
    comparisons. measure_family always supplies an independent reference error.
    paths maps executed family names to segment counts.
    """

    wall_seconds: float
    cpu_seconds: float
    density_error: float | None
    paths: dict[str, int]
    routing: dict[str, FamilyRouting] = field(default_factory=dict)
    """Immutable simulator reports; thread CPU includes guard refusals."""
    family_cpu_s: dict[str, float] = field(default_factory=dict)
    """Per-family prepare+apply thread CPU, not whole-process elapsed CPU."""

    @property
    def family_cpu_shares(self) -> dict[str, float]:
        """Family fractions of summed reported prepare+apply thread CPU.

        Includes rejected-guard preparation. Returns zero fractions if the sum
        is zero. These are neither wall nor process-CPU shares and exclude
        unreported routing/geometry overhead.
        """
        total = sum(report.actual_cpu_s for report in self.routing.values())
        return {
            name: report.actual_cpu_s / total if total else 0.0
            for name, report in self.routing.items()
        }


@dataclass(frozen=True)
class ModeResult:
    """One history mode's raw samples and separate timing medians.

    mode is first_evolve, warm_clear_history or warm_retain_history.
    Normal modes contain five samples. A slow family's first and retained
    modes contain one each. Its clear-history mode has samples=[], None times
    and unmeasured_reason='slow family not measured'. It has no comparison cell.
    wall_median and cpu_median are separate elapsed-second medians when measured.
    """

    mode: str
    samples: list[Sample]
    wall_median: float | None
    cpu_median: float | None
    unmeasured_reason: str | None = None


@dataclass(frozen=True)
class Refusal:
    """Public facts about only the first failing group, not later segments.

    domain names the structural domain. points and segments preserve affected
    caller IDs and row IDs. attempted preserves final-round candidate offer
    order; it is not necessarily first-attempt order.
    family_reasons classifies each whitelisted family: not_candidate means it
    is absent from attempted; guard_or_precondition means it is present.
    Unrestricted default routing has no per-family classification.
    """

    domain: str
    points: tuple[int, ...]
    segments: tuple[int, ...]
    attempted: tuple[str, ...]
    family_reasons: dict[str, Literal["not_candidate", "guard_or_precondition"]]


FamilyStatus = Literal[
    "qualified",
    "over_limit",
    "guard_or_precondition",
    "not_candidate",
    "companion_blocked",
]
ACCURACY_LIMIT = 2e-3
DIRECTLY_RETAINED = frozenset(
    {"t4", "adiabatic_map", "flux_dop853", "slow_envelope", "t6", "dop853"}
)


@dataclass(frozen=True)
class FamilyResult:
    """Pinned family or default policy; refusals have a reason and no samples.

    family is a pinned family name, or 'default' for normal routing.
    error is the maximum density error across all timed evolves for
    overall accuracy reporting/PASS-FAIL, never an assumed exact-solver zero.
    unavailable is the refusal reason, with error=None and modes=[]; otherwise
    unavailable=None and modes contains each of the three history modes once,
    in any order, unless companion_blocked describes a companion refusal.
    modes retain separate raw samples and wall/CPU medians.
    """

    family: str
    error: float | None
    unavailable: str | None
    modes: list[ModeResult]
    expected_route: dict[str, tuple[int, float]] = field(default_factory=dict)
    """Family -> expected logical segment count and total physical ns."""
    attribution: str = "target"
    """target, target_and_companions, default or fixed_family_group.

    Engine wall/process CPU describe the complete policy, not isolated target
    speed. Family thread CPU is reported separately and is not process CPU.
    """
    whitelist: tuple[str, ...] | None = None
    """Actual families restriction; None denotes unrestricted default routing."""
    refusal: Refusal | None = None
    """Structured first-group refusal facts, or None for an available policy."""
    companion_blocked: str | None = None
    """Exact refusal when only companions were attempted in the failing group.

    Such a group has no samples/error and unavailable=None. It is not evidence
    that the target is unavailable or that its numerical guard refused.
    """

    @property
    def status(self) -> FamilyStatus:
        """Classify complete-sequence accuracy or the first public refusal.

        Companion refusal takes precedence. Missing error/refusal facts raise
        ValueError or KeyError; no classification is inferred from the message.
        Default/fixed groups use attempted families rather than a target name.
        """
        if self.companion_blocked is not None:
            return "companion_blocked"
        if self.unavailable is not None:
            if self.refusal is None:
                raise ValueError(f"{self.family}: missing public refusal facts")
            if self.attribution in {"default", "fixed_family_group"}:
                return (
                    "guard_or_precondition"
                    if self.refusal.attempted
                    else "not_candidate"
                )
            return self.refusal.family_reasons[self.family]
        if self.error is None:
            raise ValueError(f"{self.family}: missing final density error")
        return "qualified" if self.error <= ACCURACY_LIMIT else "over_limit"


def _mode_time(mode: ModeResult, metric: str) -> float:
    value = mode.wall_median if metric == "wall" else mode.cpu_median
    if value is None:
        raise ValueError("measured mode requires timing")
    return value


def mode_error(mode: ModeResult) -> float:
    """Return the largest measured sample density error; missing errors raise ValueError."""
    errors = []
    for sample in mode.samples:
        if sample.density_error is None:
            raise ValueError(f"{mode.mode}: require measured density errors")
        errors.append(sample.density_error)
    if not errors:
        raise ValueError(f"{mode.mode}: require measured density errors")
    return max(errors)


def _available_modes(rows: list[FamilyResult]) -> dict[str, dict[str, ModeResult]]:
    modes = {}
    for row in rows:
        if (
            row.unavailable is not None
            or row.companion_blocked is not None
            or row.family in {"default", "fixed_default_group"}
        ):
            continue
        if row.error is None:
            raise ValueError(f"{row.family}: require measured density error")
        by_name = {mode.mode: mode for mode in row.modes}
        if len(row.modes) != 3 or set(by_name) != {
            "first_evolve",
            "warm_clear_history",
            "warm_retain_history",
        }:
            raise ValueError(f"{row.family}: require all three unique history modes")
        first = by_name["first_evolve"]
        measured = {}
        for mode in row.modes:
            if mode.unmeasured_reason is not None:
                if not (
                    mode.mode == "warm_clear_history"
                    and mode.unmeasured_reason == "slow family not measured"
                    and not mode.samples
                    and mode.wall_median is None
                    and mode.cpu_median is None
                    and len(first.samples) == 1
                    and first.samples[0].wall_seconds > 60
                    and len(by_name["warm_retain_history"].samples) == 1
                ):
                    raise ValueError(f"{row.family}: invalid unmeasured history mode")
                continue
            mode_error(mode)
            if mode.wall_median is None or mode.cpu_median is None:
                raise ValueError(f"{row.family}: measured mode requires timing")
            measured[mode.mode] = mode
        modes[row.family] = measured
    return modes


def dominance(rows: list[FamilyResult]) -> dict[str, dict[str, list[str]]]:
    """List dominators among qualified, non-retained pinned families.

    Both policies must have complete-sequence FamilyResult.error <= 2e-3.
    Qualified policies are equally accurate; only time < .9*target time
    dominates. Exactly ten percent is equivalent timing. DIRECTLY_RETAINED
    families, default/fixed groups and refused policies are not competitors.
    All three unique modes are required in any order. Missing, duplicate and
    unknown modes raise. A slow family's explicitly unmeasured clear-history
    mode is excluded. Other empty modes, missing errors and timing/path-only
    rows raise ValueError. Comparisons use only shared measured modes.
    """
    modes = _available_modes(rows)
    available = [
        r
        for r in rows
        if r.family in modes
        and r.family not in DIRECTLY_RETAINED
        and r.status == "qualified"
    ]
    result: dict[str, dict[str, list[str]]] = {}
    for target in available:
        cells = {}
        for mode in modes[target.family].values():
            for metric in ("wall", "cpu"):
                target_time = _mode_time(mode, metric)
                cells[f"{mode.mode}/{metric}"] = [
                    other.family
                    for other in available
                    if other.family != target.family
                    and mode.mode in modes[other.family]
                    and _mode_time(modes[other.family][mode.mode], metric)
                    < 0.9 * target_time
                ]
        result[target.family] = cells
    return result


@dataclass(frozen=True)
class InputResult:
    """Complete input row, with independent wall/CPU and history-mode cells.

    name is the stable manifest input name. physical_sha256 hashes the complete
    input; oracle_sha256 excludes nonphysical identity for shared references.
    oracle_convergence_error is the maximum density difference between the two
    oracle tolerance levels. policies contains pinned results and default.
    dominators maps pinned family -> 'mode/wall' or 'mode/cpu' -> dominating
    family names. timing_equivalents has the same keys and time-equivalent peers.
    Unavailable policies have no comparison cells; default is not a competitor.
    """

    name: str
    physical_sha256: str
    oracle_sha256: str
    oracle_convergence_error: float
    policies: list[FamilyResult]
    dominators: dict[str, dict[str, list[str]]]
    timing_equivalents: dict[str, dict[str, list[str]]]
    category: str = "synthetic"
    """real or synthetic; deletion conclusions use only real workloads."""
    oracle_wall_seconds: float = 0.0
    """Elapsed solve/cache-load wall seconds, zero for an in-memory reference hit."""
    oracle_cpu_seconds: float = 0.0
    """Process CPU seconds spent in that solve/cache load, excluding setup."""
    fixed_default_group: FamilyResult | None = None
    """Separately timed whitelist of the default route's observed family union."""

    @property
    def dominance_same_companions(self) -> dict[str, dict[str, dict[str, bool]]]:
        """Mark each target/cell/dominator relation by equal companion sets.

        Cell names are mode/wall or mode/cpu, as in dominators. False means
        group-level observation only, not independent evidence for deletion.
        Ordering of a whitelist does not affect companion equality. A missing
        whitelist denotes a single-family policy for legacy synthetic results.
        """
        companions = {
            policy.family: set(policy.whitelist or ()) - {policy.family}
            for policy in self.policies
        }
        return {
            target: {
                cell: {peer: companions[target] == companions[peer] for peer in peers}
                for cell, peers in cells.items()
            }
            for target, cells in self.dominators.items()
        }


def timing_equivalents(rows: list[FamilyResult]) -> dict[str, dict[str, list[str]]]:
    """List time-equivalent peers when neither is >10% faster, ignoring error.

    This is symmetric: other_time >= .9*target_time and target_time >=
    .9*other_time. It is not Pareto dominance and never merges wall and CPU.
    All three unique history modes are required in any order. A slow family's
    explicitly unmeasured clear-history mode is excluded. Other empty modes,
    missing/duplicate/unknown modes, missing errors and timing/path-only rows
    raise ValueError. Unavailable and default rows are omitted.
    """
    modes = _available_modes(rows)
    available = [r for r in rows if r.family in modes]
    equivalents: dict[str, dict[str, list[str]]] = {}
    for target in available:
        cells = {}
        for mode in modes[target.family].values():
            for metric in ("wall", "cpu"):
                target_time = _mode_time(mode, metric)
                peers = []
                for other in available:
                    if (
                        other.family == target.family
                        or mode.mode not in modes[other.family]
                    ):
                        continue
                    other_time = _mode_time(modes[other.family][mode.mode], metric)
                    if (
                        other_time >= 0.9 * target_time
                        and target_time >= 0.9 * other_time
                    ):
                        peers.append(other.family)
                cells[f"{mode.mode}/{metric}"] = peers
        equivalents[target.family] = cells
    return equivalents
