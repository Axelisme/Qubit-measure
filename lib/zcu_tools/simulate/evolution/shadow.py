"""Opt-in deterministic shadow plans; no numerical kernel is ever executed."""

from collections.abc import Iterator
from dataclasses import dataclass, field
from math import isfinite
from typing import Literal

import numpy as np

from .cache import CacheEntry, CacheNamespace
from .declarations import KernelDeclaration
from .kernel import SegmentTable
from .profiles import OneCoreProfile


@dataclass(frozen=True)
class ShadowCandidate:
    """One interval option supplied by the numeric owner.

    candidate_id is globally unique and provides canonical tie ordering. row
    indexes table, not a timeline. declaration supplies eligibility and revision.
    namespace is the cache's (kernel reuse name, complete model identity, budget);
    it must partition algorithm settings/revision/representation. prepare_key
    is the opaque complete kernel key. binding_key, if present, must include an
    already computed input certificate, port and every endpoint-affecting
    parameter. No inferred/unevolved prefix is a certificate. guard is an
    existing explicit row result, or tentative; cache proofs never change it.
    """

    candidate_id: str
    row: int
    declaration: KernelDeclaration
    namespace: CacheNamespace
    prepare_key: bytes
    guard: Literal["accepted", "rejected", "tentative"] = "tentative"
    binding_key: bytes | None = None

    def __post_init__(self) -> None:
        if (
            not self.candidate_id
            or type(self.row) is not int
            or self.row < 0
            or self.guard not in ("accepted", "rejected", "tentative")
            or type(self.prepare_key) is not bytes
            or (self.binding_key is not None and type(self.binding_key) is not bytes)
            or len(self.namespace) != 3
            or not all(self.namespace[:2])
            or not isfinite(self.namespace[2])
            or self.namespace[2] <= 0
        ):
            raise ValueError("invalid shadow candidate identity, row, key or guard")


@dataclass(frozen=True)
class SearchLimits:
    """Positive fixed bounds on input options/row, beam width, rounds and scored moves.

    max_candidates rejects oversized row input rather than silently losing the
    incumbent. max_moves includes all scored covers, including the incumbent.
    No bound depends on elapsed time, worker completion or machine load.
    """

    max_candidates: int = 16
    beam_width: int = 4
    rounds: int = 4
    max_moves: int = 256

    def __post_init__(self) -> None:
        for value in (
            self.max_candidates,
            self.beam_width,
            self.rounds,
            self.max_moves,
        ):
            if type(value) is not int or value < 1:
                raise ValueError("search limits must be positive integers")


@dataclass(frozen=True)
class ShadowRequest:
    """Fixed domain, resources and search limits for one complete acquire.

    dimension is a positive Hilbert-space size. budget is the positive finite
    numeric accuracy allowance, not a cost. workers is positive logical one-core
    capacity and launches no pool. memory_allowance limits explicitly known
    live-work bytes, or only estimated large T6 arrays when totals are unknown.
    It is separate from cache budget. state_bytes is the caller's total
    simultaneously live density/state storage for the entire acquire, counted
    once, not per row/worker. Include any retained certified prefix states here.
    limits fixes candidate/beam/round/move bounds, never elapsed time.
    """

    dimension: int
    budget: float
    workers: int
    memory_allowance: int
    state_bytes: int
    limits: SearchLimits = field(default_factory=SearchLimits)

    def __post_init__(self) -> None:
        for value in (self.dimension, self.workers):
            if type(value) is not int or value < 1:
                raise ValueError("dimension and workers must be positive integers")
        for value in (self.memory_allowance, self.state_bytes):
            if type(value) is not int or value < 0:
                raise ValueError("memory allowances must be nonnegative integer bytes")
        if not isfinite(self.budget) or self.budget <= 0:
            raise ValueError("numeric budget must be finite and positive")


@dataclass(frozen=True)
class ShadowTask:
    """One indivisible one-core task: ID, stage, predecessors and seconds.

    stage is guard/build/bind/basis/apply/lift. candidate_ids lists consumers;
    shared build/bind tasks appear once. cached build tasks cost zero seconds.
    """

    task_id: str
    stage: str
    dependencies: tuple[str, ...]
    seconds: float
    candidate_ids: tuple[str, ...]


@dataclass(frozen=True)
class ShadowScore:
    """Complete cover and graph diagnostics, never an executable production plan.

    cover contains exactly one candidate per table row in canonical time order.
    wall_seconds is deterministic nonpreemptive list scheduling on logical
    workers. lower_bound_seconds is max(work/workers, critical path, largest
    indivisible build). required_bytes is conservative live work: whole pinned
    cache allocations, all resident builds/bindings/states, plus workers times
    max(task workspace, apply scratch), or None if any charge is unknown.
    large_allocation_bytes estimates T6's dominant channel arrays only. With
    unknown totals, feasible checks only that limited estimate, not RSS or all
    workspace. With complete explicit synthetic charges it checks the total.
    tentative lists rows without existing accepted
    guard results. cache_allocations lists whole allocation IDs charged once.
    """

    cover: tuple[ShadowCandidate, ...]
    tasks: tuple[ShadowTask, ...]
    wall_seconds: float
    lower_bound_seconds: float
    required_bytes: int | None
    feasible: bool
    tentative: tuple[str, ...]
    cache_allocations: tuple[int, ...]
    large_allocation_bytes: int = 0


@dataclass(frozen=True)
class ShadowPlan:
    """Best bounded shadow score and preserved accepted safe incumbent.

    selected can contain tentative candidates; neither score authorizes execution.
    evaluated/rounds record bounded search effort; exhausted means a fixed bound
    stopped search. reasons explains exclusions/assumptions. profile_id,
    profile_revision and planner_revision identify fixed provenance. production_safe
    is always False: no plan was executed or numerically qualified by shadow.
    """

    selected: ShadowScore
    incumbent: ShadowScore
    evaluated: int
    rounds: int
    exhausted: bool
    reasons: tuple[str, ...]
    profile_id: str
    profile_revision: str
    planner_revision: str = "shadow-interval-v1"
    production_safe: bool = False


def shadow_plan(
    table: SegmentTable,
    candidates: tuple[ShadowCandidate, ...],
    incumbent: tuple[str, ...],
    profile: OneCoreProfile,
    inventory: tuple[CacheEntry, ...],
    request: ShadowRequest,
) -> ShadowPlan:
    """Score and improve a complete accepted incumbent without numerical work.

    table must cover the entire acquire through readout, with contiguous,
    nonoverlapping intervals per timeline. incumbent names one accepted option
    per row. The caller owns explicit guard results and complete reuse keys;
    this function only evaluates declaration eligibility. inventory is a cache
    snapshot obtained under external serialization, never find/reserve/touch.
    request supplies model/accuracy assumptions, logical capacity, live-work
    bytes and search bounds. Its dimension/budget must lie in profile domain.
    Invalid inputs, unsupported
    incumbent revisions, or an infeasible incumbent raise ValueError. Unsupported
    optional options are diagnosed and excluded. Search uses bounded row and
    reuse-partition moves, not timeline Cartesian products. Nothing builds a
    guard table, binds input, evolves a prefix, applies a kernel or changes Engine.
    """
    from ._shadow_graph import score_cover

    rows = _request_rows(table, profile, request)
    options, reasons = _options(
        table, candidates, profile, request.budget, request.limits
    )
    by_id = {candidate.candidate_id: candidate for candidate in options}
    if len(incumbent) != len(rows) or len(set(incumbent)) != len(incumbent):
        raise ValueError("incumbent must cover every interval exactly once")
    if any(name not in by_id for name in incumbent):
        raise ValueError("incumbent contains an unavailable candidate")
    safe = tuple(
        sorted((by_id[name] for name in incumbent), key=lambda c: rows.index(c.row))
    )
    if tuple(c.row for c in safe) != rows or any(c.guard != "accepted" for c in safe):
        raise ValueError("incumbent must be a complete explicitly accepted cover")
    initial = score_cover(table, safe, profile, inventory, request)
    if not initial.feasible:
        raise ValueError("incumbent exceeds live memory allowance")
    search = _Search(table, options, profile, inventory, request)
    best = search.run(initial)
    reasons.update(search.reasons)
    reasons.add(
        f"bounded minimum estimated wall {best.wall_seconds:g}s versus accepted "
        f"incumbent {initial.wall_seconds:g}s on {request.workers} logical workers"
    )
    reasons.add(
        "uncalibrated one-core estimates; logical workers only; not production-safe"
    )
    reasons.add(
        "known memory retains payloads through acquire; unknown totals stay unknown; "
        "only estimated T6 channel arrays are limited when totals are unknown"
    )
    if best.tentative:
        reasons.add(
            "tentative guards remain unresolved; no guard builds or prefix evolution performed"
        )
    return ShadowPlan(
        best,
        initial,
        search.evaluated,
        search.rounds,
        search.exhausted,
        tuple(sorted(reasons)),
        profile.profile_id,
        profile.revision,
    )


def _request_rows(
    table: SegmentTable, profile: OneCoreProfile, request: ShadowRequest
) -> tuple[int, ...]:
    if (
        not profile.domain.dimensions[0]
        <= request.dimension
        <= profile.domain.dimensions[1]
        or not profile.domain.budgets[0] <= request.budget <= profile.domain.budgets[1]
    ):
        raise ValueError("request is outside profile dimension/budget domain")
    rows = tuple(
        sorted(
            range(len(table.timeline)),
            key=lambda row: (
                int(table.timeline[row]),
                float(table.start_ns[row]),
                float(table.duration_ns[row]),
            ),
        )
    )
    previous: dict[int, float] = {}
    for row in rows:
        timeline = int(table.timeline[row])
        start = float(table.start_ns[row])
        duration = float(table.duration_ns[row])
        if duration <= 0 or (timeline in previous and previous[timeline] != start):
            raise ValueError(
                "shadow intervals must be positive, contiguous and nonoverlapping"
            )
        previous[timeline] = start + duration
    return rows


def _options(
    table: SegmentTable,
    candidates: tuple[ShadowCandidate, ...],
    profile: OneCoreProfile,
    budget: float,
    limits: SearchLimits,
) -> tuple[tuple[ShadowCandidate, ...], set[str]]:
    ids: set[str] = set()
    counts: dict[int, int] = {}
    available: list[ShadowCandidate] = []
    reasons: set[str] = set()
    for candidate in sorted(candidates, key=lambda c: c.candidate_id):
        if candidate.candidate_id in ids or candidate.row >= len(table.timeline):
            raise ValueError("candidate IDs must be unique and rows must exist")
        ids.add(candidate.candidate_id)
        counts[candidate.row] = counts.get(candidate.row, 0) + 1
        if counts[candidate.row] > limits.max_candidates:
            raise ValueError("row candidate count exceeds fixed search limit")
        declaration = candidate.declaration
        if candidate.namespace[2] != budget:
            raise ValueError("candidate reuse budget differs from request")
        unsupported = (
            declaration.coverage != "interval"
            or declaration.parallel != "outer-independent"
            or declaration.transition.input != "full-density"
            or declaration.transition.output != "full-density"
        )
        if unsupported or candidate.guard == "rejected":
            reasons.add(
                f"{candidate.candidate_id}: unsupported capability or rejected guard"
            )
            continue
        if not declaration.eligibility.mask(
            table, np.array([candidate.row], dtype=np.int64)
        )[0]:
            reasons.add(
                f"{candidate.candidate_id}: declaration eligibility excludes row"
            )
            continue
        try:
            profile.family(declaration.identity)
        except ValueError:
            reasons.add(f"{candidate.candidate_id}: algorithm revision outside profile")
            continue
        if (
            not declaration.guard.accepts_unknown_input
            and candidate.binding_key is None
            and candidate.guard == "accepted"
        ):
            raise ValueError(
                "accepted input-restricted candidate requires an existing certificate binding key"
            )
        available.append(candidate)
    return tuple(available), reasons


def _signature(cover: tuple[ShadowCandidate, ...]) -> tuple[str, ...]:
    return tuple(candidate.candidate_id for candidate in cover)


def _rank(score: ShadowScore) -> tuple[float, tuple[str, ...]]:
    return score.wall_seconds, _signature(score.cover)


def _moves(
    cover: tuple[ShadowCandidate, ...], options: tuple[ShadowCandidate, ...]
) -> Iterator[tuple[ShadowCandidate, ...]]:
    from ._shadow_graph import BuildKey, build_key

    partitions: dict[BuildKey, dict[int, ShadowCandidate]] = {}
    families: dict[str, dict[int, ShadowCandidate]] = {}
    for candidate in options:
        partitions.setdefault(build_key(candidate), {}).setdefault(
            candidate.row, candidate
        )
        families.setdefault(candidate.declaration.identity.name, {}).setdefault(
            candidate.row, candidate
        )
    # Whole reuse and family moves escape single-row build-amortization minima.
    for replacement in (*partitions.values(), *families.values()):
        yield tuple(replacement.get(candidate.row, candidate) for candidate in cover)
    for option in options:
        yield tuple(
            option if candidate.row == option.row else candidate for candidate in cover
        )


class _Search:
    """Own a bounded beam and graph-scoring assumptions for one shadow call."""

    def __init__(
        self,
        table: SegmentTable,
        options: tuple[ShadowCandidate, ...],
        profile: OneCoreProfile,
        inventory: tuple[CacheEntry, ...],
        request: ShadowRequest,
    ):
        self.table = table
        self.options = options
        self.profile = profile
        self.inventory = inventory
        self.request = request
        self.visited: set[tuple[str, ...]] = set()
        self.evaluated = 1
        self.rounds = 0
        self.exhausted = False
        self.reasons: set[str] = set()

    def _expand(self, current: ShadowScore) -> list[ShadowScore]:
        from ._shadow_graph import score_cover

        scores = []
        best_wall = current.wall_seconds
        for cover in _moves(current.cover, self.options):
            signature = _signature(cover)
            if signature in self.visited:
                continue
            if self.evaluated >= self.request.limits.max_moves:
                self.exhausted = True
                break
            self.visited.add(signature)
            self.evaluated += 1
            if self._work_bound(cover) > best_wall:
                self.reasons.add(
                    "dominated work/build lower bounds avoid redundant graph scoring"
                )
                continue
            score = score_cover(
                self.table, cover, self.profile, self.inventory, self.request
            )
            if score.feasible:
                scores.append(score)
                best_wall = min(best_wall, score.wall_seconds)
            else:
                self.reasons.add(
                    "memory-infeasible covers excluded by conservative live-work allowance"
                )
        return scores

    def _work_bound(self, cover: tuple[ShadowCandidate, ...]) -> float:
        """Bound remaining work without constructing another dependency graph.

        Full build and binding identities include certificates and controls.
        Accepted guard work is already sunk. Only matching revisioned whole
        cache entries can remove a build charge; unknown hits remain misses.
        This is a bound, not an alternative scheduler or assignment tree.
        """
        from ._shadow_graph import build_key

        builds = {}
        bindings = set()
        work = 0.0
        for candidate in cover:
            estimate = self.profile.family(candidate.declaration.identity)
            key = build_key(candidate)
            if key not in builds:
                hit = any(
                    entry.namespace == candidate.namespace
                    and entry.revision == candidate.declaration.identity.revision
                    and candidate.prepare_key in entry.keys
                    for entry in self.inventory
                )
                builds[key] = 0.0 if hit else float(estimate.stages.build or 0)
                work += builds[key]
            binding = (key, candidate.binding_key)
            if candidate.binding_key is not None and binding not in bindings:
                work += float(estimate.stages.bind or 0)
                bindings.add(binding)
            work += sum(
                float(value or 0)
                for value in (
                    estimate.stages.apply,
                    estimate.stages.basis_conversion,
                    estimate.stages.lift,
                )
            )
            if candidate.guard == "tentative":
                work += float(estimate.stages.guard or 0)
        return max(work / self.request.workers, max(builds.values(), default=0.0))

    def run(self, initial: ShadowScore) -> ShadowScore:
        self.visited.add(_signature(initial.cover))
        beam = [initial]
        for _ in range(self.request.limits.rounds):
            self.rounds += 1
            old_signatures = tuple(_signature(score.cover) for score in beam)
            scores = list(beam)
            for current in beam:
                scores.extend(self._expand(current))
                if self.exhausted:
                    break
            beam = sorted(scores, key=_rank)[: self.request.limits.beam_width]
            if (
                self.exhausted
                or tuple(_signature(score.cover) for score in beam) == old_signatures
            ):
                break
        else:
            self.exhausted = True
        return beam[0]
