"""Opt-in deterministic shadow plans; no numerical kernel is ever executed."""

from dataclasses import dataclass
from typing import Literal

from .cache import CacheEntry, CacheNamespace
from .declarations import KernelDeclaration
from .kernel import SegmentTable
from .profiles import OneCoreProfile


@dataclass(frozen=True)
class ShadowCandidate:
    """One interval option supplied by the numeric owner.

    candidate_id is globally unique and provides canonical tie ordering. row
    indexes table, not a timeline. declaration supplies eligibility and revision.
    namespace is (reuse namespace, complete model identity, budget); prepare_key
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
    max(task workspace, apply scratch). It is not RSS. feasible compares this
    with the explicit allowance. tentative lists rows without existing accepted
    guard results. cache_allocations lists whole allocation IDs charged once.
    """

    cover: tuple[ShadowCandidate, ...]
    tasks: tuple[ShadowTask, ...]
    wall_seconds: float
    lower_bound_seconds: float
    required_bytes: int
    feasible: bool
    tentative: tuple[str, ...]
    cache_allocations: tuple[int, ...]


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
    *,
    dimension: int,
    budget: float,
    workers: int,
    memory_allowance: int,
    state_bytes: int,
    limits: SearchLimits = SearchLimits(),
) -> ShadowPlan:
    """Score and improve a complete accepted incumbent without numerical work.

    table must cover the entire acquire through readout, with contiguous,
    nonoverlapping intervals per timeline. incumbent names one accepted option
    per row. The caller owns explicit guard results and complete reuse keys;
    this function only evaluates declaration eligibility. inventory is a cache
    snapshot obtained under external serialization, never find/reserve/touch.
    dimension/budget must lie in profile domain. workers is logical capacity,
    not a request to launch threads. memory_allowance/state_bytes are explicit
    live-work bytes, separate from cache budget. Invalid inputs, unsupported
    incumbent revisions, or an infeasible incumbent raise ValueError. Unsupported
    optional options are diagnosed and excluded. Search uses bounded row and
    reuse-partition moves, not timeline Cartesian products. Nothing builds a
    guard table, binds input, evolves a prefix, applies a kernel or changes Engine.
    """
    raise NotImplementedError("shadow planning is not implemented")
