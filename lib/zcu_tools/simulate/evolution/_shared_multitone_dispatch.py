"""Dispatch the sampled dual-tone overlap without changing either guard."""

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ._multitone_dispatch import gate_cost
from ._sampled_rwa_dispatch import (
    gpu_sampled_groups,
    prefer_sampled_rwa,
    sampled_common_prices,
    sampled_rwa_cpu_seconds,
)
from .kernel import Kernel, KernelContext, SegmentTable
from .kernels._multitone_spot import SampledGroup, sampled_groups
from .kernels._rwa_gate import canonical_keys
from .kernels.multitone_gate import MultitoneGateKernel, gate_candidates
from .kernels.multitone_rwa import MultitoneRWAKernel

if TYPE_CHECKING:
    from ._dispatch import Classifier


@dataclass(frozen=True)
class _Offer:
    group: SampledGroup
    gate_rows: tuple[int, ...]
    rwa_first: bool
    rwa_beats_original: bool


@dataclass(frozen=True)
class _Competition:
    """Complete geometry, remaining gate inventory, and original CPU prices."""

    table: SegmentTable
    context: KernelContext
    keys: tuple[bytes, ...]
    inventory: NDArray[np.int64]
    original: NDArray[np.float64]

    def offer(self, group: SampledGroup, uses: Counter[bytes]) -> _Offer:
        """Allocate build cost to the group in proportion to each key's uses."""
        gate_rows = []
        rows = np.asarray(group.rows, dtype=np.int64)
        eligible = gate_candidates(self.table, rows)
        gate_seconds = float(self.original[np.setdiff1d(rows, eligible)].sum())
        original_seconds = float(self.original[rows].sum())
        for key, members in self.by_key(tuple(int(row) for row in eligible)).items():
            alternative = float(self.original[members].sum())
            count = uses[key]
            if not count:
                gate_seconds += alternative
                continue
            cost = gate_cost(self.table, members[0], self.context)
            apportioned = len(members) * (
                cost.prepare / count + cost.guard + cost.apply
            )
            gate_seconds += min(apportioned, alternative)
            if apportioned < alternative:
                gate_rows.extend(members)
        rwa_seconds = sampled_rwa_cpu_seconds(self.table, group, self.context)
        return _Offer(
            group,
            tuple(gate_rows),
            rwa_seconds < gate_seconds,
            rwa_seconds < original_seconds,
        )

    def by_key(self, rows: tuple[int, ...]) -> dict[bytes, list[int]]:
        selected: dict[bytes, list[int]] = defaultdict(list)
        for row in rows:
            selected[self.keys[row]].append(row)
        return selected

    def payable(self, taken: NDArray[np.int64], active_keys: set[bytes]) -> set[bytes]:
        """Recheck a full build against the original costs of remaining users."""
        remaining = tuple(int(row) for row in np.setdiff1d(self.inventory, taken))
        admitted = set()
        for key, rows in self.by_key(remaining).items():
            if key not in active_keys:
                continue
            cost = gate_cost(self.table, rows[0], self.context)
            if (
                cost.prepare + len(rows) * (cost.guard + cost.apply)
                < self.original[rows].sum()
            ):
                admitted.add(key)
        return admitted


def _original_prices(
    router: "Classifier",
    table: SegmentTable,
    context: KernelContext,
    rows: NDArray[np.int64],
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """Price original routes and return accepted and refused common rows."""
    prices = np.full(len(table.timeline), np.inf)
    if "dop853" in router.kernels:
        prices[rows] = [
            gate_cost(table, int(row), context).dop853_competitor for row in rows
        ]
    common = router.kernels.get("common_frame")
    if common is None or not len(rows):
        none = np.empty(0, dtype=np.int64)
        return prices, none, none
    offered = sampled_common_prices(table, rows, context, None)
    contenders = np.flatnonzero(np.isfinite(offered))
    candidates = rows[contenders]
    accepted = router.price_common_frame(common, table, candidates)
    admitted = candidates[accepted]
    prices[admitted] = offered[contenders[accepted]]
    return prices, admitted, candidates[~accepted]


def _direct_candidates(router: "Classifier") -> NDArray[np.int64]:
    if router.context.opencl is None:
        return np.empty(0, dtype=np.int64)
    names = router.names
    if router.selection is not None and router.selection.names is not None:
        names = np.where(names == "", router.selection.names, names)
    # Resolved non-direct neighbors and consumed prefixes break a chain.
    return (
        np.flatnonzero(
            router.geometry.pending
            & ((names == "dop853") | ((names == "") & (router.table.tones > 0)))
        )
        + router.context.complete_row_offset
    )


def _legacy_route(
    router: "Classifier",
    table: SegmentTable,
    rows: NDArray[np.int64],
    rwa: MultitoneRWAKernel,
    direct_candidates: NDArray[np.int64],
) -> None:
    """Use the established sampled RWA competition, including OpenCL waves."""
    if not len(rows):
        return
    common = router.kernels.get("common_frame")
    preferred = prefer_sampled_rwa(
        table,
        rows + router.context.complete_row_offset,
        router.context,
        rwa,
        None if common is None else partial(router.price_common_frame, common),
        direct_candidates,
    )
    direct = ("dop853",) if "dop853" in router.kernels else ()
    cost_skipped = ("common_frame", rwa.name)
    probes = router.try_kernel(rwa, rows[preferred])
    probes = router.try_common_frame(probes)
    fallback = (rwa.name, *direct) if router.context.families is not None else direct
    router.accept("multiple", probes, fallback, cost_skipped=cost_skipped)
    skipped = router.try_common_frame(rows[~preferred])
    router.accept("multiple", skipped, direct, cost_skipped=cost_skipped)


def _legacy_groups(
    router: "Classifier",
    competition: _Competition,
    offers: dict[int, _Offer],
    common: NDArray[np.int64],
    direct: NDArray[np.int64],
    rwa: MultitoneRWAKernel,
) -> set[int]:
    """Keep the whole group on the established scale if GPU can serve its original route."""
    nonoverlap = {
        index
        for index, offer in offers.items()
        if not any(np.isin(offer.group.rows, competition.inventory))
    }
    backend = competition.context.opencl
    if backend is None or "dop853" not in router.kernels:
        return nonoverlap
    gpu_groups = gpu_sampled_groups(
        competition.table,
        competition.context,
        {i: offer.group for i, offer in offers.items() if i not in nonoverlap},
        np.setdiff1d(direct, common),
    )
    if not gpu_groups:
        return nonoverlap
    rows = np.array(
        sorted(row for i in gpu_groups for row in offers[i].group.rows), dtype=np.int64
    )
    kernel = router.kernels.get("common_frame")
    legacy_prefers_rwa = prefer_sampled_rwa(
        competition.table,
        rows,
        competition.context,
        rwa,
        None if kernel is None else partial(router.price_common_frame, kernel),
        direct,
    )
    # When neither scale proposes a different family, device availability
    # cannot affect this query. Leave the backend lazy in that case.
    decisive = bool(np.any(legacy_prefers_rwa)) or any(
        offers[i].gate_rows or offers[i].rwa_beats_original for i in gpu_groups
    )
    if (
        not decisive
        or backend.candidate(
            competition.context.source.initial_density.shape[0],
            dp54=competition.context.opencl_dp54,
        )
        is not None
    ):
        return nonoverlap | gpu_groups
    return nonoverlap


def _guard_complete(
    router: "Classifier",
    competition: _Competition,
    kernel: Kernel,
    rows: NDArray[np.int64],
    scope: NDArray[np.int64],
) -> NDArray[np.int64]:
    """Resolve one proposal across its complete rows and adopt local consumers."""
    if not len(rows):
        return rows
    offset = router.context.complete_row_offset
    visible = rows[np.isin(rows, scope)] - offset
    if len(visible) == len(rows):
        router.try_kernel(kernel, visible, resolve_guard=kernel.guard_builds)
        return rows[router.names[visible] == kernel.name]
    context = competition.context
    with context.work_plan.routing.measure(kernel.name, "prepare"):
        accepted = kernel.guard(competition.table, rows, context)
    router.try_kernel(kernel, visible, resolve_guard=kernel.guard_builds)
    return rows[accepted]


def _gate_rows(
    offers: list[_Offer],
    visible: set[np.int64],
    membership: NDArray[np.int64],
    taken: NDArray[np.int64],
    competition: _Competition,
) -> NDArray[np.int64]:
    """Select still-payable gate rows in complete groups requested by this call."""
    active_keys = {competition.keys[row] for offer in offers for row in offer.gate_rows}
    payable = competition.payable(taken, active_keys)
    return np.array(
        sorted(
            row
            for offer in offers
            if membership[offer.group.rows[0]] in visible
            and not np.any(np.isin(offer.group.rows, taken))
            for row in offer.gate_rows
            if competition.keys[row] in payable
        ),
        dtype=np.int64,
    )


def _attempt_gate_groups(
    router: "Classifier",
    competition: _Competition,
    offers: list[_Offer],
    rows: NDArray[np.int64],
    rwa: MultitoneRWAKernel,
    gate: MultitoneGateKernel,
) -> Counter[bytes]:
    """Resolve planned RWA probes first, then try each still-profitable gate key."""
    context, table = competition.context, competition.table
    offset = router.context.complete_row_offset
    scope = rows + offset
    first = np.array(
        sorted(row for offer in offers if offer.rwa_first for row in offer.group.rows),
        dtype=np.int64,
    )
    if context.capture is not None:
        for offer in offers:
            if np.any(np.isin(offer.group.rows, scope)):
                context.capture.sampled_group(
                    offer.group, "not_probed" if offer.rwa_first else "cost_skipped"
                )
    taken = _guard_complete(router, competition, rwa, first, scope)
    membership = sampled_groups(table, context).by_row
    gate_rows = _gate_rows(
        offers, set(membership[scope]), membership, taken, competition
    )
    # One member's success decides whether the rest of its group may try RWA.
    # Resolve all of these keys before a prefix or worker sees a partial group.
    accepted = _guard_complete(router, competition, gate, gate_rows, scope)
    successful = set(membership[accepted])
    fallback = {
        row
        for offer in offers
        if not offer.rwa_first
        and offer.rwa_beats_original
        and membership[offer.group.rows[0]] not in successful
        for row in offer.group.rows
    }
    router.try_kernel(rwa, rows[np.isin(scope, tuple(fallback))])
    return Counter(competition.keys[row] for row in taken)


def _route_original(
    router: "Classifier", rows: NDArray[np.int64], common: NDArray[np.int64]
) -> None:
    """Adopt only admitted original routes; no numerics are retried on refusal."""
    pending = rows[(router.names[rows] == "") & ~router.deferred[rows]]
    kernel = router.kernels.get("common_frame")
    if kernel is not None:
        router.try_kernel(
            kernel,
            pending[np.isin(pending + router.context.complete_row_offset, common)],
        )
        pending = pending[router.names[pending] == ""]
    direct = ("dop853",) if "dop853" in router.kernels else ()
    # All viable alternatives were tried above. Keep exhaustion diagnostics
    # without running an expensive, already rejected candidate again.
    if len(pending) and not direct:
        attempted = router.candidate_order(
            (), ("common_frame", "multitone_rwa", "multitone_gate")
        )
        router.reject("multiple", pending, attempted)
    router.accept("multiple", pending, direct)


def split_gate_outsiders(
    router: "Classifier", rows: NDArray[np.int64]
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Delay only group-external rows whose gate key can lose users to RWA."""
    table = router.context.work_plan.complete_table
    gate = router.kernels.get("multitone_gate")
    rwa = router.kernels.get("multitone_rwa")
    if (
        not len(rows)
        or table is None
        or not isinstance(gate, MultitoneGateKernel)
        or not isinstance(rwa, MultitoneRWAKernel)
    ):
        return rows, np.empty(0, dtype=np.int64)
    membership = sampled_groups(table, router.context).by_row
    keys = canonical_keys(table, router.context)
    sampled_keys = {keys[row] for row in np.flatnonzero(membership >= 0)}
    delayed = [
        keys[row + router.context.complete_row_offset] in sampled_keys for row in rows
    ]
    mask = np.array(delayed, dtype=bool)
    return rows[~mask], rows[mask]


def route_sampled(
    router: "Classifier",
    rows: NDArray[np.int64],
    outside: NDArray[np.int64],
    rwa: MultitoneRWAKernel,
) -> Counter[bytes]:
    """Route sampled groups and return gate uses diverted by their first RWA probes."""
    table = router.context.work_plan.complete_table
    if table is None:
        raise RuntimeError("sampled RWA pricing requires complete geometry")
    direct = _direct_candidates(router)
    gate = router.kernels.get("multitone_gate")
    if not isinstance(gate, MultitoneGateKernel):
        _legacy_route(router, table, rows, rwa, direct)
        return Counter()
    offset = router.context.complete_row_offset
    context = replace(router.context, complete_row_offset=0)
    admission = sampled_groups(table, context)
    keys = canonical_keys(table, context)
    needed_keys = {keys[row + offset] for row in np.r_[rows, outside]}
    relevant = {
        i: group
        for i, group in enumerate(admission.groups)
        if any(keys[row] in needed_keys for row in group.rows)
    }
    active_keys = {keys[row] for group in relevant.values() for row in group.rows}
    inventory = gate_candidates(table, np.arange(len(table.timeline)))
    selected = inventory[
        np.array([keys[row] in active_keys for row in inventory], dtype=bool)
    ]
    quoted = np.union1d(
        selected,
        np.array(
            [row for group in relevant.values() for row in group.rows], dtype=np.int64
        ),
    )
    original, common, refused_common = _original_prices(router, table, context, quoted)
    competition = _Competition(table, context, keys, selected, original)
    counts = Counter(keys[row] for row in selected)
    offers = {i: competition.offer(group, counts) for i, group in relevant.items()}
    legacy = _legacy_groups(router, competition, offers, common, direct, rwa)
    legacy_mask = np.isin(admission.by_row[rows + offset], tuple(legacy))
    _legacy_route(router, table, rows[legacy_mask], rwa, direct)
    if legacy:
        selected = selected[~np.isin(admission.by_row[selected], tuple(legacy))]
        competition = replace(competition, inventory=selected)
        counts = Counter(keys[row] for row in selected)
        offers = {
            i: competition.offer(group, counts)
            for i, group in relevant.items()
            if i not in legacy
        }
    shared_rows = rows[~legacy_mask]
    common_kernel = router.kernels.get("common_frame")
    if common_kernel is not None:
        rejected = shared_rows[np.isin(shared_rows + offset, refused_common)]
        router.record_rejection(common_kernel, rejected)
    diverted = _attempt_gate_groups(
        router, competition, list(offers.values()), shared_rows, rwa, gate
    )
    _route_original(router, shared_rows, common)
    return diverted
