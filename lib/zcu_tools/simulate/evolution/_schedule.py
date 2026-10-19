"""Shared opaque-payload LRU and independent chunk-local apply waves."""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from threading import Event

import numpy as np
from joblib import delayed
from numpy.typing import NDArray

from ._dispatch import Dispatch, Selection, classify, prepare
from ._dispatch import diagnostics as dispatch_diagnostics
from ._gain_plan import admit_shared_builds
from ._parallel import Executor
from ._planning import Geometry
from ._prepare_keys import group_keys
from ._resident import ResidentAdmission
from .cache import PrepareCache, PreparedPayload
from .engine import Cancelled, EvolutionResult, PathDiagnostics
from .kernel import ApplyContext, Kernel, KernelContext, RowPlan, SegmentTable


@dataclass(frozen=True)
class _Entries:
    kernel: Kernel
    keys: NDArray[np.void]
    representative: NDArray[np.int64]
    rows: NDArray[np.int64]
    inverse: NDArray[np.int64]
    shared: NDArray[np.bool_]
    counts: NDArray[np.int64]


@dataclass(frozen=True)
class _Binding:
    kernel: Kernel
    payload: PreparedPayload
    rows: NDArray[np.int64]
    indices: NDArray[np.int64]
    plan: RowPlan | None = None
    chain: NDArray[np.int64] | None = None


@dataclass(frozen=True)
class _ChunkResult:
    points: NDArray[np.int64]
    states: NDArray[np.complex128]
    exceeded_by_kernel: Counter[str]


def _deduplicate(
    table: SegmentTable,
    dispatch: Dispatch,
    kernels: Mapping[str, Kernel],
    context: KernelContext,
) -> list[_Entries]:
    entries: list[_Entries] = []
    for name in np.unique(dispatch.names[dispatch.names != ""]):
        kernel = kernels[str(name)]
        rows = np.flatnonzero(dispatch.names == name)
        groups = group_keys(kernel, table, rows, context)
        entries.append(
            _Entries(
                kernel,
                groups.keys,
                groups.representative,
                rows,
                groups.inverse,
                groups.shared,
                groups.counts,
            )
        )
    return entries


def _build(
    entry: _Entries,
    selection: NDArray[np.int64],
    table: SegmentTable,
    context: KernelContext,
    cancel: Event,
) -> PreparedPayload:
    if cancel.is_set():
        raise Cancelled()
    keys = entry.keys[selection]
    prepared = prepare(
        entry.kernel, keys, table, entry.representative[selection], context
    )
    return PreparedPayload(prepared, tuple(key.tobytes() for key in keys))


def _bind(
    entry: _Entries,
    payload: PreparedPayload,
    eligible: NDArray[np.bool_],
    dispatch: Dispatch,
) -> _Binding:
    positions = {key: i for i, key in enumerate(payload.keys)}
    chosen = np.flatnonzero(eligible)
    needed, inverse = np.unique(entry.inverse[chosen], return_inverse=True)
    lookup = np.array(
        [positions.get(entry.keys[index].tobytes(), -1) for index in needed],
        dtype=np.int64,
    )
    indices = lookup[inverse]
    present = indices >= 0
    return _Binding(
        entry.kernel,
        payload,
        entry.rows[chosen[present]],
        indices[present],
        dispatch.plans.get(entry.kernel.name),
        dispatch.chain,
    )


class Schedule:
    """One acquire's grouped prepare stage and independent timeline chunks."""

    def __init__(
        self,
        geometry: Geometry,
        dispatch: Dispatch,
        context: KernelContext,
        cancel: Event,
        resident: ResidentAdmission,
        initial_density: NDArray[np.complex128] | None = None,
    ):
        self.geometry, self.dispatch, self.cancel = geometry, dispatch, cancel
        self.context = context
        self.resident = resident
        self.source, self.budget = context.source, context.budget
        self.table = geometry.table
        self.initial_density = (
            context.source.initial_density
            if initial_density is None
            else initial_density
        )

    def prepare_shared(
        self, entries: list[_Entries], cache: PrepareCache, pool: Executor, chunks: int
    ) -> list[_Binding]:
        """Reuse guard payloads/LRU hits and prepare missing repeated keys.

        Dispatch belongs to this Schedule's source and budget. Guard payloads
        are eligible only for matching kernel keys selected by that dispatch.
        Timeline-local candidates defer guard preparation to their worker.
        Caller products still serve accepted subsets of shared candidates,
        even if only one timeline remains. No accepted guard product is rebuilt.
        """
        bindings: list[_Binding] = []
        local_products: list[tuple[tuple[str, str, float], PreparedPayload]] = []
        jobs: list[tuple[_Entries, NDArray[np.int64], tuple[str, str, float]]] = []
        dc = self.source.model(tuple(0 for _ in range(self.table.level.shape[1])))
        self.dc_identity = dc.identity
        for entry in entries:
            shared = entry.shared
            key_positions = {key.tobytes(): i for i, key in enumerate(entry.keys)}
            available = set(key_positions)
            key_set = {key.tobytes() for key in entry.keys[shared]}
            namespace = entry.kernel.name, dc.identity, self.budget
            for payload in cache.find(namespace, available):
                if cache.is_rejected(payload):
                    cache.add(namespace, payload)
                bindings.append(self._bind_hit(entry, payload, key_positions))
                key_set.difference_update(payload.keys)
                available.difference_update(payload.keys)
            shared_keys = {key.tobytes() for key in entry.keys[shared]}
            for preparation in self.dispatch.preparations.get(entry.kernel.name, ()):
                payload = PreparedPayload(
                    preparation.prepared,
                    tuple(key.tobytes() for key in preparation.keys),
                    array_bytes=preparation.nbytes,
                )
                reusable = available.intersection(payload.keys)
                if not reusable:
                    continue
                eligible = np.array(
                    [key.tobytes() in reusable for key in entry.keys], dtype=bool
                )
                bindings.append(
                    _bind(entry, payload, eligible[entry.inverse], self.dispatch)
                )
                if shared_keys.intersection(payload.keys):
                    self.resident.retain_shared(namespace, payload)
                else:
                    local_products.append((namespace, payload))
                key_set.difference_update(payload.keys)
                available.difference_update(payload.keys)
            missing = np.array(
                [i for i, key in enumerate(entry.keys) if key.tobytes() in key_set],
                dtype=np.int64,
            )
            if len(missing):
                jobs.extend(
                    (entry, selected, namespace)
                    for selected in np.array_split(missing, min(chunks, len(missing)))
                )
        if self.context.work_plan.gain_groups:
            self._admit_gain_builds(entries, jobs)
        # Coarse key chunks, grouped by kernel; prepare never receives density.
        built = pool(
            delayed(_build)(entry, selected, self.table, self.context, self.cancel)
            for entry, selected, _ in jobs
        )
        for (entry, _, namespace), payload in zip(jobs, built, strict=True):
            if payload is None:
                raise RuntimeError("prepare job returned no payload")
            self.resident.retain_shared(namespace, payload)
            bindings.append(
                _bind(entry, payload, entry.shared[entry.inverse], self.dispatch)
            )
        for namespace, payload in local_products:
            self.resident.offer(namespace, payload)
        self.retain_rejected(self.dispatch)
        return self.bind_inputs(bindings)

    def _bind_hit(
        self, entry: _Entries, payload: PreparedPayload, key_positions: dict[bytes, int]
    ) -> _Binding:
        """Bind a resident hit to the rows whose keys it covers."""
        eligible = np.zeros(len(entry.keys), dtype=bool)
        for key in payload.keys:
            if key in key_positions:
                eligible[key_positions[key]] = True
        binding = _bind(entry, payload, eligible[entry.inverse], self.dispatch)
        self.context.work_plan.routing.hit(entry.kernel.name, self.table, binding.rows)
        return binding

    def _admit_gain_builds(
        self,
        entries: list[_Entries],
        jobs: list[tuple[_Entries, NDArray[np.int64], tuple[str, str, float]]],
    ) -> None:
        """Admit caller-built gain rows; local and deferred rows keep groups open."""
        pending_gain_rows = [
            entry.representative[selected]
            for entry, selected, _ in jobs
            if entry.kernel.name == "floquet_period"
        ]
        open_gain_rows = [
            entry.rows[~entry.shared[entry.inverse]]
            for entry in entries
            if entry.kernel.name == "floquet_period"
        ]
        if self.dispatch.deferred is not None:
            open_gain_rows.append(np.flatnonzero(self.dispatch.deferred))
        admit_shared_builds(
            self.context,
            self.table,
            np.concatenate(pending_gain_rows)
            if pending_gain_rows
            else np.empty(0, dtype=np.int64),
            np.concatenate(open_gain_rows)
            if open_gain_rows
            else np.empty(0, dtype=np.int64),
        )

    def retain_rejected(self, dispatch: Dispatch) -> None:
        """Transfer unused handoffs to admission, releasing declined/evicted data.

        Dispatch must not remain a second owner after an offer: these products
        have no apply consumers to justify retaining them outside the byte account.
        """
        for name in tuple(dispatch.rejected_preparations):
            preparations = dispatch.rejected_preparations.pop(name)
            for preparation in preparations:
                payload = PreparedPayload(
                    preparation.prepared,
                    tuple(key.tobytes() for key in preparation.keys),
                    array_bytes=preparation.nbytes,
                )
                self.resident.offer_rejected(
                    (name, self.dc_identity, self.budget), payload
                )

    def retain_local(self, kernel: Kernel, payload: PreparedPayload) -> None:
        """Submit model-only worker products without touching the caller's LRU."""
        self.resident.offer((kernel.name, self.dc_identity, self.budget), payload)

    def bind_inputs(self, bindings: list[_Binding]) -> list[_Binding]:
        """Add evolve-local input bindings after model-only cache admission.

        Only repeated keys receive the hook. Shared and timeline-local payloads
        reach this hook separately, after their respective model-only prepare.
        All eligible rows for a key/port reach one call before chunk partitioning.
        Opaque returned data is
        referenced by the returned list alone, never by Schedule or the cache.
        """
        bound: list[_Binding] = []
        zero = tuple(0 for _ in range(self.table.level.shape[1]))
        dc = self.source.model(zero)
        prefixes = self.table.prefix_inputs
        for binding in bindings:
            for key, rows, group in self.input_groups(binding, prefixes):
                level = tuple(int(v) for v in self.table.level[rows[0]])
                initial = (
                    self.source.initial_density if group is None else prefixes[group]
                )
                if level != zero:
                    model = self.source.model(level)
                    overlap = model.eigenvectors.conj().T @ dc.eigenvectors
                    initial = overlap @ initial @ overlap.conj().T
                initial.setflags(write=False)
                with self.context.work_plan.routing.measure(
                    binding.kernel.name, "prepare"
                ):
                    payload = binding.kernel.bind_input(
                        binding.payload.prepared,
                        key,
                        self.table,
                        rows,
                        initial,
                        self.context,
                    )
                if payload is not None:
                    bound.append(
                        _Binding(
                            binding.kernel,
                            PreparedPayload(payload, (), lifetime="evolve"),
                            rows,
                            np.zeros(len(rows), dtype=np.int64),
                            binding.plan,
                            binding.chain,
                        )
                    )
        # Later bindings override model-only ones for eligible rows in _Chunk.
        return [*bindings, *bound]

    def input_groups(
        self, binding: _Binding, prefixes: Mapping[int, NDArray[np.complex128]]
    ):
        """Yield repeated-key/port/input groups; None denotes the source at start zero."""
        groups = self.table.prefix_group
        uses = np.bincount(binding.indices)
        selected: dict[tuple[int, int, int | None], list[int]] = {}
        for row, key in zip(binding.rows, binding.indices, strict=True):
            if uses[key] < 2:
                # Guard-built singletons keep their per-batch payload.
                continue
            group = groups[row] if groups is not None else None
            if self.table.start_ns[row] != 0 and group not in prefixes:
                continue
            port = (
                int(self.table.tone_port[row, 0])
                if self.table.tone_port.shape[1]
                else -1
            )
            selected.setdefault((int(key), port, group), []).append(int(row))
        for (key, _, group), rows in selected.items():
            yield key, np.array(rows, dtype=np.int64), group

    def timeline_chunks(self, workers: int, fallback: int) -> int:
        """Batch short gate inventories without changing preparation parallelism.

        Calibration used 21-depth, 20-seed IRB at integer, 0.48 and half flux:
        roughly 180 rows/timeline, half gate and half t4, with at most 1% Sambe.
        One timeline chunk/worker cut apply calls by 50--80%. Flux scans had
        about three rows/timeline and lost up to 4%, so keep six chunks there.
        Require at least 64 rows/timeline and 95% cheap matrix-action rows:
        prebuilt single-/multi-tone gates or analytic no-drive propagation.
        These families have low per-apply overhead regardless of gate duration;
        ordinary RWA and direct integration do not belong to this group.
        The 64/95% margins admit half-depth and minor Sambe tails, not short
        scans or integrator-heavy inventories. Require multiple timelines per
        worker. These thresholds select scheduling, not numerical admission.
        The full classified inventory is fixed before any worker starts; no
        resident state, prepare outcome or execution order enters this choice.
        """
        count = len(self.geometry.points)
        names = self.dispatch.names
        if count < 2 * workers or len(names) < 64 * count:
            return fallback
        cheap = np.isin(names, ("t4", "single_tone_gate", "multitone_gate"))
        if np.count_nonzero(cheap) < 0.95 * len(names):
            return fallback
        return workers

    def run(
        self,
        kernels: Mapping[str, Kernel],
        cache: PrepareCache,
        pool: Executor,
        chunks: int,
        *,
        timeline_workers: int | None = None,
    ) -> EvolutionResult:
        """Return point-ordered densities and aggregate post-apply diagnostics.

        chunks fixes prepare partitioning. timeline_workers opts into complete-
        inventory timeline batching; None retains the fixed chunks for both.
        """
        self.kernels = kernels
        entries = _deduplicate(self.table, self.dispatch, kernels, self.context)
        timeline_chunks = (
            chunks
            if timeline_workers is None
            else self.timeline_chunks(timeline_workers, chunks)
        )
        shared = self.prepare_shared(entries, cache, pool, chunks)
        count = len(self.geometry.points)
        cost = np.bincount(
            self.table.timeline, weights=self.dispatch.costs, minlength=count
        )
        ordered = np.argsort(-cost, kind="stable")
        width = min(timeline_chunks, count)
        work = [ordered[i::width] for i in range(width)]
        results = pool(delayed(self.chunk)(points, entries, shared) for points in work)
        self.resident.commit(cache)
        output = np.empty(
            (count, *self.source.initial_density.shape), dtype=np.complex128
        )
        exceeded: Counter[str] = Counter()
        for result in results:
            if result is None:
                raise RuntimeError("timeline job returned no densities")
            output[result.points] = result.states
            exceeded.update(result.exceeded_by_kernel)
        diagnostics = self.diagnostics(output, exceeded)
        return EvolutionResult(self.geometry.points, output, diagnostics)

    def diagnostics(
        self, output: NDArray[np.complex128], exceeded: Counter[str]
    ) -> PathDiagnostics:
        """Count suffix work plus actual prefix paths once per adopting timeline."""
        self.context.work_plan.routing.planned(
            self.table, self.dispatch.names, self.dispatch.costs
        )
        paths = dispatch_diagnostics(self.table, self.dispatch.names)
        segments = Counter(paths.segments_by_kernel)
        points = {
            name: set(self.table.timeline[self.dispatch.names == name].tolist())
            for name in paths.points_by_class
        }
        groups = self.table.prefix_group
        if groups is not None:
            for point, row in enumerate(self.geometry.resume_rows):
                if row < 0:
                    continue
                group = groups[row]
                if group is None:
                    raise RuntimeError("adopted prefix has no group")
                prefix = self.geometry.prefix_paths[group]
                segments.update(prefix.segments_by_kernel)
                exceeded.update(prefix.exceeded_by_kernel)
                for name in prefix.points_by_class:
                    points.setdefault(name, set()).add(point)
        return PathDiagnostics(
            segments,
            {name: len(timelines) for name, timelines in points.items()},
            exceeded,
            float(np.max(np.abs(np.trace(output, axis1=1, axis2=2) - 1))),
            routing=self.context.work_plan.routing.snapshot(segments),
        )

    def resolve_deferred(
        self, rows: NDArray[np.int64]
    ) -> tuple[list[_Binding], Dispatch]:
        """Guard/fall back once inside a timeline and reuse accepted guard data.

        rows belong to this worker's single timeline. Classification receives
        model/table inputs and the acquire-wide cost plan, never worker density.
        Returned bindings own the tables only until this timeline finishes.
        """
        dispatch = classify(
            self.geometry,
            self.kernels,
            self.context,
            selection=Selection(
                rows,
                self.dispatch.shaped,
                self.dispatch.rejected,
                self.dispatch.two_tone_direct,
                self.dispatch.names,
                self.dispatch.plans,
                self.dispatch.cached,
            ),
        )
        bindings: list[_Binding] = []
        for entry in _deduplicate(self.table, dispatch, self.kernels, self.context):
            for preparation in dispatch.preparations.get(entry.kernel.name, ()):
                payload = PreparedPayload(
                    preparation.prepared,
                    tuple(key.tobytes() for key in preparation.keys),
                    array_bytes=preparation.nbytes,
                )
                self.retain_local(entry.kernel, payload)
                bindings.append(
                    _bind(
                        entry, payload, np.ones(len(entry.rows), dtype=bool), dispatch
                    )
                )
            covered = {int(row) for binding in bindings for row in binding.rows}
            missing = np.array([int(row) not in covered for row in entry.rows])
            unique = np.unique(entry.inverse[missing])
            if len(unique):
                payload = _build(entry, unique, self.table, self.context, self.cancel)
                self.retain_local(entry.kernel, payload)
                bindings.append(_bind(entry, payload, missing, dispatch))
        self.retain_rejected(dispatch)
        # Workers write disjoint row slices; only the caller reads all names
        # after the pool joins. Entries and bindings never read this array.
        self.dispatch.names[rows] = dispatch.names[rows]
        self.dispatch.costs[rows] = dispatch.costs[rows]
        return bindings, dispatch

    def chunk(
        self,
        points: NDArray[np.int64],
        entries: Sequence[_Entries],
        shared: Sequence[_Binding],
    ):
        """Run local waves; group each next row by kernel and opaque payload."""
        if self.cancel.is_set():
            raise Cancelled()
        selected = np.isin(self.table.timeline, points)
        # Builders and repeated local keys own whole timelines. Single-use
        # local keys prepare at their apply wave without breaking shared batches.
        deferred = (
            np.zeros(len(selected), dtype=bool)
            if self.dispatch.deferred is None
            else self.dispatch.deferred & selected
        )
        owned = deferred.copy()
        for entry in entries:
            owned[entry.rows] |= (
                selected[entry.rows]
                & ~entry.shared[entry.inverse]
                & (entry.counts[entry.inverse] > 1)
            )
        local_points = np.isin(points, self.table.timeline[owned])
        if len(points) > 1 and local_points.any():
            # Stream only timelines that own retained local payloads. One such
            # timeline must not split every other point's shared apply waves.
            work = [point[None] for point in points[local_points]]
            if not local_points.all():
                work.append(points[~local_points])
            results = [self.chunk(part, entries, shared) for part in work]
            exceeded: Counter[str] = Counter()
            for result in results:
                exceeded.update(result.exceeded_by_kernel)
            return _ChunkResult(
                np.concatenate([result.points for result in results]),
                np.concatenate([result.states for result in results]),
                exceeded,
            )
        local_bindings, local_dispatch = (
            self.resolve_deferred(np.flatnonzero(deferred))
            if np.any(deferred)
            else ([], self.dispatch)
        )
        # A local guard can change the frame of an already accepted neighbor.
        # Borrow its payload but use this worker's complete timeline decisions.
        shared = [
            replace(
                binding,
                plan=local_dispatch.plans.get(binding.kernel.name),
                chain=local_dispatch.chain,
            )
            for binding in shared
        ]
        covered = np.zeros(len(selected), dtype=bool)
        for binding in (*shared, *local_bindings):
            covered[binding.rows] = True
        for entry in entries:
            local = (
                selected[entry.rows]
                & ~entry.shared[entry.inverse]
                & ~covered[entry.rows]
                & (entry.counts[entry.inverse] > 1)
            )
            unique = np.unique(entry.inverse[local])
            if len(unique):
                payload = _build(entry, unique, self.table, self.context, self.cancel)
                self.retain_local(entry.kernel, payload)
                local_bindings.append(_bind(entry, payload, local, local_dispatch))
        bindings = [*shared, *self.bind_inputs(local_bindings)]
        return _Chunk(self, points, selected, bindings, entries, local_dispatch).run()


class _Chunk:
    def __init__(
        self,
        schedule: Schedule,
        points: NDArray[np.int64],
        selected: NDArray[np.bool_],
        bindings: Sequence[_Binding],
        entries: Sequence[_Entries],
        dispatch: Dispatch,
    ):
        self.schedule, self.points = schedule, points
        self.dispatch = dispatch
        self.table = schedule.table
        self.bindings = list(bindings)
        self.entries = entries
        self.exceeded: Counter[str] = Counter()
        rows = len(self.table.timeline)
        self.binding = np.full(rows, -1, dtype=np.int64)
        self.key = np.zeros(rows, dtype=np.int64)
        self.local_entry = np.full(rows, -1, dtype=np.int64)
        self.local_key = np.zeros(rows, dtype=np.int64)
        for index, entry in enumerate(entries):
            local = selected[entry.rows] & ~entry.shared[entry.inverse]
            self.local_entry[entry.rows[local]] = index
            self.local_key[entry.rows[local]] = entry.inverse[local]
        for index, binding in enumerate(bindings):
            mask = selected[binding.rows]
            self.binding[binding.rows[mask]] = index
            self.key[binding.rows[mask]] = binding.indices[mask]
        self.cursor = np.searchsorted(self.table.timeline, points, side="left")
        self.end = np.searchsorted(self.table.timeline, points, side="right")
        self.states = np.repeat(
            schedule.initial_density[None], len(points), axis=0
        ).copy()
        zero = tuple(0 for _ in range(self.table.level.shape[1]))
        self.dc = schedule.source.model(zero)
        self.level = np.zeros((len(points), len(zero)), dtype=np.int64)
        # Prefix inputs are DC-coordinate outputs of the one representative
        # solve. Adopt the latest one, so binding, apply and free baselines all
        # see exactly the same numerical input despite reuse-dependent dispatch.
        groups, inputs = self.table.prefix_group, self.table.prefix_inputs
        if groups is not None:
            for slot, row in enumerate(schedule.geometry.resume_rows[points]):
                if row >= 0:
                    group = groups[row]
                    if group is None:
                        raise RuntimeError("adopted prefix has no group")
                    self.states[slot] = inputs[group]
                    self.cursor[slot] = row

    def _basis(self, slots: NDArray[np.int64], target: NDArray[np.int64]) -> None:
        """Convert only at apply-call boundaries, never within a DOP853 chain."""
        changed = np.any(self.level[slots] != target, axis=1)
        if not changed.any():
            return
        transitions = np.concatenate([self.level[slots], target], axis=1)
        for transition in np.unique(transitions[changed], axis=0):
            mask = np.all(transitions == transition, axis=1)
            width = target.shape[1]
            old = self.schedule.source.model(tuple(int(v) for v in transition[:width]))
            new = self.schedule.source.model(tuple(int(v) for v in transition[width:]))
            overlap = new.eigenvectors.conj().T @ old.eigenvectors
            chosen = slots[mask]
            self.states[chosen] = overlap @ self.states[chosen] @ overlap.conj().T
        self.level[slots] = target

    def _apply(self, index: int, slots: NDArray[np.int64]) -> None:
        binding = self.bindings[index]
        first = self.cursor[slots]
        self._basis(slots, self.table.level[first])
        lengths = np.ones(len(slots), dtype=np.int64)
        rows, state_slots = first, slots
        last = np.arange(len(slots))
        chain = binding.chain
        chainable = chain is not None and np.any(chain[first] >= 0)
        if chainable:
            # Extend each chain until the binding, timeline or continuity changes.
            while True:
                next_row = first + lengths
                safe = np.minimum(next_row, len(self.binding) - 1)
                extend = (next_row < self.end[slots]) & (self.binding[safe] == index)
                if chain is not None:
                    extend &= (chain[first] >= 0) & (chain[safe] == chain[first])
                if not extend.any():
                    break
                lengths += extend
            offsets = np.r_[0, np.cumsum(lengths)]
            repeated = np.repeat(np.arange(len(slots)), lengths)
            rows = (
                first[repeated]
                + np.arange(offsets[-1])
                - np.repeat(offsets[:-1], lengths)
            )
            state_slots = slots[repeated]
            last = offsets[1:] - 1
        # Integer-array indexing already owns a copy. Ordinary kernels need
        # neither chain expansion nor a second density copy.
        states = self.states[state_slots]
        try:
            with self.schedule.context.work_plan.routing.measure(
                binding.kernel.name, "apply"
            ):
                exceeded = binding.kernel.apply(
                    binding.payload.prepared,
                    self.key[rows],
                    self.table,
                    rows,
                    states,
                    ApplyContext(
                        self.schedule.budget,
                        binding.plan,
                        self.schedule.context.work_plan,
                    ),
                )
        except RuntimeError as error:
            raise RuntimeError(
                f"point {self.schedule.geometry.points[self.points[slots[0]]]} segment {rows[0]}: {error}"
            ) from error
        finite = np.isfinite(states).all(axis=(1, 2))
        if not finite.all():
            bad = int(np.flatnonzero(~finite)[0])
            raise RuntimeError(
                f"nonfinite density at point {self.schedule.geometry.points[self.table.timeline[rows[bad]]]} segment {rows[bad]}"
            )
        count = int(np.count_nonzero(exceeded))
        if count:
            self.exceeded[binding.kernel.name] += count
        self.states[slots] = states[last] if chainable else states
        self.level[slots] = self.table.level[rows[last] if chainable else rows]
        self.cursor[slots] += lengths

    def _apply_local(self, slot: int) -> None:
        """Build/apply/release one local wave, preserving direct-solver chains."""
        row = self.cursor[slot]
        index = int(self.local_entry[row])
        if index < 0:
            raise RuntimeError(f"missing preparation for segment {row}")
        entry = self.entries[index]
        rows = np.array([row], dtype=np.int64)
        chain = self.dispatch.chain
        chainable = chain[row] >= 0
        if chainable:
            candidates = np.arange(row, self.end[slot], dtype=np.int64)
            contiguous = (self.local_entry[candidates] == index) & (
                self.binding[candidates] < 0
            )
            contiguous &= chain[candidates] == chain[row]
            stop = np.flatnonzero(~contiguous)
            rows = candidates[: int(stop[0]) if len(stop) else len(candidates)]
        unique = np.unique(self.local_key[rows]) if chainable else self.local_key[rows]
        payload = _build(
            entry,
            unique,
            self.table,
            self.schedule.context,
            self.schedule.cancel,
        )
        self.schedule.retain_local(entry.kernel, payload)
        indices = (
            np.searchsorted(unique, self.local_key[rows])
            if chainable
            else np.zeros(1, dtype=np.int64)
        )
        binding = _Binding(
            entry.kernel,
            payload,
            rows,
            indices,
            self.dispatch.plans.get(entry.kernel.name),
            chain,
        )
        transient = len(self.bindings)
        self.bindings.append(binding)
        self.binding[rows], self.key[rows] = transient, indices
        try:
            self._apply(transient, np.array([slot], dtype=np.int64))
        finally:
            self.binding[rows] = -1
            self.bindings.pop()

    def run(self):
        """Propagate local timelines and convert final densities to DC basis."""
        while np.any(self.cursor < self.end):
            pending = np.flatnonzero(self.cursor < self.end)
            binding_ids = self.binding[self.cursor[pending]]
            for index in np.unique(binding_ids):
                slots = pending[binding_ids == index]
                if index < 0:
                    for slot in slots:
                        self._apply_local(int(slot))
                else:
                    self._apply(int(index), slots)
        self._basis(np.arange(len(self.points)), np.zeros_like(self.level))
        return _ChunkResult(self.points, self.states, self.exceeded)
