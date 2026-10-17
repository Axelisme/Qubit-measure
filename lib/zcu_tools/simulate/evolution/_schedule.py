"""Shared opaque-payload LRU and independent chunk-local apply waves."""

from collections import Counter, OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from threading import Event

import numpy as np
from joblib import delayed
from numpy.typing import NDArray

from ._dispatch import Dispatch, Selection, classify
from ._dispatch import diagnostics as dispatch_diagnostics
from ._parallel import Executor
from ._planning import Geometry
from ._prepare_keys import group_keys
from .engine import Cancelled, EvolutionResult, PathDiagnostics
from .kernel import Kernel, Prepared, SegmentTable
from .secular import ModelSource

_CHAINABLE_KERNELS = frozenset(("dop853", "flux_dop853"))


@dataclass(frozen=True)
class _Payload:
    prepared: Prepared
    keys: tuple[bytes, ...]


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
    payload: _Payload
    rows: NDArray[np.int64]
    indices: NDArray[np.int64]


@dataclass(frozen=True)
class _ChunkResult:
    points: NDArray[np.int64]
    states: NDArray[np.complex128]
    exceeded_by_kernel: Counter[str]


class PrepareCache:
    """LRU charged once per opaque payload, scoped by source, budget and family."""

    def __init__(self, limit: int):
        self.limit = limit
        self.bytes = 0
        self.payloads: OrderedDict[int, tuple[tuple[str, str, float], _Payload]] = (
            OrderedDict()
        )
        self.next_id = 0

    def find(
        self, namespace: tuple[str, str, float], keys: set[bytes]
    ) -> list[_Payload]:
        """Return matching payloads and mark them recently used."""
        found: list[_Payload] = []
        for entry, (space, payload) in list(self.payloads.items()):
            if space == namespace and keys.intersection(payload.keys):
                found.append(payload)
                self.payloads.move_to_end(entry)
        return found

    def add(self, namespace: tuple[str, str, float], payload: _Payload) -> None:
        """Admit a bounded payload; oversized data lives only for this acquire."""
        size = payload.prepared.nbytes
        if size < 0:
            raise ValueError("prepared nbytes must be nonnegative")
        if size > self.limit or self.limit == 0:
            return
        while self.bytes + size > self.limit:
            _, (_, evicted) = self.payloads.popitem(last=False)
            self.bytes -= evicted.prepared.nbytes
        self.payloads[self.next_id] = namespace, payload
        self.next_id += 1
        self.bytes += size

    def clear(self) -> None:
        """Release all resident prepared arrays."""
        self.payloads.clear()
        self.bytes = 0


def _deduplicate(
    table: SegmentTable, dispatch: Dispatch, kernels: Mapping[str, Kernel]
) -> list[_Entries]:
    entries: list[_Entries] = []
    for name in np.unique(dispatch.names[dispatch.names != ""]):
        kernel = kernels[str(name)]
        rows = np.flatnonzero(dispatch.names == name)
        groups = group_keys(kernel, table, rows)
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
    source: ModelSource,
    budget: float,
    cancel: Event,
) -> _Payload:
    if cancel.is_set():
        raise Cancelled()
    keys = entry.keys[selection]
    prepared = entry.kernel.prepare(
        keys, table, entry.representative[selection], source, budget
    )
    return _Payload(prepared, tuple(key.tobytes() for key in keys))


def _bind(entry: _Entries, payload: _Payload, eligible: NDArray[np.bool_]) -> _Binding:
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
        entry.kernel, payload, entry.rows[chosen[present]], indices[present]
    )


class Schedule:
    """One acquire's grouped prepare stage and independent timeline chunks."""

    def __init__(
        self,
        geometry: Geometry,
        dispatch: Dispatch,
        source: ModelSource,
        budget: float,
        cancel: Event,
        initial_density: NDArray[np.complex128] | None = None,
    ):
        self.geometry, self.dispatch, self.source, self.budget, self.cancel = (
            geometry,
            dispatch,
            source,
            budget,
            cancel,
        )
        self.table = geometry.table
        self.initial_density = (
            source.initial_density if initial_density is None else initial_density
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
        jobs: list[tuple[_Entries, NDArray[np.int64], tuple[str, str, float]]] = []
        dc = self.source.model(tuple(0 for _ in range(self.table.level.shape[1])))
        for entry in entries:
            shared = entry.shared
            key_set = {key.tobytes() for key in entry.keys[shared]}
            namespace = entry.kernel.name, dc.identity, self.budget
            hits = cache.find(namespace, key_set)
            for payload in hits:
                bindings.append(_bind(entry, payload, shared[entry.inverse]))
                key_set.difference_update(payload.keys)
            available = {key.tobytes() for key in entry.keys}
            shared_keys = {key.tobytes() for key in entry.keys[shared]}
            for preparation in self.dispatch.preparations.get(entry.kernel.name, ()):
                payload = _Payload(
                    preparation.prepared,
                    tuple(key.tobytes() for key in preparation.keys),
                )
                reusable = available.intersection(payload.keys)
                if not reusable:
                    continue
                eligible = np.array(
                    [key.tobytes() in reusable for key in entry.keys], dtype=bool
                )
                bindings.append(_bind(entry, payload, eligible[entry.inverse]))
                if set(payload.keys).issubset(shared_keys):
                    cache.add(namespace, payload)
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
        # Coarse key chunks, grouped by kernel; prepare never receives density.
        built = pool(
            delayed(_build)(
                entry, selected, self.table, self.source, self.budget, self.cancel
            )
            for entry, selected, _ in jobs
        )
        for (entry, _, namespace), payload in zip(jobs, built, strict=True):
            if payload is None:
                raise RuntimeError("prepare job returned no payload")
            cache.add(namespace, payload)
            bindings.append(_bind(entry, payload, entry.shared[entry.inverse]))
        return self.bind_inputs(bindings)

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
                payload = binding.kernel.bind_input(
                    binding.payload.prepared,
                    key,
                    self.table,
                    rows,
                    initial,
                    self.budget,
                )
                if payload is not None:
                    bound.append(
                        _Binding(
                            binding.kernel,
                            _Payload(payload, ()),
                            rows,
                            np.zeros(len(rows), dtype=np.int64),
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

    def run(
        self,
        kernels: Mapping[str, Kernel],
        cache: PrepareCache,
        pool: Executor,
        chunks: int,
    ) -> EvolutionResult:
        """Return point-ordered densities and aggregate post-apply diagnostics."""
        self.kernels = kernels
        entries = _deduplicate(self.table, self.dispatch, kernels)
        shared = self.prepare_shared(entries, cache, pool, chunks)
        count = len(self.geometry.points)
        cost = np.bincount(
            self.table.timeline, weights=self.dispatch.costs, minlength=count
        )
        ordered = np.argsort(-cost, kind="stable")
        work = [ordered[i :: min(chunks, count)] for i in range(min(chunks, count))]
        results = pool(delayed(self.chunk)(points, entries, shared) for points in work)
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
        )

    def resolve_deferred(self, rows: NDArray[np.int64]) -> list[_Binding]:
        """Guard/fall back once inside a timeline and reuse accepted guard data.

        rows belong to this worker's single timeline. Classification receives
        model/table inputs and the acquire-wide cost plan, never worker density.
        Returned bindings own the tables only until this timeline finishes.
        """
        dispatch = classify(
            self.geometry,
            self.kernels,
            self.source,
            self.budget,
            selection=Selection(rows, self.dispatch.shaped, self.dispatch.rejected),
        )
        bindings: list[_Binding] = []
        for entry in _deduplicate(self.table, dispatch, self.kernels):
            for preparation in dispatch.preparations.get(entry.kernel.name, ()):
                payload = _Payload(
                    preparation.prepared,
                    tuple(key.tobytes() for key in preparation.keys),
                )
                bindings.append(
                    _bind(entry, payload, np.ones(len(entry.rows), dtype=bool))
                )
            covered = {int(row) for binding in bindings for row in binding.rows}
            missing = np.array([int(row) not in covered for row in entry.rows])
            unique = np.unique(entry.inverse[missing])
            if len(unique):
                payload = _build(
                    entry, unique, self.table, self.source, self.budget, self.cancel
                )
                bindings.append(_bind(entry, payload, missing))
        # Workers write disjoint row slices; only the caller reads all names
        # after the pool joins. Entries and bindings never read this array.
        self.dispatch.names[rows] = dispatch.names[rows]
        return bindings

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
        local_keys = np.any(deferred) or any(
            np.any(
                selected[entry.rows]
                & ~entry.shared[entry.inverse]
                & (entry.counts[entry.inverse] > 1)
            )
            for entry in entries
        )
        if len(points) > 1 and local_keys:
            results = [self.chunk(point[None], entries, shared) for point in points]
            exceeded: Counter[str] = Counter()
            for result in results:
                exceeded.update(result.exceeded_by_kernel)
            return _ChunkResult(
                points, np.concatenate([result.states for result in results]), exceeded
            )
        local_bindings = (
            self.resolve_deferred(np.flatnonzero(deferred)) if np.any(deferred) else []
        )
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
                payload = _build(
                    entry, unique, self.table, self.source, self.budget, self.cancel
                )
                local_bindings.append(_bind(entry, payload, local))
        bindings = [*shared, *self.bind_inputs(local_bindings)]
        return _Chunk(self, points, selected, bindings, entries).run()


class _Chunk:
    def __init__(
        self,
        schedule: Schedule,
        points: NDArray[np.int64],
        selected: NDArray[np.bool_],
        bindings: Sequence[_Binding],
        entries: Sequence[_Entries],
    ):
        self.schedule, self.points = schedule, points
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
        transitions = np.concatenate([self.level[slots], target], axis=1)
        changed = np.any(self.level[slots] != target, axis=1)
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
        if binding.kernel.name in _CHAINABLE_KERNELS:
            # Extend each chain until the binding, timeline or continuity changes.
            while True:
                next_row = first + lengths
                safe = np.minimum(next_row, len(self.binding) - 1)
                extend = (next_row < self.end[slots]) & (self.binding[safe] == index)
                previous = safe - 1
                extend &= (
                    self.table.start_ns[safe]
                    == self.table.start_ns[previous] + self.table.duration_ns[previous]
                )
                if not extend.any():
                    break
                lengths += extend
        offsets = np.r_[0, np.cumsum(lengths)]
        repeated = np.repeat(np.arange(len(slots)), lengths)
        rows = (
            first[repeated] + np.arange(offsets[-1]) - np.repeat(offsets[:-1], lengths)
        )
        states = self.states[slots[repeated]].copy()
        try:
            exceeded = binding.kernel.apply(
                binding.payload.prepared,
                self.key[rows],
                self.table,
                rows,
                states,
                self.schedule.budget,
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
        last = offsets[1:] - 1
        self.states[slots] = states[last]
        self.level[slots] = self.table.level[rows[last]]
        self.cursor[slots] += lengths

    def _apply_local(self, slot: int) -> None:
        """Build/apply/release one local wave, preserving direct-solver chains."""
        row = self.cursor[slot]
        index = int(self.local_entry[row])
        if index < 0:
            raise RuntimeError(f"missing preparation for segment {row}")
        entry = self.entries[index]
        rows = np.array([row], dtype=np.int64)
        if entry.kernel.name in _CHAINABLE_KERNELS:
            candidates = np.arange(row, self.end[slot], dtype=np.int64)
            contiguous = (self.local_entry[candidates] == index) & (
                self.binding[candidates] < 0
            )
            previous = candidates[:-1]
            contiguous[1:] &= self.table.start_ns[candidates[1:]] == (
                self.table.start_ns[previous] + self.table.duration_ns[previous]
            )
            stop = np.flatnonzero(~contiguous)
            rows = candidates[: int(stop[0]) if len(stop) else len(candidates)]
        unique = np.unique(self.local_key[rows])
        payload = _build(
            entry,
            unique,
            self.table,
            self.schedule.source,
            self.schedule.budget,
            self.schedule.cancel,
        )
        indices = np.searchsorted(unique, self.local_key[rows])
        binding = _Binding(entry.kernel, payload, rows, indices)
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
