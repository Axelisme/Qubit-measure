"""Shared opaque-payload LRU and independent chunk-local apply waves."""

from collections import Counter, OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from threading import Event

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import NDArray

from ._dispatch import Dispatch
from ._planning import Geometry
from .engine import Cancelled, EvolutionResult
from .kernel import Kernel, Prepared, SegmentTable
from .secular import ModelSource


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
    for name in np.unique(dispatch.names):
        kernel = kernels[str(name)]
        rows = np.flatnonzero(dispatch.names == name)
        keys, first, inverse, counts = np.unique(
            kernel.prepare_keys(table, rows),
            return_index=True,
            return_inverse=True,
            return_counts=True,
        )
        entries.append(_Entries(kernel, keys, rows[first], rows, inverse, counts))
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
    lookup = np.array(
        [positions.get(key.tobytes(), -1) for key in entry.keys], dtype=np.int64
    )
    indices = lookup[entry.inverse]
    select = eligible & (indices >= 0)
    return _Binding(entry.kernel, payload, entry.rows[select], indices[select])


class Schedule:
    """One acquire's grouped prepare stage and independent timeline chunks."""

    def __init__(
        self,
        geometry: Geometry,
        dispatch: Dispatch,
        source: ModelSource,
        budget: float,
        cancel: Event,
    ):
        self.geometry, self.dispatch, self.source, self.budget, self.cancel = (
            geometry,
            dispatch,
            source,
            budget,
            cancel,
        )
        self.table = geometry.table

    def prepare_shared(
        self, entries: list[_Entries], cache: PrepareCache, pool: Parallel, chunks: int
    ) -> list[_Binding]:
        """Reuse guard payloads/LRU hits and prepare missing repeated keys.

        Dispatch belongs to this Schedule's source and budget. Guard payloads
        are eligible only for matching kernel keys selected by that dispatch.
        Singleton guard payloads bypass the LRU but reach timeline chunks too.
        """
        bindings: list[_Binding] = []
        jobs: list[tuple[_Entries, NDArray[np.int64], tuple[str, str, float]]] = []
        dc = self.source.model(tuple(0 for _ in range(self.table.level.shape[1])))
        for entry in entries:
            shared = entry.counts >= 2
            key_set = {key.tobytes() for key in entry.keys[shared]}
            namespace = entry.kernel.name, dc.identity, self.budget
            hits = cache.find(namespace, key_set)
            for payload in hits:
                bindings.append(_bind(entry, payload, shared[entry.inverse]))
                key_set.difference_update(payload.keys)
            available = key_set | {key.tobytes() for key in entry.keys[~shared]}
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
            bindings.append(_bind(entry, payload, entry.counts[entry.inverse] >= 2))
        return self.bind_inputs(bindings)

    def bind_inputs(self, bindings: list[_Binding]) -> list[_Binding]:
        """Add evolve-local input bindings after model-only cache admission.

        Only repeated keys receive the hook; guard-built singletons bypass it.
        All eligible rows for a key/port reach one call before chunk partitioning.
        Opaque returned data is
        referenced by the returned list alone, never by Schedule or the cache.
        """
        bound: list[_Binding] = []
        zero = tuple(0 for _ in range(self.table.level.shape[1]))
        dc = self.source.model(zero)
        for binding in bindings:
            eligible = self.table.start_ns[binding.rows] == 0
            for key in np.unique(binding.indices[eligible]):
                if np.count_nonzero(binding.indices == key) < 2:
                    continue
                candidates = binding.rows[eligible & (binding.indices == key)]
                ports = (
                    self.table.tone_port[candidates, 0]
                    if self.table.tone_port.shape[1]
                    else np.full(len(candidates), -1, dtype=np.int64)
                )
                for port in np.unique(ports):
                    rows = candidates[ports == port]
                    level = tuple(int(v) for v in self.table.level[rows[0]])
                    initial = self.source.initial_density
                    if level != zero:
                        model = self.source.model(level)
                        overlap = model.eigenvectors.conj().T @ dc.eigenvectors
                        initial = overlap @ initial @ overlap.conj().T
                    initial.setflags(write=False)
                    payload = binding.kernel.bind_input(
                        binding.payload.prepared,
                        int(key),
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

    def run(
        self,
        kernels: Mapping[str, Kernel],
        cache: PrepareCache,
        pool: Parallel,
        chunks: int,
    ) -> EvolutionResult:
        """Return point-ordered densities and aggregate post-apply diagnostics."""
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
        diagnostics = replace(self.dispatch.diagnostics, exceeded_by_kernel=exceeded)
        return EvolutionResult(self.geometry.points, output, diagnostics)

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
        bindings = list(shared)
        covered = np.zeros(len(selected), dtype=bool)
        for binding in shared:
            covered[binding.rows] = True
        for entry in entries:
            local = (
                selected[entry.rows]
                & (entry.counts[entry.inverse] == 1)
                & ~covered[entry.rows]
            )
            unique = np.unique(entry.inverse[local])
            if len(unique):
                payload = _build(
                    entry, unique, self.table, self.source, self.budget, self.cancel
                )
                bindings.append(_bind(entry, payload, local))
        return _Chunk(self, points, selected, bindings).run()


class _Chunk:
    def __init__(
        self,
        schedule: Schedule,
        points: NDArray[np.int64],
        selected: NDArray[np.bool_],
        bindings: Sequence[_Binding],
    ):
        self.schedule, self.points = schedule, points
        self.table = schedule.table
        self.bindings = bindings
        self.exceeded: Counter[str] = Counter()
        rows = len(self.table.timeline)
        self.binding = np.full(rows, -1, dtype=np.int64)
        self.key = np.zeros(rows, dtype=np.int64)
        for index, binding in enumerate(bindings):
            mask = selected[binding.rows]
            self.binding[binding.rows[mask]] = index
            self.key[binding.rows[mask]] = binding.indices[mask]
        self.cursor = np.searchsorted(self.table.timeline, points, side="left")
        self.end = np.searchsorted(self.table.timeline, points, side="right")
        self.states = np.repeat(
            schedule.source.initial_density[None], len(points), axis=0
        ).copy()
        zero = tuple(0 for _ in range(self.table.level.shape[1]))
        self.dc = schedule.source.model(zero)
        self.level = np.zeros((len(points), len(zero)), dtype=np.int64)

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
        if binding.kernel.name in ("dop853", "flux_dop853"):
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

    def run(self):
        """Propagate local timelines and convert final densities to DC basis."""
        while np.any(self.cursor < self.end):
            pending = np.flatnonzero(self.cursor < self.end)
            binding_ids = self.binding[self.cursor[pending]]
            for index in np.unique(binding_ids):
                self._apply(int(index), pending[binding_ids == index])
        self._basis(np.arange(len(self.points)), np.zeros_like(self.level))
        return _ChunkResult(self.points, self.states, self.exceeded)
