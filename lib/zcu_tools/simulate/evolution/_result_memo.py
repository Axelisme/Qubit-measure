"""Engine-local exact-result LRU, separate from model-only preparations.

Entries never cross Engine/process lifetimes. The owning Engine fixes the kernel
collection for that lifetime. Like PrepareCache, reuse requires deterministic,
pure numerical kernels and the whole-source zero-level model identity contract.
No ADC samples, evolve-local plans or caller-owned arrays enter this memo.
"""

from collections import OrderedDict
from collections.abc import Sequence
from hashlib import sha256
from sys import getsizeof

import numpy as np
from numpy.typing import NDArray

from ._validation import shape
from .engine import EvolutionResult, TimelineBatch
from .kernel import KernelContext, SegmentTable
from .secular import ModelSource

# Default additional allowance; Engine clamps it to the prepare budget unless
# the caller supplies an explicit independent result-memo byte cap.
RESULT_MEMO_BYTES = 64 * 1024**2


class _InputDigest:
    """Length-framed bulk content hashing; never constructs per-row objects."""

    def __init__(self) -> None:
        self.value = sha256()

    def text(self, value: str) -> None:
        encoded = value.encode("utf-8")
        self.value.update(len(encoded).to_bytes(8, "little"))
        self.value.update(encoded)

    def array(self, value: NDArray[np.generic]) -> None:
        self.text(value.dtype.str)
        self.text(str(value.shape))
        self.value.update(value.tobytes(order="C"))

    def continuation(self, batch: TimelineBatch, state_shape: tuple[int, ...]) -> None:
        """Validate source dimensions and hash the paired absolute/DC inputs."""
        if batch.start_ns is None or batch.initial_densities is None:
            self.text("source initial density")
        else:
            shape(
                batch.initial_densities,
                (len(batch.points), *state_shape),
                "initial densities",
            )
            self.text("point-aligned continuation")
            self.array(batch.start_ns)
            self.array(batch.initial_densities)


def input_key(
    source: ModelSource, batches: Sequence[TimelineBatch], budget: float
) -> bytes:
    """Digest full input records before geometry/planning, with source identity.

    The DC identity describes the entire source builder, including nonzero flux
    levels. Initial density and optional continuous Hamiltonian expansion are
    separate source inputs and are hashed by content. Every batch/track field,
    including unused envelope candidates and beyond-readout values, participates.
    The source's midpoint subdivision policy also participates. Other fixed
    kernel policies belong to the owning Engine's in-process lifetime, not to
    a persistent or cross-engine key namespace.
    """
    digest = _InputDigest()
    loops = batches[0].flux.level.shape[1]
    digest.text(source.model((0,) * loops).identity)
    digest.array(source.initial_density)
    digest.text(float(budget).hex())
    digest.text(str(source.slow_envelope_segments_per_period))
    try:
        expansion = source.flux_expansion()
    except ValueError:
        # The source API uses ValueError to report an absent optional expansion.
        digest.text("no expansion")
    else:
        digest.text("expansion")
        digest.array(expansion.static)
        digest.text(str(len(expansion.terms)))
        for term in expansion.terms:
            digest.text(str(term.loop))
            digest.text(term.kind)
            digest.array(term.matrix)
    digest.text(str(len(batches)))
    for batch in batches:
        digest.array(batch.points)
        digest.array(batch.end_ns)
        digest.continuation(batch, source.initial_density.shape)
        for times in batch.segment_boundaries:
            digest.array(np.asarray(times, dtype=np.float64))
        digest.text("segment shifts")
        for shifts in batch.segment_shifts:
            digest.text(str(len(shifts)))
            for shift in shifts:
                digest.text(float(shift.start_ns).hex())
                digest.text(float(shift.end_ns).hex())
                digest.text(shift.content.decode())
        digest.array(batch.flux.offsets)
        digest.array(batch.flux.start_ns)
        digest.array(batch.flux.level)
        if batch.flux_waveform is None:
            digest.text("no waveform")
        else:
            digest.text("waveform")
            digest.array(batch.flux_waveform.offsets)
            digest.array(batch.flux_waveform.time_ns)
            digest.array(batch.flux_waveform.flux)
        digest.text(str(len(batch.tracks)))
        for track in batch.tracks:
            digest.text(track.port)
            for array in (
                track.start_ns,
                track.duration_ns,
                track.frequency_ghz,
                track.phase_rad,
                track.amplitude,
                track.envelope_index,
                track.active,
            ):
                digest.array(array)
            digest.text(str(len(track.envelopes)))
            for envelope in track.envelopes:
                # The immutable record already hashes kind, complex samples and
                # their time axis. Reuse it rather than serializing every pulse.
                digest.text(envelope.identity)
    return digest.value.digest()


def route_key(key: bytes, table: SegmentTable, context: KernelContext) -> bytes:
    """Bind an exact-input digest to source-amortized gate and multitone routes.

    Complete inventory counting precedes this call, including memo hits. Growing
    counters alone do not change output; selected keys and direct rows do.
    Only plans without cumulative candidates retain the raw key for safe early lookup.
    """
    from ._multitone_dispatch import populate_multitone_plan
    from ._shaped_dispatch import populate_single_gate_plan

    direct = populate_multitone_plan(
        table, context.source, context.work_plan, context.budget
    )
    choices = populate_single_gate_plan(
        table, context.source, context.work_plan, context.budget
    )
    if not choices and not np.any(table.tones >= 2):
        return key
    selected = sorted({k for choice in choices for k in choice.keys})
    rows = np.array(sorted(direct), dtype="<i8")
    return sha256(key + b"".join(selected) + b"multitone" + rows.tobytes()).digest()


class ResultMemo:
    """Serialized, byte-bounded LRU of owned final results for one Engine.

    limit is a nonnegative byte allowance. Charges include array allocations,
    diagnostics, keys and conservative per-entry Python/container overhead.
    find returns an independent owned copy, including writeable-flag isolation;
    add copies successful results so the original caller also cannot mutate an
    entry. Oversize results do not evict existing entries. clear drops ownership.
    Calls require the Engine lock; kernels must be deterministic and pure for
    equal inputs, just as when reusing model-only preparations.
    """

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._entries: OrderedDict[bytes, tuple[EvolutionResult, int]] = OrderedDict()
        self._bytes = 0

    def find(self, key: bytes, *, telemetry: bool = False) -> EvolutionResult | None:
        """Return one owned copy, with optional zero-CPU exact-hit routing.

        Telemetry is per lookup, not part of the stored numeric result. Disabled
        lookups retain no routing work; neither mode adds another array copy.
        """
        entry = self._entries.get(key)
        if entry is None:
            return None
        self._entries.move_to_end(key)
        result, _ = entry
        return EvolutionResult(
            result.points, result.densities, result.paths.for_reuse(telemetry=telemetry)
        )

    def add(self, key: bytes, result: EvolutionResult) -> None:
        """Copy and admit a successful result if its full charge fits the LRU."""
        size = _charge(key, result)
        if size > self._limit:
            return
        existing = self._entries.pop(key, None)
        if existing is not None:
            self._bytes -= existing[1]
        while self._bytes + size > self._limit:
            _, (_, evicted_size) = self._entries.popitem(last=False)
            self._bytes -= evicted_size
        stored = EvolutionResult(
            result.points, result.densities, result.paths.for_reuse(telemetry=False)
        )
        self._entries[key] = stored, size
        self._bytes += size

    def clear(self) -> None:
        """Release every result retained by this Engine."""
        self._entries.clear()
        self._bytes = 0


def _charge(key: bytes, result: EvolutionResult) -> int:
    # Arrays own their data, so getsizeof includes both buffer and array header.
    # The extra KiB covers OrderedDict capacity/entry, the value tuple and size
    # integer, with slack for allocator rounding and dataclass split dictionaries.
    size = (
        1024
        + getsizeof(key)
        + getsizeof(result)
        + getsizeof(vars(result))
        + getsizeof(result.points)
        + getsizeof(result.densities)
        + getsizeof(result.paths)
        + getsizeof(vars(result.paths))
        + getsizeof(result.paths.max_trace_error)
    )
    for counts in (
        result.paths.segments_by_kernel,
        result.paths.points_by_class,
        result.paths.exceeded_by_kernel,
    ):
        size += getsizeof(counts) + getsizeof(dict(counts))
        size += sum(
            getsizeof(name) + getsizeof(count) for name, count in counts.items()
        )
    return size
