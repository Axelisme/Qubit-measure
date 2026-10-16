"""Internal prepare/apply seam shared by algorithm families and scheduling.

This is deliberately not re-exported by the evolution package. Kernels own
opaque prepared payloads; the engine only sees their resident array byte size.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from ._validation import integers, owned, shape
from .engine import EnvelopeSpec
from .secular import FluxWaveform, ModelSource


@dataclass(frozen=True)
class SegmentTable:
    """Structure-of-arrays segment instances; no initial density is stored here.

    timeline: nonnegative int64 (rows,), batch-local timeline index.
    start_ns, duration_ns: nonnegative float64 (rows,), segment support.
    level: int64 (rows,loops), constant quantized flux level.
    tones: int64 (rows,), number of playing tones, from 0 to max_tones.
    tone_port: int64 (rows,max_tones), index into the model's drives, -1 pad.
    tone_frequency_ghz: float64 (rows,max_tones), carrier cycles/ns.
    tone_amplitude: float64 (rows,max_tones), signed real port-scaled gain.
    tone_theta_rad: float64 (rows,max_tones), effective phase at segment start
        2*pi*f*start_ns + common-origin phase, including virtual Z.
    tone_envelope: int64 (rows,max_tones), index into envelopes; -1 pad.
    tone_envelope_offset_ns: float64 (rows,max_tones), envelope-local time
        at segment start (a segment may cut an existing pulse).
    envelopes: shared EnvelopeSpec tuple. Only slots before tones[row] play;
        floating padding is finite and ignored.
    flux_waveform: optional FluxWaveform indexed by batch-local timeline, not
        row. Its offsets are (timelines+1,), samples (total,loops), with the
        same loop count as level. Linear interpolation holds endpoint samples.
        None means all rows lack continuous samples. In combined batches a
        timeline without samples has a single zero-time placeholder; only
        quantized kernels may consume such a timeline.
    flux_waveform_available: optional bool (timelines,) mask. False marks
        placeholder timelines without continuous samples. None means all
        waveform timelines have samples (or no waveform is supplied).
    Arrays own read-only copies. Invalid shapes/counts/indices raise ValueError.
    Adjacent rows of one timeline may be merged by DOP853 kernels into a
    multi-interval work item: stop at boundaries and retain the accepted step.

    Raises:
        ValueError: message names shape for misaligned arrays, tone counts
        for excess tones, or nonnegative for invalid segment support.
    """

    timeline: NDArray[np.int64]
    start_ns: NDArray[np.float64]
    duration_ns: NDArray[np.float64]
    level: NDArray[np.int64]
    tones: NDArray[np.int64]
    tone_port: NDArray[np.int64]
    tone_frequency_ghz: NDArray[np.float64]
    tone_amplitude: NDArray[np.float64]
    tone_theta_rad: NDArray[np.float64]
    tone_envelope: NDArray[np.int64]
    tone_envelope_offset_ns: NDArray[np.float64]
    envelopes: tuple[EnvelopeSpec, ...]
    flux_waveform: FluxWaveform | None = None
    flux_waveform_available: NDArray[np.bool_] | None = None

    def __post_init__(self) -> None:
        timeline = integers(self.timeline, "segment timelines")
        if timeline.ndim != 1 or np.any(timeline < 0):
            raise ValueError("segment timelines must be a nonnegative vector")
        rows = len(timeline)
        object.__setattr__(self, "timeline", timeline)
        for name in ("start_ns", "duration_ns"):
            array = owned(getattr(self, name), np.float64, name)
            shape(array, (rows,), name)
            if np.any(array < 0):
                raise ValueError("segment support must be nonnegative")
            object.__setattr__(self, name, array)
        levels = integers(self.level, "segment levels")
        if levels.ndim != 2 or levels.shape[0] != rows or levels.shape[1] == 0:
            raise ValueError("segment levels must have shape (rows, loops)")
        object.__setattr__(self, "level", levels)
        if self.flux_waveform is not None:
            if self.flux_waveform.flux.shape[1] != levels.shape[1]:
                raise ValueError("flux waveform loop shape must match segment levels")
            if rows and timeline.max() >= len(self.flux_waveform.offsets) - 1:
                raise ValueError("flux waveform must cover segment timelines")
        if self.flux_waveform_available is not None:
            if self.flux_waveform is None:
                raise ValueError("waveform availability requires a flux waveform")
            mask = owned(
                self.flux_waveform_available, np.bool_, "waveform availability"
            )
            shape(mask, (len(self.flux_waveform.offsets) - 1,), "waveform availability")
            object.__setattr__(self, "flux_waveform_available", mask)
        self._validate_tones(rows)

    def _validate_tones(self, rows: int) -> None:
        tones = integers(self.tones, "tone counts")
        shape(tones, (rows,), "tone counts")
        ports = integers(self.tone_port, "tone ports")
        if ports.ndim != 2 or ports.shape[0] != rows:
            raise ValueError("tone ports must have shape (rows, max_tones)")
        width = ports.shape[1]
        if np.any(tones < 0) or np.any(tones > width):
            raise ValueError("tone counts exceed table width")
        envelopes = integers(self.tone_envelope, "tone envelopes")
        shape(envelopes, ports.shape, "tone envelopes")
        playing = np.arange(width)[None, :] < tones[:, None]
        if np.any(ports[playing] < 0) or np.any(ports[~playing] != -1):
            raise ValueError(
                "tone ports require nonnegative playing indices and -1 padding"
            )
        if (
            np.any(envelopes[playing] < 0)
            or np.any(envelopes[playing] >= len(self.envelopes))
            or np.any(envelopes[~playing] != -1)
        ):
            raise ValueError("tone envelope index out of range or invalid padding")
        for name in (
            "tone_frequency_ghz",
            "tone_amplitude",
            "tone_theta_rad",
            "tone_envelope_offset_ns",
        ):
            array = owned(getattr(self, name), np.float64, name)
            shape(array, ports.shape, name)
            object.__setattr__(self, name, array)
        if np.any(self.tone_envelope_offset_ns[playing] < 0):
            raise ValueError("envelope-local times must be nonnegative")
        object.__setattr__(self, "tones", tones)
        object.__setattr__(self, "tone_port", ports)
        object.__setattr__(self, "tone_envelope", envelopes)


class Prepared(Protocol):
    """Opaque state-independent kernel payload, shared read-only by workers.

    Each kernel defines its own concrete payload. nbytes counts resident array
    bytes owned by the whole batch of prepared entries, not Python overhead,
    models, solver workspaces or process RSS. It must be a nonnegative integer
    and must not change while the payload resides in the engine's LRU cache.
    """

    @property
    def nbytes(self) -> int:
        """Resident array bytes charged to the prepare cache for this payload."""
        ...


class Kernel(Protocol):
    """One stateless algorithm family; all persistent data lives in Prepared.

    name: stable algorithm name for diagnostics and cache partitioning.
    Multiple numerical families share this seam so the scheduler need not know
    their representation. All index arrays below are int64 vectors.
    """

    name: str

    def guard(
        self,
        table: SegmentTable,
        rows: NDArray[np.int64],
        source: ModelSource,
        budget: float,
    ) -> NDArray[np.bool_]:
        """Pure validity mask (len(rows),) at positive per-segment error budget.

        Only algorithm-local guards apply here; the engine owns branch order
        and cross-family rules such as prepare-key usage counts.
        """
        ...

    def prepare_keys(
        self, table: SegmentTable, rows: NDArray[np.int64]
    ) -> NDArray[np.void]:
        """Fixed-width keys (len(rows),), equal keys sharing one preparation.

        Keys include the model identity and exclude apply-only parameters;
        the exact algorithm determines which parameters belong in each half.
        """
        ...

    def cost(self, table: SegmentTable, rows: NDArray[np.int64]) -> NDArray[np.float64]:
        """Return nonnegative finite relative row costs for chunk interleaving."""
        ...

    def prepare(
        self,
        keys: NDArray[np.void],
        table: SegmentTable,
        rows: NDArray[np.int64],
        source: ModelSource,
        budget: float,
    ) -> Prepared:
        """Build data for unique keys in their given order; never read density.

        rows: one representative row of table per key, aligned with keys;
        prepare_keys(table, rows) equals keys. Content a fixed-width key cannot
        carry (e.g. envelope samples) is read from these rows; the result must
        depend only on the keys, so any representative gives the same payload.
        budget is the positive absolute per-segment error allowance.
        Numerical failure raises RuntimeError. Prepared data is immutable to
        apply workers and reports its resident array bytes through nbytes.
        """
        ...

    def apply(
        self,
        prepared: Prepared,
        key_index: NDArray[np.int64],
        table: SegmentTable,
        rows: NDArray[np.int64],
        states: NDArray[np.complex128],
        budget: float,
    ) -> None:
        """Advance complex128 states (len(rows),M,M) in place through rows.

        key_index maps each row to its prepared entry. budget is the positive
        absolute per-segment allowance. Called by a joblib worker thread;
        numba implementations must be nogil and single-threaded. Prepared
        data and table stay unchanged. Numerical failure raises RuntimeError.
        """
        ...
