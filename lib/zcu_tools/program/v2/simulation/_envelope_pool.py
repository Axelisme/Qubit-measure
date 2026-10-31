"""Binding-owned, byte-bounded storage for detached DAC envelopes."""

from collections import OrderedDict
from dataclasses import dataclass
from sys import getsizeof

from zcu_tools.core.simulation import PulseEnvelope


@dataclass(frozen=True, slots=True)
class EnvelopeContent:
    """Exact compiled DAC inputs, never program identities or buffer references.

    dtype/shape/samples describe the current DAC array. maxv normalizes its
    amplitude; sample_period_ticks sets the sample interval in the request time base.
    flat_duration_ticks is playback duration for flat-tops, otherwise None.
    """

    dtype: str
    shape: tuple[int, ...]
    samples: bytes
    maxv: float
    fabric_mhz: float
    sample_period_ticks: int
    flat_duration_ticks: int | None


class EnvelopePool:
    """LRU immutable envelopes owned by one synchronous simulation binding.

    No compiled buffers are retained. get/put require exact EnvelopeContent keys.
    Eviction drops only pool references, never changes a caller's borrowed record.
    This class is not thread-safe; the binding's acquisition admission serializes
    its use. Accounting includes keys, Python numeric tuples and containers,
    conservatively charging shared objects separately. It is not a process RSS cap.
    """

    def __init__(self, max_bytes: int = 32 * 1024**2) -> None:
        """Set a nonnegative retained-byte budget; zero disables retention.

        Raises ValueError for negative budgets. Entries exceeding the budget are
        returned by the frontend normally but are not retained here.
        """
        if max_bytes < 0:
            raise ValueError("envelope pool budget must be nonnegative")
        self._max_bytes = max_bytes
        self._entries: OrderedDict[EnvelopeContent, tuple[PulseEnvelope, int]] = (
            OrderedDict()
        )
        self._bytes = 0

    def get(self, key: EnvelopeContent) -> PulseEnvelope | None:
        """Borrow the immutable envelope for current DAC content, or return None.

        A hit becomes most recently used. The returned object remains valid after
        eviction and after the binding is released.
        """
        entry = self._entries.get(key)
        if entry is None:
            return None
        self._entries.move_to_end(key)
        return entry[0]

    def put(self, key: EnvelopeContent, envelope: PulseEnvelope) -> None:
        """Retain an owned immutable envelope, evicting least recently used entries.

        envelope must contain a built-in integer sample period and complex values,
        as produced by the compiled frontend. Replacing a key releases the old
        pool reference only. Oversize entries are not retained.
        """
        previous = self._entries.pop(key, None)
        if previous is not None:
            self._bytes -= previous[1]
        size = _entry_bytes(key, envelope)
        if size <= self._max_bytes:
            self._entries[key] = (envelope, size)
            self._bytes += size
        while (
            self._entries and self._bytes + getsizeof(self._entries) > self._max_bytes
        ):
            _, (_, removed_bytes) = self._entries.popitem(last=False)
            self._bytes -= removed_bytes
        if not self._entries:
            # OrderedDict otherwise keeps an empty high-water hash allocation.
            self._entries = OrderedDict()


def _entry_bytes(key: EnvelopeContent, envelope: PulseEnvelope) -> int:
    # Samples are built-in complex values with one integer period. Accounting avoids
    # another Python walk over all samples on the cold conversion path.
    key_bytes = getsizeof(key) + sum(
        getsizeof(value)
        for value in (
            key.dtype,
            key.shape,
            key.samples,
            key.maxv,
            key.fabric_mhz,
            key.sample_period_ticks,
            key.flat_duration_ticks,
            *key.shape,
        )
    )
    envelope_bytes = (
        getsizeof(envelope)
        + getsizeof(envelope.__dict__)
        + getsizeof(envelope.sample_period_ticks)
        + getsizeof(envelope.values)
        + len(envelope.values) * getsizeof(0j)
    )
    return key_bytes + envelope_bytes + getsizeof((envelope, 0)) + getsizeof(0)
