"""Exact detached-content keys and bounded ownership of noiseless preparations."""

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from hashlib import sha256
from sys import getsizeof
from typing import Generic, TypeVar

import numpy as np

from zcu_tools.core.simulation import AcquireRequest, PulseEnvelope

from .readout import FTFReadoutConfig

T = TypeVar("T")


def preparation_key(request: AcquireRequest, readout: FTFReadoutConfig) -> bytes:
    """Hash all detached C1/layout/readout values, excluding sampling identities.

    Physical/numeric generation and engine lifetime are managed by the owner,
    which clears the cache before lookup on any generation change. Borrowed C1
    arrays are read on every lookup; no program or array identity is trusted.
    Numeric envelopes use bulk binary encoding rather than per-sample recursion.
    """
    digest = sha256()

    def frame(data: bytes) -> None:
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(data)

    def visit(value: object) -> None:
        frame(type(value).__qualname__.encode())
        if isinstance(value, np.ndarray):
            frame(value.dtype.str.encode())
            frame(repr(value.shape).encode())
            frame(value.tobytes())
        elif isinstance(value, PulseEnvelope):
            visit(np.asarray(value.sample_us, dtype=np.float64))
            visit(np.asarray(value.values, dtype=np.complex128))
        elif is_dataclass(value) and not isinstance(value, type):
            for field in fields(value):
                frame(field.name.encode())
                visit(getattr(value, field.name))
        elif isinstance(value, tuple):
            frame(str(len(value)).encode())
            for child in value:
                visit(child)
        elif value is None or isinstance(value, (str, int, float, complex, bool)):
            frame(repr(value).encode())
        else:
            raise ValueError(f"unsupported preparation content: {type(value).__name__}")

    for field in fields(request):
        if field.name not in ("root_seed", "sequence", "rng_version"):
            frame(field.name.encode())
            visit(getattr(request, field.name))
    visit(readout)
    return digest.digest()


class PreparationCache(Generic[T]):
    """Byte-bounded LRU of owned, immutable noiseless preparation records.

    The source serializes calls and clears on physical/numeric/engine changes,
    failure and close. max_bytes is a nonnegative resident limit, adjustable
    between calls; zero disables residency. Active/caller references are outside
    the budget. Records must contain only numeric arrays, dataclasses, mappings
    and tuples, never models, modules or sampling identities. Accounting charges
    keys, Python containers, entry metadata and entire ndarray backing allocations;
    shared objects within an entry are deduplicated. Across entries shared objects
    are charged conservatively again. builds/hits are cumulative lookup misses/
    hits across clear(); resident_bytes counts retained entries and LRU storage.
    """

    def __init__(self, max_bytes: int = 4 * 1024**2) -> None:
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError("preparation max_bytes must be a nonnegative integer")
        self.max_bytes = max_bytes
        self.builds = 0
        self.hits = 0
        self.resident_bytes = 0
        self._entries: OrderedDict[bytes, tuple[T, int]] = OrderedDict()

    def clear(self) -> None:
        """Release every resident without resetting cumulative diagnostics."""
        self._entries.clear()
        self.resident_bytes = 0

    def get(self, key: bytes) -> T | None:
        """Return an exact hit and refresh LRU order, or count a preparation miss."""
        self._trim()
        entry = self._entries.get(key)
        if entry is None:
            self.builds += 1
            return None
        self._entries.move_to_end(key)
        self.hits += 1
        return entry[0]

    def put(self, key: bytes, value: T) -> None:
        """Retain a completed immutable record if it fits; evict oldest entries."""
        # The size scalar and tuple metadata are themselves resident. The walk
        # follows bases of views so a small slice cannot hide a large allocation.
        size = _owned_bytes((key, value)) + getsizeof(0) + getsizeof((value, 0))
        self._entries[key] = (value, size)
        self._trim()

    def _trim(self) -> None:
        self.resident_bytes = getsizeof(self._entries) + sum(
            entry[1] for entry in self._entries.values()
        )
        while self._entries and self.resident_bytes > self.max_bytes:
            self._entries.popitem(last=False)
            self.resident_bytes = getsizeof(self._entries) + sum(
                entry[1] for entry in self._entries.values()
            )
        if not self._entries:
            self.resident_bytes = 0


def _owned_bytes(root: object) -> int:
    seen: set[int] = set()

    def visit(value: object) -> int:
        if id(value) in seen:
            return 0
        seen.add(id(value))
        size = getsizeof(value)
        if isinstance(value, np.ndarray):
            return size + (visit(value.base) if value.base is not None else 0)
        if isinstance(value, Mapping):
            # MappingProxyType owns a hidden dictionary as well as its wrapper.
            if not isinstance(value, dict):
                size += getsizeof(dict(value))
            return size + sum(visit(k) + visit(v) for k, v in value.items())
        if isinstance(value, tuple):
            return size + sum(visit(child) for child in value)
        if is_dataclass(value) and not isinstance(value, type):
            if hasattr(value, "__dict__"):
                size += getsizeof(vars(value))
            return size + sum(
                visit(f.name) + visit(getattr(value, f.name)) for f in fields(value)
            )
        return size

    return visit(root)
