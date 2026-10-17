"""Opaque model-only preparation cache collaborator for evolution scheduling.

Calls require the owning Engine call lock, or equivalent external serialization.
Only model-only prepare/guard products may be admitted. Input bindings, densities
and endpoint results belong to one evolve call and must never enter this cache.
"""

from collections import OrderedDict
from dataclasses import dataclass

from .kernel import Prepared


@dataclass(frozen=True)
class PreparedPayload:
    """Whole read-only model-only batch and its ordered opaque prepare keys.

    prepared is the kernel-owned payload; its nbytes is the nonnegative resident
    array charge, not process RSS. keys gives every entry in prepared, in kernel
    order. Callers retain ordinary strong references independently of the LRU.
    """

    prepared: Prepared
    keys: tuple[bytes, ...]


class PrepareCache:
    """LRU charged once per opaque payload, scoped by source, budget and family."""

    def __init__(self, limit: int):
        self.limit = limit
        self.bytes = 0
        self.payloads: OrderedDict[
            int, tuple[tuple[str, str, float], PreparedPayload]
        ] = OrderedDict()
        self.next_id = 0

    def find(
        self, namespace: tuple[str, str, float], keys: set[bytes]
    ) -> list[PreparedPayload]:
        """Return matching payloads and mark them recently used."""
        found: list[PreparedPayload] = []
        for entry, (space, payload) in list(self.payloads.items()):
            if space == namespace and keys.intersection(payload.keys):
                found.append(payload)
                self.payloads.move_to_end(entry)
        return found

    def add(self, namespace: tuple[str, str, float], payload: PreparedPayload) -> None:
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
