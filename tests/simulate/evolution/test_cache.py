"""Whole-payload LRU admission."""

from dataclasses import dataclass

import pytest
from zcu_tools.simulate.evolution.cache import PrepareCache, PreparedPayload

from ._engine import Payload

SPACE = ("synthetic", "source", 1e-4)
OTHER_SPACE = ("synthetic", "other-source", 1e-4)


@dataclass
class AllocationCounts:
    """Count synthetic whole allocations, independently of cache accounting."""

    created: int = 0
    released: int = 0


class CountedPrepared:
    """Tiny opaque payload with an explicit synthetic resident byte charge."""

    def __init__(self, counts: AllocationCounts, nbytes: int):
        self.counts = counts
        self.nbytes = nbytes
        counts.created += 1

    def __del__(self):
        self.counts.released += 1


def payload(counts: AllocationCounts, size: int, *keys: bytes) -> PreparedPayload:
    return PreparedPayload(CountedPrepared(counts, size), keys)


def test_repeated_admission_and_oversize_preserve_resident_budget():
    cache = PrepareCache(10)
    counts = AllocationCounts()
    shared = payload(counts, 10, b"a", b"b")
    cache.add(SPACE, shared)
    cache.add(SPACE, shared)
    same_allocation = PreparedPayload(shared.prepared, shared.keys)
    cache.add(SPACE, same_allocation)
    assert cache.bytes == 10
    assert cache.find(SPACE, {b"a"}) == [shared]
    oversized = payload(counts, 11, b"large")
    cache.add(SPACE, oversized)
    assert cache.bytes == 10
    assert cache.find(SPACE, {b"large"}) == []
    assert cache.find(SPACE, {b"a"}) == [shared]
    disabled = PrepareCache(0)
    disabled.add(SPACE, shared)
    assert disabled.bytes == 0
    assert disabled.find(SPACE, {b"a"}) == []


def test_rejected_priority_and_whole_batch_lru_keep_exact_array_charges():
    cache = PrepareCache(30)
    counts = AllocationCounts()
    first = payload(counts, 10, b"a", b"shared")
    failed = payload(counts, 5, b"failed")
    recent = payload(counts, 10, b"b")
    cache.add(SPACE, first)
    cache.add(OTHER_SPACE, failed, rejected=True)
    cache.add(SPACE, recent)
    assert cache.bytes == 25
    assert cache.find(SPACE, {b"shared"}) == [first]
    cache.add(SPACE, payload(counts, 10, b"new"))
    assert cache.bytes == 30
    assert cache.find(OTHER_SPACE, {b"failed"}) == []
    cache.add(SPACE, payload(counts, 10, b"last"))
    assert cache.bytes == 30
    assert cache.find(SPACE, {b"b"}) == []
    assert cache.find(SPACE, {b"a"}) == [first]


def test_model_only_payloads_persist_in_matching_namespace():
    cache = PrepareCache(10)
    counts = AllocationCounts()
    prepared = payload(counts, 10, b"a")
    cache.add(SPACE, prepared)
    assert cache.find(OTHER_SPACE, {b"a"}) == []
    assert cache.find(SPACE, {b"a"}) == [prepared]
    # Input-bound wrappers have evolve lifetime even if their object has nbytes.
    bound = PreparedPayload(CountedPrepared(counts, 1), (), lifetime="evolve")
    with pytest.raises(ValueError, match="model-only"):
        cache.add(SPACE, bound)
    assert cache.bytes == 10


@pytest.mark.parametrize("value", [False, True, -1, 1.5])
def test_cache_constructor_rejects_noninteger_or_negative_limits(value):
    with pytest.raises(ValueError, match="nonnegative integers"):
        PrepareCache(value)


@pytest.mark.parametrize("value", [False, True, -1, 1.5])
def test_invalid_byte_charges_preserve_resident_ownership(value):
    cache = PrepareCache(10)
    counts = AllocationCounts()
    cache.add(SPACE, payload(counts, 10, b"resident"))
    before = cache.find(SPACE, {b"resident"})
    invalid = payload(counts, value, b"invalid")
    with pytest.raises(ValueError, match="nonnegative integers"):
        cache.add(SPACE, invalid)
    assert cache.find(SPACE, {b"resident"}) == before
    assert cache.bytes == 10


def test_invalid_charges_and_conflicting_allocation_metadata_fail_before_eviction():
    cache = PrepareCache(10)
    counts = AllocationCounts()
    whole = payload(counts, 10, b"a")
    cache.add(SPACE, whole)
    before = cache.find(SPACE, {b"a"})
    with pytest.raises(ValueError, match="nonnegative"):
        cache.add(SPACE, payload(counts, -1, b"bad"))
    with pytest.raises(ValueError, match="metadata"):
        cache.add(SPACE, PreparedPayload(whole.prepared, (b"different",)))
    with pytest.raises(ValueError, match="metadata"):
        cache.add(OTHER_SPACE, whole)
    assert cache.find(SPACE, {b"a"}) == before
    assert cache.bytes == 10


def test_cache_admits_singleton_whole_allocation_and_preserves_active_reference():
    cache = PrepareCache(32)
    namespace = ("synthetic", "source", 0.001)
    singleton = PreparedPayload(Payload(2), (b"one",))
    cache.add(namespace, singleton)
    active = cache.find(namespace, {b"one"})
    assert active == [singleton]
    assert cache.bytes == 32
    cache.add(namespace, PreparedPayload(Payload(3), (b"oversize",)))
    assert cache.bytes == 32
    assert cache.find(namespace, {b"one"}) == [singleton]
    cache.add(namespace, PreparedPayload(Payload(1), (b"replacement",)))
    assert cache.find(namespace, {b"one"}) == []
    assert active[0].prepared.nbytes == 32


def test_exact_extension_replaces_whole_charge_without_invalidating_borrowers():
    cache = PrepareCache(22)
    counts = AllocationCounts()
    old = payload(counts, 10, b"a")
    other = payload(counts, 10, b"b")
    cache.add(SPACE, old)
    cache.add(SPACE, other)
    active = cache.find(SPACE, {b"a"})[0]
    extended = payload(counts, 12, b"a")
    replace_allocation = getattr(cache, "replace", None)
    assert callable(replace_allocation), "cache must replace exact extension ownership"
    replace_allocation(SPACE, old, extended)
    assert cache.bytes == 22
    assert cache.find(SPACE, {b"a"}) == [extended]
    assert cache.find(SPACE, {b"b"}) == [other]
    assert active.prepared.nbytes == 10
    with pytest.raises(ValueError, match="keys"):
        replace_allocation(SPACE, extended, payload(counts, 1, b"wrong"))
    with pytest.raises(ValueError, match="namespace"):
        replace_allocation(OTHER_SPACE, extended, payload(counts, 12, b"a"))
    with pytest.raises(ValueError, match="nonnegative"):
        replace_allocation(SPACE, extended, payload(counts, -1, b"a"))
    replace_allocation(SPACE, extended, payload(counts, 23, b"a"))
    assert cache.find(SPACE, {b"a"}) == [extended]
    assert cache.bytes == 22


def test_rejected_extension_grows_only_into_spare_capacity():
    cache = PrepareCache(22)
    counts = AllocationCounts()
    old = payload(counts, 10, b"a")
    other = payload(counts, 10, b"b")
    cache.add(SPACE, old, rejected=True)
    cache.add(SPACE, other)
    replace_allocation = getattr(cache, "replace", None)
    assert callable(replace_allocation), (
        "rejected domain extensions need bounded replacement"
    )
    replace_allocation(SPACE, old, payload(counts, 13, b"a"))
    assert cache.find(SPACE, {b"a"}) == [old]
    extended = payload(counts, 12, b"a")
    replace_allocation(SPACE, old, extended)
    assert cache.is_rejected(extended)
    assert cache.bytes == 22
    assert cache.find(SPACE, {b"b"}) == [other]
    # A stale extension cannot reinsert a predecessor removed by another owner.
    replace_allocation(SPACE, old, payload(counts, 11, b"a"))
    assert cache.find(SPACE, {b"a"}) == [extended]
