"""Whole opaque allocation ownership through the documented cache collaborator."""

import weakref
from dataclasses import FrozenInstanceError, dataclass

import pytest
from zcu_tools.simulate.evolution.cache import PrepareCache, PreparedPayload

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


@pytest.mark.parametrize("field_name", ["nbytes", "keys", "revision", "proof"])
def test_inventory_is_immutable_whole_batch_metadata_without_lru_touch(field_name):
    cache = PrepareCache(20)
    assert cache.inventory() == ()
    counts = AllocationCounts()
    first = payload(counts, 10, b"a", b"b")
    cache.add(SPACE, first)
    cache.add(SPACE, payload(counts, 10, b"c"))
    snapshot = cache.inventory()
    assert [(entry.namespace, entry.keys, entry.nbytes) for entry in snapshot] == [
        (SPACE, (b"a", b"b"), 10),
        (SPACE, (b"c",), 10),
    ]
    assert snapshot[0].revision == "legacy"
    assert snapshot[0].proof == ()
    with pytest.raises(FrozenInstanceError):
        setattr(snapshot[0], field_name, 1)
    # Merely looking at the first allocation must not protect it from eviction.
    cache.add(SPACE, payload(counts, 10, b"d"))
    assert cache.find(SPACE, {b"a"}) == []
    assert [entry.keys for entry in cache.inventory()] == [(b"c",), (b"d",)]
    assert snapshot[0].keys == (b"a", b"b")
    cache.clear()
    del first
    assert counts.released == counts.created


def test_partial_hit_reservations_and_consumers_charge_whole_allocation_once():
    cache = PrepareCache(20)
    assert cache.live_bytes == 0
    counts = AllocationCounts()
    whole = payload(counts, 20, b"a", b"b", b"c")
    cache.add(SPACE, whole)
    entry = cache.inventory()[0]
    # A plan needing only one key must still reserve the entire opaque batch.
    reservation = cache.reserve((entry, entry))
    another = cache.reserve((entry,))
    assert reservation.entries == (entry,)
    assert reservation.nbytes == 20
    assert cache.live_bytes == 20
    consumer = reservation.acquire(entry)
    second_consumer = another.acquire(entry)
    assert consumer.payload is whole
    assert cache.bytes == 20
    assert cache.live_bytes == 20
    reservation.release()
    reservation.release()
    another.release()
    cache.clear()
    assert cache.bytes == 0
    assert cache.live_bytes == 20
    consumer.release()
    assert cache.live_bytes == 20
    second_consumer.release()
    assert cache.live_bytes == 0
    with pytest.raises(RuntimeError, match="released"):
        reservation.acquire(entry)
    with pytest.raises(RuntimeError, match="released"):
        _ = consumer.payload


def test_eviction_and_clear_keep_reserved_hits_until_consumer_release():
    cache = PrepareCache(10)
    assert cache.inventory() == ()
    counts = AllocationCounts()
    original = payload(counts, 10, b"a", b"b")
    observed = weakref.ref(original.prepared)
    cache.add(SPACE, original)
    entry = cache.inventory()[0]
    reservation = cache.reserve((entry,))
    del original
    cache.add(SPACE, payload(counts, 10, b"c"))
    cache.clear()
    assert cache.bytes == 0
    assert observed() is not None
    consumer = reservation.acquire(entry)
    reservation.release()
    assert consumer.payload.keys == (b"a", b"b")
    assert counts.released == 1
    consumer.release()
    consumer.release()
    assert observed() is None
    assert counts.released == counts.created
    # Metadata snapshots must not themselves retain any allocation.
    assert entry.nbytes == 10


def test_stale_or_foreign_reservation_fails_atomically_without_touching_lru():
    cache = PrepareCache(20)
    assert cache.inventory() == ()
    counts = AllocationCounts()
    cache.add(SPACE, payload(counts, 10, b"a"))
    cache.add(SPACE, payload(counts, 10, b"b"))
    stale, valid = cache.inventory()
    cache.add(SPACE, payload(counts, 10, b"c"))
    with pytest.raises(ValueError, match="resident"):
        cache.reserve((valid, stale))
    assert cache.live_bytes == 0
    other = PrepareCache(20)
    other.add(SPACE, payload(counts, 10, b"b"))
    with pytest.raises(ValueError, match="resident"):
        cache.reserve(other.inventory())
    with cache.reserve((valid,)) as reservation:
        assert reservation.nbytes == 10
    assert cache.live_bytes == 0
    # Successful reservations also leave canonical LRU order unchanged.
    cache.add(SPACE, payload(counts, 10, b"d"))
    assert cache.find(SPACE, {b"b"}) == []


def test_repeated_admission_and_live_oversize_have_separate_budgets():
    cache = PrepareCache(10)
    assert cache.live_bytes == 0
    counts = AllocationCounts()
    shared = payload(counts, 10, b"a", b"b")
    cache.add(SPACE, shared)
    cache.add(SPACE, shared)
    same_allocation = PreparedPayload(shared.prepared, shared.keys)
    cache.add(SPACE, same_allocation)
    assert cache.bytes == 10
    assert len(cache.inventory()) == 1
    oversized = payload(counts, 11, b"large")
    with cache.reference(SPACE, oversized) as consumer:
        cache.add(SPACE, oversized)
        assert cache.bytes == 10
        assert cache.live_bytes == 11
        assert consumer.payload is oversized
        assert cache.find(SPACE, {b"large"}) == []
        cache.clear()
        assert cache.live_bytes == 11
    assert cache.live_bytes == 0
    disabled = PrepareCache(0)
    disabled.add(SPACE, shared)
    assert disabled.inventory() == ()


def test_model_only_revision_and_proof_are_metadata_not_row_acceptance():
    cache = PrepareCache(10)
    assert cache.inventory() == ()
    counts = AllocationCounts()
    prepared = PreparedPayload(
        CountedPrepared(counts, 10), (b"a",), revision="r2", proof=("model-proof",)
    )
    cache.add(SPACE, prepared)
    entry = cache.inventory()[0]
    assert (entry.revision, entry.proof) == ("r2", ("model-proof",))
    assert cache.find(OTHER_SPACE, {b"a"}) == []
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
@pytest.mark.parametrize("operation", ["add", "reference"])
def test_invalid_byte_charges_preserve_resident_and_live_ownership(value, operation):
    cache = PrepareCache(10)
    counts = AllocationCounts()
    cache.add(SPACE, payload(counts, 10, b"resident"))
    before = cache.inventory()
    invalid = payload(counts, value, b"invalid")
    admit = cache.add if operation == "add" else cache.reference
    with pytest.raises(ValueError, match="nonnegative integers"):
        admit(SPACE, invalid)
    assert cache.inventory() == before
    assert cache.bytes == 10
    assert cache.live_bytes == 0


def test_invalid_charges_and_conflicting_allocation_metadata_fail_before_eviction():
    cache = PrepareCache(10)
    assert cache.inventory() == ()
    counts = AllocationCounts()
    whole = payload(counts, 10, b"a")
    cache.add(SPACE, whole)
    before = cache.inventory()
    with pytest.raises(ValueError, match="nonnegative"):
        cache.add(SPACE, payload(counts, -1, b"bad"))
    with pytest.raises(ValueError, match="metadata"):
        cache.add(SPACE, PreparedPayload(whole.prepared, (b"different",)))
    with pytest.raises(ValueError, match="metadata"):
        cache.add(OTHER_SPACE, whole)
    assert cache.inventory() == before
    assert cache.bytes == 10


def test_evolve_workspace_charge_tracks_growth_and_releases_on_failure():
    cache = PrepareCache(10)
    counts = AllocationCounts()
    workspace = CountedPrepared(counts, 0)
    resident = payload(counts, 10, b"resident")
    cache.add(SPACE, resident)

    def cancelled_evolve():
        with cache.track_workspace(workspace):
            assert cache.live_bytes == 0
            workspace.nbytes = 120
            assert cache.live_bytes == 120
            # Evolve-local arrays are live preparation, never persistent hits.
            assert cache.bytes == 10
            assert len(cache.inventory()) == 1
            cache.clear()
            assert cache.live_bytes == 120
            raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        cancelled_evolve()
    assert cache.live_bytes == 0
