"""The retention limit includes metadata, while bytes reports arrays only."""

from dataclasses import dataclass

from zcu_tools.simulate.evolution.cache import PrepareCache, PreparedPayload


@dataclass(frozen=True)
class TinyProduct:
    nbytes: int = 16


def test_metadata_consumes_capacity_and_eviction_keeps_array_bytes_separate():
    # 5000 bytes accommodates one tiny allocation with metadata, not two.
    cache = PrepareCache(5000)
    namespace = ("synthetic", "model", 0.001)
    first = PreparedPayload(TinyProduct(), (b"first",))
    cache.add(namespace, first)
    second = PreparedPayload(TinyProduct(), (b"second",))
    cache.add(namespace, second)
    assert cache.find(namespace, {b"first"}) == []
    assert cache.find(namespace, {b"second"}) == [second]
    assert cache.bytes == 16
    assert cache.bytes < cache.charged_bytes <= cache.limit
    cache.clear()
    assert cache.bytes == cache.charged_bytes == 0


def test_capacity_reclamation_releases_multiple_empty_key_allocations():
    # Two tiny metadata-bearing products fit, but both must yield to 4000 bytes.
    cache = PrepareCache(10000)
    namespace = ("synthetic", "model", 0.001)
    cache.add(namespace, PreparedPayload(TinyProduct(), ()))
    cache.add(namespace, PreparedPayload(TinyProduct(), ()))
    replacement = PreparedPayload(TinyProduct(4000), (b"large",))
    cache.add(namespace, replacement)
    assert cache.bytes == 4000
    assert cache.find(namespace, {b"large"}) == [replacement]
    assert cache.charged_bytes <= cache.limit
