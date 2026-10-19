"""Exact preparation streams with source-scoped immutable envelope reuse."""

from collections import OrderedDict
from dataclasses import dataclass, fields, replace
from sys import getsizeof

import numpy as np
import pytest
from zcu_tools.core.simulation import PulseEnvelope, RoundRequest
from zcu_tools.device import DeviceManager
from zcu_tools.experiment.ftf import FTFSource
from zcu_tools.experiment.ftf.preparation import EnvelopeEncoder, preparation_key
from zcu_tools.simulate.evolution import Engine

from tests.experiment._ftf_templates import (
    expect_statistics,
    request,
    settings,
    snapshot,
)


def test_envelope_encoding_is_a_fixed_size_content_fingerprint():
    encoder = EnvelopeEncoder()
    envelope = PulseEnvelope((0.0, 0.5, 1.0), (0j, 0.5j, 1j))
    fingerprint = encoder.encode(envelope)
    assert len(fingerprint) == 32, "outer keys must hash a SHA256 fingerprint"
    assert encoder.encode(envelope) == fingerprint
    assert encoder.builds == 1
    assert encoder.hits == 1
    assert encoder.encode(replace(envelope)) == fingerprint
    assert EnvelopeEncoder(0).encode(envelope) == fingerprint
    for other in (
        replace(envelope, sample_us=(0.0, 0.25, 1.0)),
        replace(envelope, values=(0j, -0.5j, 1j)),
        PulseEnvelope((0.0, 1.0), (0j, 1j)),
    ):
        assert encoder.encode(other) != fingerprint


class SampleTimes(tuple[float, ...]):
    """Non-builtin tuple producer accepted by the generic accounting path."""


class SampleValue(complex):
    """Non-builtin scalar with a larger object header than builtin complex."""


@dataclass(frozen=True)
class ExtendedEnvelope(PulseEnvelope):
    calibration: tuple[float, ...] = (0.125, 0.25)


def _accounting_envelope(kind: str) -> PulseEnvelope:
    times = tuple(float(i) / 64 for i in range(64))
    values = tuple(complex(i / 64, 0.5) for i in range(64))
    if kind == "shared":
        return PulseEnvelope(times, (complex(0.5, 0.25),) * len(times))
    if kind == "empty":
        return PulseEnvelope((), ())
    if kind == "scalar-subclass":
        return PulseEnvelope(times, tuple(SampleValue(v) for v in values))
    if kind == "tuple-subclass":
        return PulseEnvelope(SampleTimes(times), values)
    if kind == "envelope-subclass":
        return ExtendedEnvelope(times, values)
    return PulseEnvelope(times, values)


def _retained_charge(envelope: PulseEnvelope, fingerprint: bytes, *, deduplicate: bool):
    """Independent size oracle for these tuple-only records and one LRU entry."""
    objects: list[object] = [envelope, vars(envelope)]
    for field in fields(envelope):
        values = getattr(envelope, field.name)
        objects.extend((field.name, values))
        objects.extend(values)
    if deduplicate:
        objects = list({id(obj): obj for obj in objects}.values())
    metadata = (
        getsizeof(fingerprint)
        + getsizeof(id(envelope))
        + getsizeof((envelope, fingerprint, 0))
        + getsizeof(0)
        + getsizeof(OrderedDict([(id(envelope), (envelope, fingerprint, 0))]))
    )
    return sum(getsizeof(obj) for obj in objects) + metadata


@pytest.mark.parametrize(
    "kind",
    [
        "distinct",
        "shared",
        "empty",
        "scalar-subclass",
        "tuple-subclass",
        "envelope-subclass",
    ],
)
def test_encoder_retained_charge_is_conservative_and_fallback_is_exact(kind):
    envelope = _accounting_envelope(kind)
    encoder = EnvelopeEncoder()
    fingerprint = encoder.encode(envelope)
    generic_charge = _retained_charge(envelope, fingerprint, deduplicate=True)
    assert generic_charge <= encoder.resident_bytes <= encoder.max_bytes
    if kind in ("distinct", "shared", "empty"):
        occurrence_charge = _retained_charge(envelope, fingerprint, deduplicate=False)
        assert encoder.resident_bytes >= occurrence_charge, (
            "builtin accounting must conservatively charge every scalar/tuple occurrence"
        )
    else:
        assert encoder.resident_bytes == generic_charge, (
            "non-builtin records, containers and scalars must retain generic accounting"
        )
    assert encoder.encode(envelope) == fingerprint
    assert (encoder.builds, encoder.hits) == (1, 1)
    assert EnvelopeEncoder(0).encode(envelope) == fingerprint


@pytest.mark.parametrize("kind", ["distinct", "shared", "empty", "scalar-subclass"])
def test_encoder_accounting_respects_capacity_disable_and_clear(kind):
    envelope = _accounting_envelope(kind)
    encoder = EnvelopeEncoder()
    fingerprint = encoder.encode(envelope)
    encoder.max_bytes = encoder.resident_bytes
    for _ in range(3):
        assert encoder.encode(replace(envelope)) == fingerprint
        assert 0 < encoder.resident_bytes <= encoder.max_bytes
    builds = encoder.builds
    assert encoder.encode(envelope) == fingerprint
    assert encoder.builds == builds + 1, "the earlier entry must have been evicted"
    encoder.clear()
    assert encoder.resident_bytes == 0
    assert encoder.encode(envelope) == fingerprint
    assert encoder.builds == builds + 2
    encoder.max_bytes = 0
    assert encoder.encode(envelope) == fingerprint
    assert encoder.resident_bytes == 0
    assert encoder.encode(envelope) == fingerprint
    assert encoder.builds == builds + 4, "disabled retention must rebuild every time"
    encoder.clear()
    assert encoder.resident_bytes == 0
    assert encoder.builds == builds + 4


class RecordingEngine(Engine):
    """Observe actual density outputs at the injected evolution boundary."""

    def __init__(self):
        super().__init__(1, 1024**2)
        self.density_bytes = b""

    def evolve(self, source, batches, *, budget=1e-3, cancel):
        result = super().evolve(source, batches, budget=budget, cancel=cancel)
        self.density_bytes = result.densities.tobytes()
        return result


@pytest.mark.parametrize("change", ["equal", "borrowed", "eviction", "disabled"])
def test_execute_keeps_signals_and_densities_exact_across_fingerprint_reuse(change):
    engine, reference_engine = RecordingEngine(), RecordingEngine()
    source = FTFSource(settings(), DeviceManager(), engine=engine)
    reference = FTFSource(settings(), DeviceManager(), engine=reference_engine)
    reference.envelope_encoder.max_bytes = 0
    reference.preparation_cache.max_bytes = 0
    try:
        acquire, snap = request(), snapshot()

        def compare(current):
            actual = expect_statistics(source.execute(RoundRequest(current, snap, 0)))
            expected = expect_statistics(
                reference.execute(RoundRequest(current, snap, 0))
            )
            assert actual.state_counts.tobytes() == expected.state_counts.tobytes()
            assert (
                actual.channels[0].sums.tobytes() == expected.channels[0].sums.tobytes()
            )
            assert engine.density_bytes == reference_engine.density_bytes
            return actual.channels[0].sums.tobytes()

        original = compare(acquire)
        event = acquire.templates[0].events[0]
        envelope = event.envelopes[0]
        assert envelope is not None
        if change == "equal":
            event = replace(event, envelopes=(replace(envelope),))
        elif change == "borrowed":
            event.gain[0] *= 2
        elif change == "eviction":
            source.envelope_encoder.max_bytes = source.envelope_encoder.resident_bytes
            other = replace(envelope, values=(1j, 1j))
            intermediate = replace(
                acquire,
                templates=(
                    replace(
                        acquire.templates[0],
                        events=(replace(event, envelopes=(other,)),),
                    ),
                ),
            )
            compare(intermediate)
        else:
            source.envelope_encoder.max_bytes = 0
        later = replace(
            acquire,
            sequence=8,
            templates=(replace(acquire.templates[0], events=(event,)),),
        )
        compare(later)
        if change == "borrowed":
            # Same stream identity isolates the changed borrowed gain from RNG changes.
            assert compare(replace(later, sequence=7)) != original
            assert source.preparation_cache.builds == 2
        elif change == "equal":
            assert source.preparation_cache.builds == 1
            assert source.preparation_cache.hits == 1
            assert source.envelope_encoder.builds == 2
        elif change == "eviction":
            assert source.envelope_encoder.builds == 3
        else:
            assert source.envelope_encoder.resident_bytes == 0
            assert source.envelope_encoder.builds == 2
        source.close()
        assert source.envelope_encoder.resident_bytes == 0
        with pytest.raises(RuntimeError, match="closed"):
            source.execute(RoundRequest(later, snap, 0))
    finally:
        source.close()
        reference.close()


def test_shared_envelope_encodes_once_and_preserves_exact_stream():
    source = FTFSource(settings(), DeviceManager())
    try:
        assert hasattr(source, "envelope_encoder"), "source must own envelope encoding"
        encoder = source.envelope_encoder
        base = request()
        event = base.templates[0].events[0]
        envelope = event.envelopes[0]
        assert envelope is not None
        repeated = replace(event, envelopes=(envelope, None, envelope))
        acquire = replace(
            base, templates=(replace(base.templates[0], events=(repeated,)),)
        )
        expected = preparation_key(acquire, source.readout)
        for sequence in (7, 8):
            actual = preparation_key(
                replace(acquire, sequence=sequence), source.readout, encoder=encoder
            )
            assert actual == expected
        assert encoder.builds == 1
        assert encoder.hits == 3
        assert 0 < encoder.resident_bytes <= encoder.max_bytes
    finally:
        source.close()


def test_changed_envelope_and_borrowed_arrays_change_key_without_stale_reuse():
    source = FTFSource(settings(), DeviceManager())
    try:
        assert hasattr(source, "envelope_encoder"), "source must own envelope encoding"
        encoder = source.envelope_encoder
        acquire = request()
        initial = preparation_key(acquire, source.readout, encoder=encoder)
        event = acquire.templates[0].events[0]
        # The same request and ndarray identities must not hide mutations.
        event.gain[0] += 0.125
        changed = preparation_key(acquire, source.readout, encoder=encoder)
        assert changed != initial
        assert changed == preparation_key(acquire, source.readout)
        assert encoder.builds == 1
        assert encoder.hits == 1
        envelope = event.envelopes[0]
        assert envelope is not None
        event = replace(event, envelopes=(replace(envelope, values=(1j, -1j)),))
        acquire = replace(
            acquire, templates=(replace(acquire.templates[0], events=(event,)),)
        )
        final = preparation_key(acquire, source.readout, encoder=encoder)
        assert final != changed
        assert final == preparation_key(acquire, source.readout)
        assert encoder.builds == 2
    finally:
        source.close()


@pytest.mark.parametrize("change", ["snapshot", "settings", "engine"])
def test_generation_still_invalidates_preparation_and_keeps_output_exact(change):
    source = FTFSource(settings(), DeviceManager(), engine=RecordingEngine())
    reference_engine = RecordingEngine()
    reference = FTFSource(settings(), DeviceManager(), engine=reference_engine)
    reference.envelope_encoder.max_bytes = 0
    reference.preparation_cache.max_bytes = 0
    old_engine = source.engine
    try:
        assert hasattr(source, "envelope_encoder"), "source must own envelope encoding"
        acquire, snap = request(), snapshot()
        first = expect_statistics(source.execute(RoundRequest(acquire, snap, 0)))
        expected = expect_statistics(reference.execute(RoundRequest(acquire, snap, 0)))
        assert first.state_counts.tobytes() == expected.state_counts.tobytes()
        assert first.channels[0].sums.tobytes() == expected.channels[0].sums.tobytes()
        assert source.envelope_encoder.builds == 1
        if change == "snapshot":
            snap = replace(snap, model_identity="new-generation")
        elif change == "settings":
            source.settings = source.settings.model_copy(
                update={
                    "numerics": source.settings.numerics.model_copy(
                        update={
                            "solver": source.settings.numerics.solver.model_copy(
                                update={
                                    "rtol": source.settings.numerics.solver.rtol * 2
                                }
                            )
                        }
                    )
                }
            )
            reference.settings = source.settings
        else:
            source.engine = RecordingEngine()
        later = replace(acquire, sequence=8)
        actual = expect_statistics(source.execute(RoundRequest(later, snap, 0)))
        expected = expect_statistics(reference.execute(RoundRequest(later, snap, 0)))
        assert isinstance(source.engine, RecordingEngine)
        assert source.engine.density_bytes == reference_engine.density_bytes
        assert source.preparation_cache.builds == 2
        assert source.envelope_encoder.builds == 1
        assert source.envelope_encoder.hits == 1
        assert actual.state_counts.tobytes() == expected.state_counts.tobytes()
        assert actual.channels[0].sums.tobytes() == expected.channels[0].sums.tobytes()
    finally:
        source.close()
        reference.close()
        old_engine.close()


def test_encoder_byte_budget_eviction_disable_and_source_close():
    source = FTFSource(settings(), DeviceManager())
    try:
        assert hasattr(source, "envelope_encoder"), "source must own envelope encoding"
        encoder = source.envelope_encoder
        envelopes = [
            PulseEnvelope(tuple(np.linspace(0.0, 1.0, 64)), (complex(i),) * 64)
            for i in range(3)
        ]
        encoded = encoder.encode(envelopes[0])
        single_entry_bytes = encoder.resident_bytes
        assert single_entry_bytes > (
            getsizeof(envelopes[0])
            + getsizeof(envelopes[0].sample_us)
            + getsizeof(envelopes[0].values)
            + sum(getsizeof(t) for t in envelopes[0].sample_us)
        ), "the budget must charge the full retained envelope, not just its digest"
        encoder.max_bytes = single_entry_bytes
        encoder.encode(envelopes[1])
        assert encoder.resident_bytes <= encoder.max_bytes
        before = encoder.builds
        assert encoder.encode(envelopes[0]) == encoded
        assert encoder.builds == before + 1, "oldest entry must be evicted"
        encoder.max_bytes = 0
        encoder.encode(envelopes[2])
        assert encoder.resident_bytes == 0
        encoder.max_bytes = 1024**2
        encoder.encode(envelopes[0])
        assert encoder.resident_bytes > 0
        source.close()
        assert encoder.resident_bytes == 0
        before = encoder.builds
        assert encoder.encode(envelopes[0]) == encoded
        assert encoder.builds == before + 1, "close must release resident encodings"
    finally:
        source.close()
