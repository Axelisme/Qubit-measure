"""Exact preparation streams with source-scoped immutable envelope reuse."""

from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.core.simulation import PulseEnvelope, RoundRequest
from zcu_tools.device import DeviceManager
from zcu_tools.experiment.ftf import FTFSource
from zcu_tools.experiment.ftf.preparation import preparation_key

from tests.experiment._ftf_templates import (
    expect_statistics,
    request,
    settings,
    snapshot,
)


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
    source = FTFSource(settings(), DeviceManager())
    reference = FTFSource(settings(), DeviceManager())
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
            from zcu_tools.simulate.evolution import Engine

            source.engine = Engine(1, 1024**2)
        later = replace(acquire, sequence=8)
        actual = expect_statistics(source.execute(RoundRequest(later, snap, 0)))
        expected = expect_statistics(reference.execute(RoundRequest(later, snap, 0)))
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
        assert single_entry_bytes > len(encoded)
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
