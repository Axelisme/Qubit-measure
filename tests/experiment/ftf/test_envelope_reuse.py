"""Plan-local immutable envelope sharing through the numeric planning seam."""

from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.core.simulation import PulseEnvelope
from zcu_tools.experiment.ftf import plan_template

from tests.experiment._ftf_templates import settings, snapshot, template

pytestmark = pytest.mark.sim


def test_plan_reuses_specs_without_changing_samples_or_candidate_indices():
    base = template(tones=3)
    shared = PulseEnvelope((0.0, 0.001, 0.002), (0.2j, -0.4 + 0.1j, 0.5))
    other = PulseEnvelope(shared.sample_us, (0.1j, -0.4 + 0.1j, 0.5))
    controls = tuple(
        replace(
            event,
            envelopes=(shared, None, shared, other),
            envelope_index=np.array([0, 3], dtype=np.int64),
        )
        for event in base.events[:-1]
    )
    inputs = replace(base, events=(*controls, base.events[-1]))
    plan = plan_template(settings(), snapshot(), inputs)
    tracks = plan.batch.tracks
    for event, track in zip(controls, tracks, strict=True):
        np.testing.assert_array_equal(track.envelope_index, event.envelope_index)
        np.testing.assert_array_equal(track.amplitude, event.gain)
        np.testing.assert_array_equal(track.frequency_ghz, event.frequency_mhz / 1000)
        np.testing.assert_array_equal(
            track.phase_rad,
            np.deg2rad(event.phase_deg)
            - 2 * np.pi * event.frequency_mhz * event.phase_reference_us,
        )
        for index, value in ((0, shared), (2, shared), (3, other)):
            spec = track.envelopes[index]
            np.testing.assert_array_equal(spec.samples, value.values)
            np.testing.assert_array_equal(
                spec.sample_ns, np.array(value.sample_us) * 1000
            )
            assert not spec.samples.flags.writeable
            assert not spec.sample_ns.flags.writeable
        assert track.envelopes[1].kind == "constant"
    assert tracks[0].envelopes[0] is tracks[0].envelopes[2]
    for track in tracks[1:]:
        assert track.envelopes[0] is tracks[0].envelopes[0]
        assert track.envelopes[1] is tracks[0].envelopes[1]
    assert tracks[0].envelopes[3] is not tracks[0].envelopes[0]
    with pytest.raises(ValueError, match="read-only"):
        tracks[0].envelopes[0].samples[0] = 9
    # Reusing the same input in another plan does not create resident reuse.
    later = plan_template(settings(), snapshot(), inputs)
    assert later.batch.tracks[0].envelopes[0] is not tracks[0].envelopes[0]
    np.testing.assert_array_equal(
        later.batch.tracks[0].envelopes[0].samples, tracks[0].envelopes[0].samples
    )
