"""Synthetic template expansion and deferred integrated-shot contracts."""

from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.core import simulation


def template_request():
    """Two structural classes with local arrays in sorted flat-point order."""
    envelope = simulation.PulseEnvelope((0.0, 0.25), (1j, -1j))
    event = simulation.EventSlot(
        gen_ch=2,
        source="drive",
        iterations=(("repeat", 1),),
        start_us=np.array([5.0, 3.0]),
        duration_us=np.array([0.5, 0.75]),
        frequency_mhz=np.array([100.0, 200.0]),
        phase_deg=np.array([10.0, 20.0]),
        gain=np.array([-0.25, 0.5]),
        envelopes=(None, envelope),
        envelope_index=np.array([0, 1], dtype=np.int64),
        active=np.array([True, False]),
        phase_reference_us=0.125,
    )
    readout = simulation.ReadoutSlot(
        channel=4,
        samples=16,
        sample_rate_mhz=8.0,
        trigger_us=np.array([[7.0, 8.0], [4.0, 6.0]]),
        frequency_mhz=np.array([[91.0, 92.0], [93.0, 94.0]]),
        phase_deg=np.array([[1.0, 2.0], [3.0, 4.0]]),
        generator_channels=np.array([[2, 3], [3, 2]], dtype=np.int64),
    )
    template = simulation.TimelineTemplate(
        points=np.array([0, 3], dtype=np.int64),
        events=(
            event,
            replace(
                event, active=np.array([True, True]), start_us=np.array([1.0, 2.0])
            ),
        ),
        readouts=(readout,),
        duration_us=np.array([10.0, 12.0]),
    )
    other = simulation.TimelineTemplate(
        points=np.array([1, 2], dtype=np.int64),
        events=(),
        readouts=(),
        duration_us=np.array([0.25, 0.5]),
    )
    return simulation.AcquireRequest(
        mode="integrated",
        loop_dims=(2, 5, 2),
        avg_level=1,
        point_dims=(2, 2),
        point_names=("flux", "frequency"),
        reps=5,
        templates=(template, other),
        readouts=(
            simulation.ReadoutWindow(
                channel=4,
                gen_ch=3,
                samples=16,
                sample_rate_mhz=8.0,
                trigger_us=(0.0, 0.0),
            ),
        ),
        root_seed=42,
        sequence=0,
        rng_version="test",
    )


def test_materialize_point_selects_template_active_events_and_detaches_values():
    request = template_request()
    first = request.materialize_point(0)
    assert first.coordinates == (("flux", 0), ("frequency", 0))
    assert [event.start_us for event in first.events] == [5.0, 1.0]
    assert first.events[0].envelope is None
    last = request.materialize_point(3)
    assert last.coordinates == (("flux", 1), ("frequency", 1))
    assert last.duration_us == 12.0
    assert last.events == (
        simulation.PulseEvent(
            gen_ch=2,
            start_us=2.0,
            duration_us=0.75,
            frequency_mhz=200.0,
            phase_deg=20.0,
            gain=0.5,
            envelope=simulation.PulseEnvelope((0.0, 0.25), (1j, -1j)),
            phase_reference_us=0.125,
            source="drive",
            iterations=(("repeat", 1),),
        ),
    )
    window = last.readouts[0]
    assert window == simulation.ReadoutWindow(
        channel=4,
        gen_ch=3,
        samples=16,
        sample_rate_mhz=8.0,
        trigger_us=(4.0, 6.0),
        frequency_mhz=(93.0, 94.0),
        phase_deg=(3.0, 4.0),
        generator_channels=(3, 2),
    )
    assert request.materialize_point(2).events == ()
    assert request.materialize_point(2).duration_us == 0.5
    request.templates[0].events[1].gain[:] = 0.0
    request.templates[0].readouts[0].trigger_us[:] = 0.0
    assert last.events[0].gain == 0.5
    assert window.trigger_us == (4.0, 6.0)


@pytest.mark.parametrize("point", [-1, 4])
def test_materialize_point_rejects_out_of_range(point: int):
    with pytest.raises(ValueError, match="point"):
        template_request().materialize_point(point)


def test_templates_require_names_for_each_point_axis():
    request = template_request()
    with pytest.raises(ValueError, match="point_names"):
        replace(request, point_names=("flux",))


@pytest.mark.parametrize(
    "seed,components", [(-1, ()), (1, (-1,)), (True, ()), (1, (1.5,))]
)
def test_core_sampling_identity_rejects_invalid_entropy(seed, components):
    with pytest.raises(ValueError, match="nonnegative integers"):
        simulation.SamplingIdentity(seed, components)
