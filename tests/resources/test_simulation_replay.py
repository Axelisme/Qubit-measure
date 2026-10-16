"""Detached private-input storage, not normal data.h5 or waveform assets."""

from pathlib import Path

import pytest
from zcu_tools.core.simulation import (
    AcquireRequest,
    EnvironmentSnapshot,
    PointTimeline,
    PulseEnvelope,
    PulseEvent,
    ReadoutWindow,
)
from zcu_tools.resources.ftf import parse_ftf

from tests.resources._ftf_data import ftf_document


def test_replay_retains_complex_samples_and_deduplicated_acquire(
    tmp_path: Path,
) -> None:
    from zcu_tools.resources.simulation_replay import (
        ReplayAcquire,
        read_replay,
        save_replay,
    )

    window = ReadoutWindow(0, 0, 3, 100.0, (0.02,))
    event = PulseEvent(
        2, 0.0, 0.016, 42.0, 90.0, -0.3, PulseEnvelope((0.0, 0.008), (1 + 2j, -3 - 4j))
    )
    timeline = PointTimeline((("reps", 0),), (event,), (window,), 0.05)
    request = AcquireRequest(
        "integrated",
        (1,),
        0,
        (window,),
        ((event,),),
        12345,
        17,
        "test-rng-v1",
        (timeline,),
    )
    fixed = ReplayAcquire(
        settings=parse_ftf(ftf_document()),
        request=request,
        versions=(("backend", "test-v1"),),
    )
    a = EnvironmentSnapshot("ftf-v1", "fixed-model", (("bias", 0.2),))
    b = EnvironmentSnapshot("ftf-v1", "fixed-model", (("bias", 0.4),))
    first = save_replay(tmp_path, fixed, a, 0)
    second = save_replay(tmp_path, fixed, b, 1)
    restored = read_replay(first)
    assert restored.acquire.request == request
    assert restored.round.snapshot == a
    assert read_replay(second).round.snapshot == b
    assert restored.round.acquire_identity == read_replay(second).round.acquire_identity
    assert restored.acquire.settings == fixed.settings
    # A missing shared input is an error, not current configuration substitution.
    acquire_path = tmp_path / "acquires" / f"{restored.round.acquire_identity}.json"
    acquire_path.unlink()
    with pytest.raises(FileNotFoundError):
        read_replay(second)


def test_replay_rejects_changed_saved_content(tmp_path: Path) -> None:
    from zcu_tools.resources.simulation_replay import read_replay

    damaged = tmp_path / ("0" * 64 + ".json")
    damaged.write_bytes(b"{}")
    with pytest.raises(ValueError, match="integrity"):
        read_replay(damaged)


def test_replay_round_trips_legacy_acquire_without_templates(tmp_path: Path) -> None:
    from zcu_tools.resources.simulation_replay import (
        ReplayAcquire,
        read_replay,
        save_replay,
    )

    request = AcquireRequest(
        mode="integrated",
        loop_dims=(3,),
        avg_level=0,
        readouts=(ReadoutWindow(0, 0, 4, 100.0, (0.02,)),),
        controls=((), (), ()),
        root_seed=5,
        sequence=8,
        rng_version="test-rng",
    )
    acquire = ReplayAcquire(
        settings=parse_ftf(ftf_document()),
        request=request,
        versions=(("backend", "test-v1"),),
    )
    snapshot = EnvironmentSnapshot("test-v1", "detached")
    path = save_replay(tmp_path, acquire, snapshot, 2)
    restored = read_replay(path)
    assert restored.acquire == acquire
    assert restored.acquire.request == request
    assert restored.acquire.request.templates == ()
    assert restored.round.snapshot == snapshot
    assert restored.round.index == 2


def test_replay_round_trips_every_template_array_with_dtype_and_shape(tmp_path: Path):
    from dataclasses import fields

    import numpy as np
    from zcu_tools.core.simulation import EventSlot, ReadoutSlot, TimelineTemplate
    from zcu_tools.resources.simulation_replay import (
        ReplayAcquire,
        read_replay,
        save_replay,
    )

    event = EventSlot(
        gen_ch=2,
        source="selected",
        iterations=(("inner", 1),),
        start_us=np.array([0.125, 0.25]),
        duration_us=np.array([0.016, 0.032]),
        frequency_mhz=np.array([42.0, 43.125]),
        phase_deg=np.array([90.0, -30.0]),
        gain=np.array([-0.3, 0.25]),
        envelopes=(None, PulseEnvelope((0.0, 0.008), (1 + 2j, -3 - 4j))),
        envelope_index=np.array([1, 0], dtype=np.int64),
        active=np.array([True, False], dtype=np.bool_),
        phase_reference_us=0.125,
    )
    readout = ReadoutSlot(
        channel=0,
        samples=3,
        sample_rate_mhz=100.0,
        trigger_us=np.array([[0.02, 0.04], [0.03, 0.05]]),
        frequency_mhz=np.array([[42.0, 43.0], [44.0, 45.0]]),
        phase_deg=np.array([[0.0, 90.0], [180.0, -90.0]]),
        generator_channels=np.array([[2, 0], [0, 2]], dtype=np.int64),
    )
    template = TimelineTemplate(
        points=np.array([0, 1], dtype=np.int64),
        events=(event,),
        readouts=(readout,),
        duration_us=np.array([0.4, 0.5]),
    )
    request = AcquireRequest(
        mode="integrated",
        loop_dims=(3, 2),
        avg_level=0,
        readouts=(ReadoutWindow(0, 0, 3, 100.0, (0.02, 0.04)),),
        controls=(),
        root_seed=12345,
        sequence=17,
        rng_version="test-rng-v1",
        point_dims=(2,),
        point_names=("axis",),
        reps=3,
        templates=(template,),
    )
    acquire = ReplayAcquire(
        settings=parse_ftf(ftf_document()),
        request=request,
        versions=(("backend", "test-v1"),),
    )
    snapshot = EnvironmentSnapshot("test-v1", "detached")
    restored = read_replay(save_replay(tmp_path, acquire, snapshot, 2))
    actual = restored.acquire.request
    for field in fields(request):
        if field.name != "templates":
            assert getattr(actual, field.name) == getattr(request, field.name)
    assert len(actual.templates) == 1
    observed = actual.templates[0]
    assert len(observed.events) == len(template.events)
    assert len(observed.readouts) == len(template.readouts)
    pairs = [
        (observed, template),
        (observed.events[0], event),
        (observed.readouts[0], readout),
    ]
    for loaded, original in pairs:
        for field in fields(original):
            value, expected = getattr(loaded, field.name), getattr(original, field.name)
            if isinstance(expected, np.ndarray):
                assert value.dtype == expected.dtype
                assert value.shape == expected.shape
                np.testing.assert_array_equal(value, expected)
            elif field.name not in ("events", "readouts"):
                assert value == expected
    assert restored.round.snapshot == snapshot
    assert restored.round.index == 2
