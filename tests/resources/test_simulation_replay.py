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
