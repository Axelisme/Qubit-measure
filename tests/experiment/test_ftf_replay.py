"""Author-opted persistent round replay through acquisition and detached inputs."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from zcu_tools.core.simulation import RoundRequest
from zcu_tools.device import DeviceManager
from zcu_tools.experiment.ftf import FTFAcquisitionOptions, FTFSource, ftf, replay_round
from zcu_tools.program.v2 import ModularProgramV2
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.program.v2.simulation import IntegratedRound, SimulationClock
from zcu_tools.resources.ftf import parse_ftf
from zcu_tools.resources.simulation_replay import read_replay, replay_paths

from tests.resources._ftf_data import ftf_document

pytestmark = pytest.mark.sim


@pytest.mark.parametrize("mode", ["integrated", "decimated"])
def test_replay_retains_round_raw_after_source_release(tmp_path: Path, mode):
    from tests.experiment._ftf_templates import request, settings, snapshot

    source = FTFSource(
        settings().with_telemetry(enabled=False),
        DeviceManager(),
        acquisition=FTFAcquisitionOptions(private_replay=tmp_path / "archive"),
    )
    acquire = request(mode)
    try:
        result = source.execute(
            RoundRequest(
                acquire, replace(snapshot(), model_identity=source.identity), 2
            )
        )
        if isinstance(result, IntegratedRound):
            expected = result.raw.materialize(acquire.reps, acquire.avg_level)
        else:
            expected = {
                channel.channel: channel.iq.copy() for channel in result.channels
            }
        archived_round = replay_paths(tmp_path / "archive")[0]
        # Eviction and a later round must not change this delivery's identity.
        source.preparation_cache.max_bytes = 0
        source.execute(
            RoundRequest(
                replace(acquire, sequence=8),
                replace(snapshot(), model_identity=source.identity),
                3,
            )
        )
    finally:
        source.close()
    if isinstance(result, IntegratedRound):
        retained = result.raw.materialize(acquire.reps, acquire.avg_level)
        for channel, values in expected.items():
            np.testing.assert_array_equal(retained[channel], values)
            retained[channel][...] = 0
        # Each source call owns fresh buffers, even after close.
        for channel, values in result.raw.materialize(
            acquire.reps, acquire.avg_level
        ).items():
            np.testing.assert_array_equal(values, expected[channel])
    replayed = replay_round(archived_round)
    if isinstance(replayed, IntegratedRound):
        actual = replayed.raw.materialize(acquire.reps, acquire.avg_level)
    else:
        actual = {channel.channel: channel.iq for channel in replayed.channels}
    for channel, values in actual.items():
        np.testing.assert_array_equal(values, expected[channel])
    if isinstance(result, IntegratedRound) and isinstance(replayed, IntegratedRound):
        np.testing.assert_array_equal(
            result.statistics.channels[0].sums, replayed.statistics.channels[0].sums
        )


def test_replay_freezes_drifted_settings_and_identity_without_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.experiment._ftf_templates import request, settings, snapshot

    document = settings().model_dump(mode="json")
    for loop in ("f1", "coupler", "f2"):
        document["chip"][loop]["q_cap"] = 1e6
    document["drift"] = {
        loop: {"q_cap": {"sigma": 0.3, "tau_s": 10.0}}
        for loop in ("f1", "coupler", "f2")
    }
    configuration = parse_ftf(document).with_telemetry(enabled=False)
    now = [100.0]
    archive = tmp_path / "archive"
    source = FTFSource(
        configuration,
        DeviceManager(),
        acquisition=FTFAcquisitionOptions(
            clock=SimulationClock(monotonic=lambda: now[0]),
            truth_log=tmp_path / "truth.jsonl",
            private_replay=archive,
        ),
    )
    acquire = request()
    caller_snapshot = snapshot()
    try:
        source.begin_acquire(seed=acquire.root_seed)
        now[0] += 10.0
        source.begin_acquire(seed=acquire.root_seed)
        effective = source.effective_settings
        identity = source.identity
        assert effective.chip != configuration.chip
        result = source.open_acquire(acquire).sample_round(caller_snapshot, 2)
        if not isinstance(result, IntegratedRound):
            raise AssertionError("integrated acquisition must retain raw shots")
        expected_raw = result.raw.materialize(acquire.reps, acquire.avg_level)
        archived_round = replay_paths(archive)[0]
        saved = read_replay(archived_round)
        assert saved.acquire.settings == effective
        assert saved.acquire.settings.drift is None
        assert saved.round.snapshot == replace(caller_snapshot, model_identity=identity)
        assert caller_snapshot == snapshot()
        now[0] += 1000.0
        source.begin_acquire(seed=acquire.root_seed)
        assert source.identity != identity
    finally:
        source.close()

    def unavailable_clock(clock: SimulationClock) -> float:
        raise AssertionError("private replay must not read the acquisition clock")

    # The clock is an external boundary, not part of the replay calculation.
    monkeypatch.setattr(SimulationClock, "now", unavailable_clock)
    replayed = replay_round(archived_round)
    if not isinstance(replayed, IntegratedRound):
        raise AssertionError("integrated replay must retain raw shots")
    np.testing.assert_array_equal(
        replayed.statistics.state_counts, result.statistics.state_counts
    )
    for actual, expected in zip(
        replayed.statistics.channels, result.statistics.channels, strict=True
    ):
        assert actual.channel == expected.channel
        np.testing.assert_array_equal(actual.sums, expected.sums)
    replayed_raw = replayed.raw.materialize(acquire.reps, acquire.avg_level)
    assert replayed_raw.keys() == expected_raw.keys()
    for channel, values in expected_raw.items():
        np.testing.assert_array_equal(replayed_raw[channel], values)


@pytest.mark.parametrize(
    "backend",
    [
        "ftf-direct-v1",
        "ftf-direct-v2",
        "ftf-segment-v6",
        "ftf-segment-v7",
        "ftf-segment-v10",
        "ftf-segment-v11",
        "ftf-segment-v12",
    ],
)
def test_replay_rejects_archive_from_obsolete_numeric_backend(
    tmp_path: Path,
    backend: str,
) -> None:
    from zcu_tools.experiment.ftf import replay_round
    from zcu_tools.resources.simulation_replay import (
        read_replay,
        replay_paths,
        save_replay,
    )

    archive = tmp_path / "current"
    env = ftf(
        parse_ftf(ftf_document()),
        acquisition=FTFAcquisitionOptions(private_replay=archive),
    )
    try:
        program = ModularProgramV2(
            env.soccfg,
            modules=[
                PulseReadoutCfg(
                    gen_ch=0,
                    ro_ch=0,
                    nqz=1,
                    freq=6000.0,
                    gain=0.1,
                    waveform=ConstWaveformCfg(length=0.016),
                    ro_length=0.016,
                ).build("ro")
            ],
            reps=1,
        )
        program.acquire(env.soc, progress=False)
    finally:
        env.devices.close_all_devices()
    saved = read_replay(replay_paths(archive)[0])
    old_versions = tuple(
        (name, backend if name == "backend" else value)
        for name, value in saved.acquire.versions
    )
    legacy = saved.acquire.model_copy(update={"versions": old_versions})
    path = save_replay(tmp_path / "legacy", legacy, saved.round.snapshot, 0)
    with pytest.raises(ValueError, match="version mismatch"):
        replay_round(path)


def test_replay_recomputes_statistics_without_numeric_records(tmp_path: Path) -> None:
    from zcu_tools.core.simulation import RoundRequest
    from zcu_tools.device import DeviceManager
    from zcu_tools.experiment.ftf import FTFSource, replay_round
    from zcu_tools.resources.simulation_replay import (
        read_replay,
        replay_paths,
        save_replay,
    )

    from tests.experiment._ftf_templates import (
        expect_statistics,
        request,
        settings,
        snapshot,
    )

    source = FTFSource(
        settings(),
        DeviceManager(),
        acquisition=FTFAcquisitionOptions(private_replay=tmp_path / "archive"),
    )
    try:
        expected = expect_statistics(
            source.execute(
                RoundRequest(
                    request(), replace(snapshot(), model_identity=source.identity), 2
                )
            )
        )
    finally:
        source.close()
    paths = replay_paths(tmp_path / "archive")
    saved = read_replay(paths[0])
    assert saved.round.numeric_manifest is None
    assert saved.acquire.library_path is None
    actual = expect_statistics(replay_round(paths[0]))
    np.testing.assert_array_equal(actual.channels[0].sums, expected.channels[0].sums)
    for name in ("numeric", "rng"):
        incompatible = saved.acquire.model_copy(
            update={
                "versions": tuple(
                    (key, "future" if key == name else value)
                    for key, value in saved.acquire.versions
                )
            }
        )
        mismatch = save_replay(tmp_path / name, incompatible, saved.round.snapshot, 2)
        with pytest.raises(ValueError, match="version mismatch"):
            replay_round(mismatch)
