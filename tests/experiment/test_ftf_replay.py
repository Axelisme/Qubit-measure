"""Author-opted persistent round replay through acquisition and detached inputs."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from zcu_tools.core.simulation import RoundRequest, RoundSignals
from zcu_tools.device import DeviceManager
from zcu_tools.experiment.ftf import FTFSource, ftf, replay_round
from zcu_tools.program.v2.simulation import IntegratedRound
from zcu_tools.resources.simulation_replay import replay_paths
from zcu_tools.program.v2 import ModularProgramV2, ProgramV2Cfg
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.resources.ftf import parse_ftf

from tests.resources._ftf_data import ftf_document

pytestmark = pytest.mark.sim


@pytest.mark.parametrize("mode", ["integrated", "decimated"])
def test_replay_retains_round_raw_after_source_release(tmp_path: Path, mode):
    from tests.experiment._ftf_templates import request, settings, snapshot

    source = FTFSource(
        settings().with_telemetry(enabled=False),
        DeviceManager(),
        private_replay=tmp_path / "archive",
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
            expected = {channel.channel: channel.iq.copy() for channel in result.channels}
        # Eviction and a later round must not change this delivery's identity.
        source.preparation_cache.max_bytes = 0
        source.readout_cache.clear()
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
    replayed = replay_round(replay_paths(tmp_path / "archive")[0])
    if isinstance(replayed, IntegratedRound):
        actual = replayed.raw.materialize(acquire.reps, acquire.avg_level)
    elif isinstance(replayed, RoundSignals):
        actual = {channel.channel: channel.iq for channel in replayed.channels}
    for channel, values in actual.items():
        np.testing.assert_array_equal(values, expected[channel])
    if isinstance(result, IntegratedRound) and isinstance(replayed, IntegratedRound):
        np.testing.assert_array_equal(
            result.statistics.channels[0].sums, replayed.statistics.channels[0].sums
        )


@pytest.mark.parametrize(
    "backend",
    [
        "ftf-direct-v1",
        "ftf-direct-v2",
        "ftf-segment-v6",
        "ftf-segment-v7",
        "ftf-segment-v10",
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
    env = ftf(parse_ftf(ftf_document()), private_replay=archive)
    try:
        program = ModularProgramV2(
            env.soccfg,
            ProgramV2Cfg(reps=1, rounds=1),
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

    source = FTFSource(settings(), DeviceManager(), private_replay=tmp_path / "archive")
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
