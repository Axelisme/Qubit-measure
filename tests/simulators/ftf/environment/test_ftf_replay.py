"""Author-opted persistent round replay through acquisition and detached inputs."""

import sys
from collections.abc import Sequence
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from joblib import Parallel
from qick import QickConfig
from zcu_tools.device import DeviceManager
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.program.v2.simulation import IntegratedRound, SimulationClock
from zcu_tools.simulate.boundary import (
    AcquireRequest,
    PulseEnvelope,
    RoundRequest,
    RoundSignals,
    RoundStatistics,
)
from zcu_tools.simulate.evolution import Engine, ModelSource, TimelineBatch
from zcu_tools.simulators.ftf.config import parse_ftf
from zcu_tools.simulators.ftf.environment import (
    FTFAcquisitionOptions,
    FTFSource,
    ftf,
    replay_round,
)
from zcu_tools.simulators.replay import read_replay, replay_paths, save_replay

from tests.core._simulation import ticks
from tests.program._profiles import profile_soccfg
from tests.program._timeline import compile_modules
from tests.simulators.ftf.config._data import ftf_document
from tests.simulators.ftf.environment._dimension_cases import (
    dimension_request,
    dimension_settings,
)
from tests.simulators.ftf.environment._templates import snapshot

pytestmark = pytest.mark.sim


def _sampled_request() -> AcquireRequest:
    from tests.simulators.ftf.environment._templates import request, template

    count = 8
    duration = ticks(0.5)
    base = template(tones=2)
    envelope = PulseEnvelope(ticks(0.125), (0j, 0.4 + 0j, 0.7 + 0j, 0.4 + 0j, 0j))
    drives = tuple(
        replace(
            event,
            start_ticks=np.zeros(count, dtype=np.int64),
            duration_ticks=np.full(count, duration),
            frequency_mhz=np.full(count, frequency),
            phase_deg=np.linspace(5.0, 85.0, count),
            gain=np.linspace(0.05, 0.2, count),
            envelopes=(envelope,),
            envelope_index=np.zeros(count, dtype=np.int64),
            active=np.ones(count, dtype=bool),
        )
        # Separated high carriers keep this sampled pair on DOP853 under
        # default family routing, so replay exercises its GPU/CPU price.
        for event, frequency in zip(base.events[:2], (5000.0, 5500.0), strict=True)
    )
    readout = base.events[-1]
    terminal = replace(
        readout,
        start_ticks=np.full(count, duration),
        duration_ticks=np.full(count, readout.duration_ticks[0]),
        frequency_mhz=np.full(count, readout.frequency_mhz[0]),
        phase_deg=np.full(count, readout.phase_deg[0]),
        gain=np.full(count, readout.gain[0]),
        envelope_index=np.zeros(count, dtype=np.int64),
        active=np.ones(count, dtype=bool),
    )
    adc = base.readouts[0]
    batch = replace(
        base,
        points=np.arange(count, dtype=np.int64),
        events=(*drives, terminal),
        readouts=(
            replace(
                adc,
                trigger_ticks=np.full((count, 1), duration),
                frequency_mhz=np.repeat(adc.frequency_mhz[:1], count, axis=0),
                phase_deg=np.repeat(adc.phase_deg[:1], count, axis=0),
                generator_channels=np.repeat(adc.generator_channels[:1], count, axis=0),
            ),
        ),
        duration_ticks=terminal.start_ticks + terminal.duration_ticks,
    )
    acquire = request()
    return replace(
        acquire,
        loop_dims=(acquire.reps, count),
        point_dims=(count,),
        templates=(batch,),
        readouts=(replace(acquire.readouts[0], trigger_ticks=(duration,)),),
    )


def _record_sampled_round(
    archive: Path, soccfg: QickConfig, engine: Engine
) -> tuple[Path, RoundStatistics]:
    from tests.simulators.ftf.environment._templates import expect_statistics, snapshot

    acquire = _sampled_request()
    with closing(
        FTFSource(
            parse_ftf(ftf_document()).with_telemetry(enabled=False),
            DeviceManager(),
            soccfg=soccfg,
            engine=engine,
            acquisition=FTFAcquisitionOptions(private_replay=archive),
        )
    ) as source:
        expected = expect_statistics(
            source.execute(RoundRequest(acquire, snapshot(), 0))
        )

    return replay_paths(archive)[0], expected


def test_cpu_round_replays_bitwise_without_probing_opencl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.simulators.ftf.environment._templates import expect_statistics

    soccfg = profile_soccfg("ftf")
    path, expected = _record_sampled_round(tmp_path, soccfg, Engine(2, backend="cpu"))
    versions = dict(read_replay(path).acquire.versions)
    assert versions["execution_device"] == "cpu"
    assert versions["workers"] == "2"
    assert "pyopencl" not in versions

    def unexpected_probe() -> None:
        raise AssertionError("CPU replay must not probe for an OpenCL device")

    # Any GPU discovery would consult this external loader boundary.
    monkeypatch.setitem(
        sys.modules,
        "pyopencl",
        SimpleNamespace(Error=RuntimeError, get_platforms=unexpected_probe),
    )
    actual = expect_statistics(replay_round(path, soccfg=soccfg))
    assert actual.state_counts.tobytes() == expected.state_counts.tobytes()
    assert actual.channels[0].sums.tobytes() == expected.channels[0].sums.tobytes()


@pytest.mark.requires_opencl
def test_gpu_round_replays_bitwise_at_recorded_worker_width_and_rejects_missing_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pyopencl as cl

    from tests.simulators.ftf.environment._templates import expect_statistics

    monkeypatch.setenv("ZCU_TOOLS_CACHE_DIR", str(tmp_path / "cache"))
    soccfg = profile_soccfg("ftf")
    path, expected = _record_sampled_round(tmp_path / "archive", soccfg, Engine(2))
    versions = dict(read_replay(path).acquire.versions)
    recorded = versions["execution_device"]
    assert recorded.startswith("opencl:")
    assert versions["workers"] == "2"
    assert "pyopencl" not in versions

    # I/Q need not change at every worker width. Observe the actual pool width
    # at the joblib boundary without replacing execution or device routing.
    pool_widths: list[int] = []
    enter = Parallel.__enter__

    def observe_pool(pool: Parallel) -> Parallel:
        pool_widths.append(pool.n_jobs)
        return enter(pool)

    monkeypatch.setattr(Parallel, "__enter__", observe_pool)
    actual = expect_statistics(replay_round(path, soccfg=soccfg))
    assert pool_widths and set(pool_widths) == {2}
    assert actual.state_counts.tobytes() == expected.state_counts.tobytes()
    assert actual.channels[0].sums.tobytes() == expected.channels[0].sums.tobytes()

    # Discovery is the external boundary that differs on a CPU-only machine.
    monkeypatch.setattr(cl, "get_platforms", list)
    with pytest.raises(ValueError, match="execution device mismatch") as mismatch:
        replay_round(path, soccfg=soccfg)
    assert f"recorded {recorded}, current cpu" in str(mismatch.value)


@pytest.mark.parametrize("mode", ["integrated", "decimated"])
def test_replay_retains_round_raw_after_source_release(tmp_path: Path, mode):
    from tests.simulators.ftf.environment._templates import request, settings, snapshot

    source = FTFSource(
        settings().with_telemetry(enabled=False),
        DeviceManager(),
        soccfg=profile_soccfg("ftf"),
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
    replayed = replay_round(archived_round, soccfg=profile_soccfg("ftf"))
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
    from tests.simulators.ftf.environment._templates import request, settings, snapshot

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
        soccfg=profile_soccfg("ftf"),
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
    replayed = replay_round(archived_round, soccfg=profile_soccfg("ftf"))
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
        "ftf-segment-v13",
    ],
)
def test_replay_rejects_archive_from_obsolete_numeric_backend(
    tmp_path: Path,
    backend: str,
) -> None:
    from zcu_tools.simulators.ftf.environment import replay_round
    from zcu_tools.simulators.replay import read_replay, replay_paths, save_replay

    archive = tmp_path / "current"
    env = ftf(
        parse_ftf(ftf_document()),
        soccfg=profile_soccfg("ftf"),
        acquisition=FTFAcquisitionOptions(private_replay=archive),
    )
    try:
        program = compile_modules(
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
        if name != "selected_dimension"
    )
    legacy = saved.acquire.model_copy(update={"versions": old_versions})
    path = save_replay(tmp_path / "legacy", legacy, saved.round.snapshot, 0)
    with pytest.raises(ValueError, match="version mismatch"):
        replay_round(path, soccfg=profile_soccfg("ftf"))


def _integrated(result: IntegratedRound | RoundSignals) -> IntegratedRound:
    if not isinstance(result, IntegratedRound):
        raise AssertionError("integrated FTF replay must retain raw shots")
    return result


@pytest.fixture
def evolution_sizes(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int]]:
    # Replay returns I/Q, not its Engine inputs. Observe the real call
    # boundary to distinguish an identical draw from repeating the pilot.
    calls: list[tuple[int, int]] = []
    evolve = Engine.evolve

    def observe(
        engine: Engine, model: ModelSource, batches: Sequence[TimelineBatch], **kwargs
    ):
        calls.append(
            (len(model.initial_density), sum(len(batch.points) for batch in batches))
        )
        return evolve(engine, model, batches, **kwargs)

    monkeypatch.setattr(Engine, "evolve", observe)
    return calls


def _assert_integrated_bits(
    actual: IntegratedRound, expected: IntegratedRound, acquire: AcquireRequest
) -> None:
    assert (
        actual.statistics.state_counts.tobytes()
        == expected.statistics.state_counts.tobytes()
    )
    for obtained, recorded in zip(
        actual.statistics.channels, expected.statistics.channels, strict=True
    ):
        assert obtained.channel == recorded.channel
        assert obtained.sums.tobytes() == recorded.sums.tobytes()
    actual_raw = actual.raw.materialize(acquire.reps, acquire.avg_level)
    expected_raw = expected.raw.materialize(acquire.reps, acquire.avg_level)
    assert actual_raw.keys() == expected_raw.keys()
    for channel in actual_raw:
        assert actual_raw[channel].tobytes() == expected_raw[channel].tobytes()


@pytest.mark.parametrize("index", [0, 1], ids=["first", "later"])
def test_replay_keeps_round_shape_and_acquire_dimension_across_dc(
    tmp_path: Path, evolution_sizes: list[tuple[int, int]], index: int
) -> None:
    soccfg = profile_soccfg("ftf")
    settings = dimension_settings(16).with_telemetry(enabled=False)
    settings = settings.model_copy(
        update={"chip": settings.chip.model_copy(update={"temperature_k": 0.053})}
    )
    acquire = dimension_request()
    before, after = snapshot(-1.0), snapshot(1.0)
    archive = tmp_path / "archive"
    with closing(
        FTFSource(
            settings, DeviceManager(), soccfg=soccfg, engine=Engine(1, backend="cpu")
        )
    ) as independent_source:
        independent = independent_source.open_acquire(acquire)
        independent.sample_round(after, 0)
        assert independent.dimension is not None
        independent_n = independent.dimension.dimension

    with closing(
        FTFSource(
            settings,
            DeviceManager(),
            soccfg=soccfg,
            engine=Engine(1, backend="cpu"),
            acquisition=FTFAcquisitionOptions(private_replay=archive),
        )
    ) as source:
        session = source.open_acquire(acquire)
        first = _integrated(session.sample_round(before, 0))
        assert session.dimension is not None
        saved_n = session.dimension.dimension
        assert saved_n < independent_n < settings.numerics.truncation.retained_levels
        second = _integrated(session.sample_round(after, 1))
    (target,) = [
        path for path in replay_paths(archive) if read_replay(path).round.index == index
    ]

    evolution_sizes.clear()
    actual = _integrated(replay_round(target, soccfg=soccfg))
    expected_sizes = [(16, 1), (saved_n, 7)] if index == 0 else [(saved_n, 8)]
    assert evolution_sizes == expected_sizes
    _assert_integrated_bits(actual, (first, second)[index], acquire)


def test_replay_first_round_at_upper_recomputes_the_recorded_split(
    tmp_path: Path, evolution_sizes: list[tuple[int, int]]
) -> None:
    soccfg = profile_soccfg("ftf")
    settings = dimension_settings(16).with_telemetry(enabled=False)
    settings = settings.model_copy(
        update={"chip": settings.chip.model_copy(update={"temperature_k": 0.3})}
    )
    acquire = dimension_request()
    with closing(
        FTFSource(
            settings,
            DeviceManager(),
            soccfg=soccfg,
            engine=Engine(1, backend="cpu"),
            acquisition=FTFAcquisitionOptions(private_replay=tmp_path),
        )
    ) as source:
        expected = _integrated(source.open_acquire(acquire).sample_round(snapshot(), 0))
    assert evolution_sizes == [(16, 1), (16, 7)]
    (path,) = replay_paths(tmp_path)

    evolution_sizes.clear()
    actual = _integrated(replay_round(path, soccfg=soccfg))
    assert evolution_sizes == [(16, 1), (16, 7)]
    _assert_integrated_bits(actual, expected, acquire)


@pytest.mark.parametrize(
    ("selected", "manifest", "error"),
    [
        (None, None, "missing selected dimension"),
        ("3", None, "selected dimension exceeds"),
        ("2", "ftf-v14:pilot-point:2", "pilot point.*out of range"),
    ],
)
def test_v14_replay_rejects_invalid_dimension_or_pilot_point(
    tmp_path: Path, selected: str | None, manifest: str | None, error: str
) -> None:
    from tests.simulators.ftf.environment._templates import request, settings

    soccfg = profile_soccfg("ftf")
    archive = tmp_path / "original"
    with closing(
        FTFSource(
            settings().with_telemetry(enabled=False),
            DeviceManager(),
            soccfg=soccfg,
            engine=Engine(1, backend="cpu"),
            acquisition=FTFAcquisitionOptions(private_replay=archive),
        )
    ) as source:
        source.open_acquire(request()).sample_round(snapshot(), 0)
    (original,) = replay_paths(archive)
    saved = read_replay(original)
    versions = tuple(
        (key, "ftf-segment-v14" if key == "backend" else value)
        for key, value in saved.acquire.versions
        if key != "selected_dimension"
    )
    if selected is not None:
        versions = (*versions, ("selected_dimension", selected))
    path = save_replay(
        tmp_path / "modified",
        saved.acquire.model_copy(update={"versions": versions}),
        saved.round.snapshot,
        saved.round.index,
        numeric_manifest=manifest,
    )
    with pytest.raises(ValueError, match=error):
        replay_round(path, soccfg=soccfg)


def test_replay_recomputes_statistics_without_numeric_records(tmp_path: Path) -> None:
    from zcu_tools.device import DeviceManager
    from zcu_tools.simulate.boundary import RoundRequest
    from zcu_tools.simulators.ftf.environment import FTFSource, replay_round
    from zcu_tools.simulators.replay import read_replay, replay_paths, save_replay

    from tests.simulators.ftf.environment._templates import (
        expect_statistics,
        request,
        settings,
        snapshot,
    )

    source = FTFSource(
        settings(),
        DeviceManager(),
        soccfg=profile_soccfg("ftf"),
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
    actual = expect_statistics(replay_round(paths[0], soccfg=profile_soccfg("ftf")))
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
            replay_round(mismatch, soccfg=profile_soccfg("ftf"))
