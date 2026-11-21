"""Three environments publish one capture at the admitted acquire boundary."""

import gzip
import json
from contextlib import contextmanager
from importlib import import_module

import pytest
from zcu_tools.cfg_model import MuxReadoutTone
from zcu_tools.device import DeviceManager, FakeDevice
from zcu_tools.program.v2 import TimelineBuilder
from zcu_tools.program.v2.modules.readout import MuxReadoutCfg, PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.program.v2.simulation import AcquireOutcome
from zcu_tools.simulate.boundary import RoundRequest
from zcu_tools.simulate.evolution import capture_directory, read_frozen_manifest
from zcu_tools.simulators.ftf.environment import FTFSource
from zcu_tools.simulators.six_fluxonium.config import parse_six_fluxonium
from zcu_tools.simulators.telemetry import RoutingTelemetry, TelemetrySettings

from tests.program._profiles import profile_soccfg
from tests.simulators._engine import EngineDouble
from tests.simulators.ftf.environment._templates import request, snapshot
from tests.simulators.ftf.environment._templates import settings as ftf_settings
from tests.simulators.single_fluxonium.environment._inputs import (
    settings as single_settings,
)
from tests.simulators.six_fluxonium.config._data import six_document

pytestmark = pytest.mark.sim


@pytest.fixture(params=["ftf", "single_fluxonium", "six_fluxonium"])
def capture_environment(request, tmp_path, monkeypatch):
    name = request.param
    monkeypatch.setenv("ZCU_TOOLS_CACHE_DIR", str(tmp_path / "cache"))
    runtime = import_module(f"zcu_tools.simulators.{name}.environment.runtime")
    engines = []

    def engine_factory(*args, **kwargs):
        engine = EngineDouble(*args, **kwargs)
        engine.reuse_allowed = False
        engines.append(engine)
        return engine

    monkeypatch.setattr(runtime, "Engine", engine_factory)
    if name == "ftf":
        config = ftf_settings()
    elif name == "single_fluxonium":
        config = single_settings()
    else:
        config = parse_six_fluxonium(six_document())

    @contextmanager
    def build(*, capture: bool | None = True, telemetry=True, fresh_seed=False):
        updates: dict[str, object] = {
            "telemetry": TelemetrySettings(
                enabled=telemetry, directory=tmp_path / "telemetry"
            )
        }
        if capture is not None:
            updates["debug_capture"] = capture
        if name != "six_fluxonium":
            updates["time_scale"] = 0
        if fresh_seed:
            updates["seed"] = None
        configured = config.model_copy(
            update={"numerics": config.numerics.model_copy(update=updates)}
        )
        env = getattr(runtime, name)(configured, soccfg=profile_soccfg(name))
        try:
            yield env, engines
        finally:
            if env.close_backend is not None:
                env.close_backend()
            env.devices.close_all_devices()

    return name, build


def capture_program(name, env):
    builder = TimelineBuilder(env.soccfg, reps=2, initial_delay=0.016, relax_delay=0)
    if name != "six_fluxonium":
        readout = PulseReadoutCfg(
            gen_ch=0,
            ro_ch=0,
            nqz=1,
            freq=6000 if name == "ftf" else 7000,
            gain=0.1,
            waveform=ConstWaveformCfg(length=0.016),
            ro_length=0.004,
        )
    else:
        frequencies = [6800 + 125 * i for i in range(6)]
        readout = MuxReadoutCfg(
            gen_ch=0,
            nqz=1,
            mixer_freq=7100,
            length=0.064,
            ro_length=0.026,
            tones=[
                MuxReadoutTone(ro_ch=i, freq=f, gain=0.1)
                for i, f in enumerate(frequencies)
            ],
        )
    with builder.repeat("reads", 2):
        builder.emit(readout.build("readout"))
        builder.timeline_inc(0.096)
    builder.delay_by_timeline()
    return builder.build()


def metadata_of(record):
    """Narrow opaque codec metadata at the capture consumer seam."""
    if not isinstance(record.metadata, dict):
        raise AssertionError("capture metadata must be a mapping")
    return record.metadata


def telemetry_records(tmp_path):
    return [
        json.loads(line)
        for path in (tmp_path / "telemetry").glob("routing-*.jsonl")
        for line in path.read_text().splitlines()
    ]


@pytest.mark.parametrize(
    "capture_environment", ["ftf", "single_fluxonium"], indirect=True
)
@pytest.mark.parametrize("fail_second_round", [False, True])
def test_capture_records_effective_settings_and_each_attempt(
    capture_environment, fail_second_round, tmp_path
):
    name, build = capture_environment
    with build(telemetry=False) as (env, engines):
        source_name = "bias" if name == "ftf" else "single_test_bias"
        bias = env.create_device("FakeDevice", source_name)
        bias.set_value(0.1)
        identity = env.soc.binding.capture_snapshot().model_identity
        program = capture_program(name, env)

        def next_round(*_args, **_kwargs):
            bias.set_value(0.3)
            if fail_second_round:
                engines[0].error = RuntimeError("second round failed")

        if fail_second_round:
            with pytest.raises(RuntimeError, match="second round failed"):
                program.acquire(
                    env.soc, rounds=2, round_hook=next_round, progress=False
                )
        else:
            program.acquire(env.soc, rounds=2, round_hook=next_round, progress=False)

    path = next(capture_directory().glob("capture-*.json.gz"))
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        metadata = json.load(stream)["metadata"]
    assert metadata["complete"] is not fail_second_round
    assert metadata["effective_settings"][identity]["numerics"]["seed"] == 123
    assert metadata["effective_settings"][identity]["numerics"]["time_scale"] == 0
    assert metadata["sampling"] == {"root_seed": 123, "sequence": 0}
    assert metadata["rounds"] == [
        {"index": 0, "dc_values": {source_name: 0.1}, "model_identity": identity},
        {"index": 1, "dc_values": {source_name: 0.3}, "model_identity": identity},
    ]
    assert telemetry_records(tmp_path) == []


@pytest.mark.parametrize(
    "capture_environment", ["ftf", "single_fluxonium"], indirect=True
)
@pytest.mark.parametrize("rounds", [1, 2])
def test_capture_matches_calibration_used_in_each_round(
    capture_environment, rounds, monkeypatch
):
    name, build = capture_environment
    runtime = import_module(f"zcu_tools.simulators.{name}.environment.runtime")
    source_name = "FTFSource" if name == "ftf" else "SingleFluxoniumSource"
    source_type = getattr(runtime, source_name)
    sources = []

    def observe_source(*args, **kwargs):
        source = source_type(*args, **kwargs)
        sources.append(source)
        return source

    # The builder does not return the source author API for changing settings.
    monkeypatch.setattr(runtime, source_name, observe_source)
    with build(telemetry=False, fresh_seed=True) as (env, _engines):
        (source,) = sources
        binding = env.soc.binding
        identities = [source.identity]
        offsets = [0.3, 0.32, 0.34] if name == "ftf" else [0.5, 0.52, 0.54]

        def reconfigure(*_args):
            dc = source.settings.dc_bias
            offset = offsets[len(identities)]
            if name == "ftf":
                offset = dc.offset_phi0.model_copy(update={"f1": offset})
            source.settings = source.settings.model_copy(
                update={"dc_bias": dc.model_copy(update={"offset_phi0": offset})}
            )
            identities.append(source.identity)

        capture_program(name, env).acquire(
            env.soc, rounds=rounds, round_hook=reconfigure, progress=False
        )
        root_seed = binding.root_seed

    path = next(capture_directory().glob("capture-*.json.gz"))
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        metadata = json.load(stream)["metadata"]
    assert metadata["sampling"] == {"root_seed": root_seed, "sequence": 0}
    assert len(identities) == rounds + 1  # Includes the unused final reconfiguration.
    assert [row["model_identity"] for row in metadata["rounds"]] == identities[:-1]
    settings = metadata["effective_settings"]
    assert set(settings) == set(identities[:-1]), (
        f"expected only settings used by recorded rounds, got {settings.get('dc_bias')}"
    )
    for identity, offset in zip(identities[:-1], offsets[:rounds], strict=True):
        assert settings[identity]["numerics"]["seed"] is None
        captured = settings[identity]["dc_bias"]["offset_phi0"]
        assert (captured["f1"] if name == "ftf" else captured) == offset


def test_acquire_capture_is_shared_and_published_before_telemetry(
    capture_environment, tmp_path
):
    name, build = capture_environment
    with build() as (env, engines):
        program = capture_program(name, env)
        kwargs = {}
        if name == "ftf":
            bias = env.create_device("FakeDevice", "bias")
            kwargs["round_hook"] = lambda *_args, **_kwargs: bias.set_value(0.3)
        assert program.acquire(env.soc, rounds=2, progress=False, **kwargs) is not None
        files = list(capture_directory().glob("capture-*.json.gz"))
        assert len(files) == 1
        records = read_frozen_manifest(files[0])
        buffers = [buffer for engine in engines for buffer in engine.captures]
        assert len(buffers) > 1
        assert buffers[0] is not None
        assert all(buffer is buffers[0] for buffer in buffers)
        assert len(records) == len(buffers[0].records)
        assert all(metadata_of(record)["complete"] for record in records)
        summary = telemetry_records(tmp_path)
        assert len(summary) == 1
        assert summary[0]["capture_file"] == files[0].name
        assert summary[0]["complete"] is True
        assert program.acquire(env.soc, progress=False) is not None
        assert len(list(capture_directory().glob("capture-*.json.gz"))) == 2
        assert telemetry_records(tmp_path)[1]["capture_file"] != files[0].name


def test_direct_ftf_finish_publishes_capture_and_telemetry_with_detached_round(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("ZCU_TOOLS_CACHE_DIR", str(tmp_path / "cache"))
    config = ftf_settings()
    config = config.model_copy(
        update={"numerics": config.numerics.model_copy(update={"debug_capture": True})}
    )
    devices = DeviceManager()
    bias = FakeDevice(fast_mode=True, value_unit=config.dc_bias.sources["bias"].unit)
    bias.set_value(0.3)
    devices.register_device("bias", bias)
    source = FTFSource(
        config,
        devices,
        soccfg=profile_soccfg("ftf"),
        engine=EngineDouble(),
        telemetry=RoutingTelemetry(tmp_path / "telemetry"),
    )
    try:
        acquire = request()
        detached = snapshot(0.2)
        session = source.open_acquire(acquire)
        session.sample_round(detached, 0)
        observed = source.snapshot()
        session.sample_round(observed, 1)
        source.finish_acquire(
            AcquireOutcome(
                (
                    RoundRequest(acquire, detached, 0),
                    RoundRequest(acquire, observed, 1),
                ),
                wall_s=7.0,
                compile_wall_s=0.0,
                complete=True,
                acquire_id="fixture-acquire",
            )
        )
    finally:
        source.close()
        devices.close_all_devices()

    (path,) = capture_directory().glob("capture-*.json.gz")
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        manifest = json.load(stream)
    assert manifest["inputs"]
    metadata = manifest["metadata"]
    assert metadata["rounds"] == [
        {
            "index": 0,
            "dc_values": {"bias": 0.2},
            "model_identity": detached.model_identity,
        },
        {
            "index": 1,
            "dc_values": {"bias": 0.3},
            "model_identity": observed.model_identity,
        },
    ]
    assert set(metadata["effective_settings"]) == {observed.model_identity}
    assert metadata["admission_settings"] is None
    (summary,) = telemetry_records(tmp_path)
    assert summary["capture_file"] == path.name
    assert summary["rounds"] == 2


@pytest.mark.parametrize("telemetry", [False, True])
def test_capture_without_telemetry_and_default_off(
    capture_environment, tmp_path, telemetry
):
    name, build = capture_environment
    with build(telemetry=telemetry) as (env, _engines):
        assert capture_program(name, env).acquire(env.soc, progress=False) is not None
    assert len(list(capture_directory().glob("capture-*.json.gz"))) == 1
    assert len(telemetry_records(tmp_path)) == int(telemetry)
    with build(capture=None, telemetry=telemetry) as (env, engines):
        assert capture_program(name, env).acquire(env.soc, progress=False) is not None
        assert engines[-1].captures[-1] is None
    assert len(list(capture_directory().glob("capture-*.json.gz"))) == 1
    if telemetry:
        assert telemetry_records(tmp_path)[-1]["capture_file"] is None


def test_failed_evolve_is_recorded_and_publication_failure_is_null(
    capture_environment, tmp_path
):
    name, build = capture_environment
    with build() as (env, engines):
        for engine in engines:
            engine.error = RuntimeError("evolution failed")
        with pytest.raises(RuntimeError, match="evolution failed"):
            capture_program(name, env).acquire(env.soc, progress=False)
    path = next(capture_directory().glob("capture-*.json.gz"))
    records = read_frozen_manifest(path)
    assert records and all(
        metadata_of(record)["complete"] is False for record in records
    )
    assert telemetry_records(tmp_path)[0]["complete"] is False
    assert telemetry_records(tmp_path)[0]["capture_file"] == path.name
    path.unlink()
    path.parent.rmdir()
    path.parent.write_text("blocked publication")
    with (
        build() as (env, _engines),
        pytest.warns(RuntimeWarning, match="capture could not be written") as seen,
    ):
        assert capture_program(name, env).acquire(env.soc, progress=False) is not None
    assert len(seen) == 1
    assert telemetry_records(tmp_path)[-1]["capture_file"] is None


def test_compile_failure_writes_empty_capture_but_begin_failure_writes_nothing(
    capture_environment, tmp_path, monkeypatch
):
    name, build = capture_environment
    with build(telemetry=False) as (env, _engines):
        with (
            pytest.raises(RuntimeError, match="compile failed"),
            env.soc.binding.acquire(),
            env.soc.binding.compiling(),
        ):
            raise RuntimeError("compile failed")
        path = next(capture_directory().glob("capture-*.json.gz"))
        assert read_frozen_manifest(path) == ()
        if name != "six_fluxonium":
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                metadata = json.load(stream)["metadata"]
            assert not metadata["complete"]
            assert metadata["rounds"] == []
            assert metadata["sampling"] is None
            assert metadata["effective_settings"] == {}
            assert metadata["admission_settings"]["numerics"]["seed"] == 123
        path.unlink()
        runtime = import_module(f"zcu_tools.simulators.{name}.environment.runtime")
        source_type = {
            "ftf": "FTFSource",
            "single_fluxonium": "SingleFluxoniumSource",
            "six_fluxonium": "SixFluxoniumSource",
        }[name]

        def fail_begin(*_args, **_kwargs):
            raise RuntimeError("begin failed")

        monkeypatch.setattr(getattr(runtime, source_type), "begin_acquire", fail_begin)
        with (
            pytest.raises(RuntimeError, match="begin failed"),
            env.soc.binding.acquire(),
        ):
            pytest.fail("failed admission must not enter the body")
        assert list(capture_directory().iterdir()) == []
    assert telemetry_records(tmp_path) == []


def test_later_acquire_failure_preserves_completed_evolves(
    capture_environment, tmp_path
):
    name, build = capture_environment
    with build() as (env, _engines):
        program = capture_program(name, env)

        def fail_after_round(*_args, **_kwargs):
            raise RuntimeError("after evolution")

        with pytest.raises(RuntimeError, match="after evolution"):
            program.acquire(
                env.soc, rounds=2, progress=False, round_hook=fail_after_round
            )
    path = next(capture_directory().glob("capture-*.json.gz"))
    records = read_frozen_manifest(path)
    assert records and all(metadata_of(record)["complete"] for record in records)
    assert telemetry_records(tmp_path)[0]["complete"] is False
