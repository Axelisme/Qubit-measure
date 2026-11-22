"""Managed and direct FTF/single acquires retire deterministic ADC work."""

from dataclasses import replace
from importlib import import_module
from threading import Event

import pytest
from zcu_tools.program.acquisition import StoppedPartialAcquireError
from zcu_tools.program.v2 import TimelineBuilder
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.program.v2.simulation import AcquireOutcome, IntegratedRound
from zcu_tools.program.v2.simulation.compiled import build_compiled_request
from zcu_tools.simulate.adc import ReadoutPulse
from zcu_tools.simulate.boundary import RoundRequest
from zcu_tools.simulators.telemetry import TelemetrySettings

from tests.program._profiles import profile_soccfg
from tests.simulators._engine import EngineDouble, controlled_engine
from tests.simulators.ftf.environment._templates import settings as ftf_settings
from tests.simulators.single_fluxonium.environment._inputs import (
    settings as single_settings,
)

pytestmark = pytest.mark.sim


@pytest.fixture(params=["ftf", "single_fluxonium"])
def raw_environment(request, monkeypatch):
    name = request.param
    runtime = import_module(f"zcu_tools.simulators.{name}.environment.runtime")
    config = ftf_settings() if name == "ftf" else single_settings()
    config = config.model_copy(
        update={
            "numerics": config.numerics.model_copy(
                update={
                    "debug_capture": False,
                    "telemetry": TelemetrySettings(enabled=False),
                    "time_scale": 0,
                }
            )
        }
    )
    source_name = "FTFSource" if name == "ftf" else "SingleFluxoniumSource"
    source_type = getattr(runtime, source_name)
    sources = []

    def build_source(*args, **kwargs):
        source = source_type(*args, **kwargs)
        sources.append(source)
        return source

    # The environment factory does not return the source for direct callers.
    monkeypatch.setattr(runtime, source_name, build_source)
    monkeypatch.setattr(runtime, "Engine", EngineDouble)
    env = getattr(runtime, name)(config, soccfg=profile_soccfg(name))
    try:
        builder = TimelineBuilder(
            env.soccfg, reps=3, initial_delay=0.016, relax_delay=0
        )
        readout = PulseReadoutCfg(
            gen_ch=0,
            ro_ch=0,
            nqz=1,
            freq=6000 if name == "ftf" else 7000,
            gain=0.1,
            waveform=ConstWaveformCfg(length=0.016),
            ro_length=0.004,
        )
        builder.emit(readout.build("readout"))
        (source,) = sources
        yield env, source, builder.build()
    finally:
        if env.close_backend is not None:
            env.close_backend()
        env.devices.close_all_devices()


class IncidentWork:
    """Count deterministic pulse evaluations, optionally requesting Stop during raw."""

    def __init__(self):
        self.calls = 0
        self.stop: Event | None = None


@pytest.fixture
def incident_work(monkeypatch):
    work = IncidentWork()
    incident = ReadoutPulse.incident

    def observed(pulse: ReadoutPulse, time_ns: float, reference_frequency_ghz: float):
        work.calls += 1
        if work.stop is not None:
            work.stop.set()
            work.stop = None
        return incident(pulse, time_ns, reference_frequency_ghz)

    # Production programs use compiled pulses, not injectable opaque ADC envelopes.
    monkeypatch.setattr(ReadoutPulse, "incident", observed)
    return work


@pytest.mark.parametrize("ending", ["finish", "begin", "open", "execute", "close"])
def test_direct_acquire_retires_the_plan_without_losing_delivered_raw(
    raw_environment, incident_work, ending
):
    env, source, program = raw_environment
    acquire = build_compiled_request(
        program, "integrated", env.soc.binding, 0, routes=((0, 0),)
    )
    first = RoundRequest(acquire, source.snapshot(), 0)
    previous = source.execute(first)
    expected = shot_bytes(previous, first)
    next_round = replace(first, index=1)
    following = source.execute(next_round)
    before = incident_work.calls
    shot_bytes(following, next_round)
    assert incident_work.calls == before, "same acquire reran noiseless raw setup"

    replacement = replace(first, acquire=replace(acquire))
    engine = controlled_engine(source.engine)
    evolutions = len(engine.calls)
    if ending == "finish":
        nonmatching = AcquireOutcome((replacement,), 0.0, 0.0, True, "other")
        source.finish_acquire(nonmatching)
        assert shot_bytes(previous, first) == expected
        assert incident_work.calls == before, "unrelated finish retired the live plan"
        source.finish_acquire(replace(nonmatching, rounds=(first, next_round)))
    elif ending == "begin":
        source.begin_acquire()
    elif ending == "close":
        source.close()
    else:
        if ending == "open":
            source.open_acquire(replacement.acquire)
        current = source.execute(replacement)
        assert len(engine.calls) == evolutions, "expected a preparation hit"
        before = incident_work.calls
        assert shot_bytes(current, replacement) == expected
        assert incident_work.calls > before, "a new acquire borrowed the old plan"

    before = incident_work.calls
    assert shot_bytes(previous, first) == expected
    assert incident_work.calls > before, f"{ending} left the previous plan live"


def shot_bytes(result, request: RoundRequest) -> dict[int, bytes]:
    if not isinstance(result, IntegratedRound):
        raise AssertionError("fixture must execute integrated rounds")
    arrays = result.raw.materialize(request.acquire.reps, request.acquire.avg_level)
    return {channel: raw.tobytes() for channel, raw in arrays.items()}


@pytest.mark.parametrize("ending", ["normal", "stop", "exception"])
def test_managed_acquire_retires_the_plan_with_diagnostics_off(
    raw_environment, incident_work, ending, monkeypatch
):
    env, _source, program = raw_environment
    binding = env.soc.binding
    execute = binding.execute
    cancel = Event()
    attempted = []
    published = []

    def observe(request: RoundRequest):
        result = execute(request)
        attempted.append((result, request, incident_work.calls))
        if ending == "stop":
            incident_work.stop = cancel
        return result

    def hook(_round, *_args):
        data = program.get_raw()
        assert data is not None
        published.append(data[0].tobytes())
        if ending == "exception":
            raise RuntimeError("processor failed after raw")

    monkeypatch.setattr(binding, "execute", observe)

    def acquire():
        return program.acquire(
            env.soc,
            threshold=0.0,
            remove_offset=False,
            progress=False,
            rounds=2,
            cancel_flag=cancel,
            round_hook=hook,
        )

    if ending == "stop":
        with pytest.raises(StoppedPartialAcquireError):
            acquire()
        assert cancel.is_set()
        assert program.get_raw() is None
        assert not published
    elif ending == "exception":
        with pytest.raises(RuntimeError, match="processor failed after raw"):
            acquire()
        assert len(published) == 1
    else:
        acquire()
        assert len(published) == 2
        # Only the first round did deterministic work inside materialize.
        assert incident_work.calls == attempted[1][2]

    assert len(attempted) == (2 if ending == "normal" else 1)
    assert incident_work.calls > attempted[0][2], "first raw never ran its plan"
    result, request, _ = attempted[-1]
    before = incident_work.calls
    later = shot_bytes(result, request)
    assert incident_work.calls > before, f"managed {ending} left the plan live"
    if published:
        assert later == {0: published[-1]}
