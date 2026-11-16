"""Acquire completion observers run once and cannot break acquisition admission."""

import time
import warnings
from dataclasses import FrozenInstanceError
from threading import Event
from fractions import Fraction

import pytest
from zcu_tools.core.simulation import (
    AcquireRequest,
    EnvironmentSnapshot,
    RoundSignals,
    TimeBase,
)
from zcu_tools.experiment.simulation import white_noise
from zcu_tools.program.acquisition import StoppedPartialAcquireError
from zcu_tools.program.v2 import TimelineBuilder
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.program.v2.simulation import (
    AcquireOutcome,
    SimulationBinding,
    SimulationCallbacks,
    SimulationSoc,
)
from zcu_tools.program.v2.simulation.compiled import build_compiled_request

from tests.program._profiles import load_soccfg


@pytest.fixture
def program_seam():
    env = white_noise(load_soccfg(), std_adc=0, seed=1, routes=((0, 0),))
    observed: list[AcquireOutcome] = []
    binding = SimulationBinding(
        env.soc.binding.capture_snapshot,
        env.soc.binding.execute,
        backend_version="synthetic",
        callbacks=SimulationCallbacks(on_end=observed.append),
    )
    soc = SimulationSoc(env.soccfg, binding, routes=((0, 0),))
    builder = TimelineBuilder(env.soccfg, reps=1)
    builder.emit(PulseReadoutCfg(
        gen_ch=0, ro_ch=0, nqz=1, freq=100, gain=0.1,
        waveform=ConstWaveformCfg(length=0.1), ro_length=0.1,
    ).build("readout"))
    try:
        yield builder.build(), soc, observed
    finally:
        env.devices.close_all_devices()


def test_program_observes_only_builder_time_after_cleanup(program_seam, monkeypatch):
    program, soc, observed = program_seam
    now = [10.0]
    # Acquire has no public clock injection; control monotonic and delegate the builder.
    monkeypatch.setattr(time, "monotonic", lambda: now[0])

    def build(*args, **kwargs):
        try:
            return build_compiled_request(*args, **kwargs)
        finally:
            now[0] += 2.0

    install = soc.install_request
    release = soc.release_request

    def install_delayed(*args, **kwargs):
        install(*args, **kwargs)
        now[0] += 3.0

    def release_delayed():
        release()
        now[0] += 5.0

    # These public collaborators expose no timing injection; preserve their real work.
    monkeypatch.setattr("zcu_tools.program.v2.simulation.compiled.build_compiled_request", build)
    monkeypatch.setattr(soc, "install_request", install_delayed)
    monkeypatch.setattr(soc, "release_request", release_delayed)
    result = program.acquire(soc, progress=False)
    assert len(result) == 1
    assert len(observed) == 1
    outcome = observed[0]
    assert outcome.compile_wall_s == 2.0
    assert outcome.wall_s == 10.0
    assert outcome.complete is True
    assert len(outcome.rounds) == 1
    assert outcome.rounds[0].acquire.sequence == 0
    with pytest.raises(FrozenInstanceError):
        setattr(outcome, "complete", False)


def test_binding_reports_compile_time_and_normal_return(monkeypatch):
    observed = []
    now = [10.0]
    # Acquire timing has no public clock injection; control its monotonic collaborator.
    monkeypatch.setattr("time.monotonic", lambda: now[0])
    binding = SimulationBinding(
        lambda: EnvironmentSnapshot("synthetic", "test", ()),
        lambda _: RoundSignals(()),
        backend_version="test",
        callbacks=SimulationCallbacks(on_end=observed.append),
    )
    with binding.acquire() as sequence:
        assert sequence == 0
        with binding.compiling():
            now[0] += 2.0
        now[0] += 3.0
    assert len(observed) == 1
    outcome = observed[0]
    assert outcome.rounds == ()
    assert outcome.wall_s == 5.0
    assert outcome.compile_wall_s == 2.0
    assert outcome.complete is True


def test_completion_observer_sees_all_rounds_and_cannot_mask_execution_failure():
    observed = []

    def finish(outcome):
        observed.append(outcome)
        raise OSError("observer cannot write")

    def execute(request):
        if request.index == 1:
            raise ValueError("original execution failure")
        return RoundSignals(())

    try:
        from zcu_tools.program.v2.simulation import SimulationCallbacks

        binding = SimulationBinding(
            lambda: EnvironmentSnapshot("synthetic", "test", ()),
            execute,
            backend_version="test",
            seed=1,
            callbacks=SimulationCallbacks(on_end=finish),
        )
    except ImportError as error:
        raise AssertionError("binding must notify acquire completion") from error
    def acquire():
        with binding.acquire() as sequence:
            request = AcquireRequest(
                "decimated", (1,), 0, (), 1, sequence, binding.rng_version,
                time_base=TimeBase(Fraction(1)),
            )
            binding.sample_round(request, 0)
            binding.sample_round(request, 1)

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="original execution failure"):
            acquire()
    assert len(observed) == 1
    assert len(observed[0].rounds) == 2
    assert type(observed[0].wall_s) is float
    assert observed[0].complete is False
    with binding.acquire():
        pass
    assert len(observed) == 2
    assert observed[1].rounds == ()
