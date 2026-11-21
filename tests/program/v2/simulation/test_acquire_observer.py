"""Acquire completion observers run once and cannot break acquisition admission."""

import time
import warnings
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
from threading import Event

import numpy as np
import pytest
from zcu_tools.experiment.simulation import white_noise
from zcu_tools.program.acquisition import CancelFlagProtocol, StoppedPartialAcquireError
from zcu_tools.program.v2 import TimelineBuilder
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.program.v2.simulation import (
    AcquireOutcome,
    IntegratedRound,
    SimulationBinding,
    SimulationCallbacks,
    SimulationSoc,
)
from zcu_tools.program.v2.simulation.compiled import build_compiled_request
from zcu_tools.simulate.boundary import (
    AcquireRequest,
    ChannelSignals,
    EnvironmentSnapshot,
    RoundRequest,
    RoundSignals,
    TimeBase,
)
from zcu_tools.simulate.evolution import Cancelled

from tests.program._profiles import load_soccfg


def synthetic_binding(
    callbacks: SimulationCallbacks | None = None,
) -> SimulationBinding:
    """A clock-free, zero-signal backend for binding lifecycle contracts."""
    return SimulationBinding(
        lambda: EnvironmentSnapshot("synthetic", "test", ()),
        lambda _: RoundSignals(()),
        backend_version="test",
        callbacks=callbacks,
    )


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
    builder.emit(
        PulseReadoutCfg(
            gen_ch=0,
            ro_ch=0,
            nqz=1,
            freq=100,
            gain=0.1,
            waveform=ConstWaveformCfg(length=0.1),
            ro_length=0.1,
        ).build("readout")
    )
    try:
        yield builder.build(), soc, observed
    finally:
        env.devices.close_all_devices()


@pytest.mark.parametrize("mode", ["integrated", "decimated"])
def test_program_observes_only_builder_time_after_cleanup(
    program_seam, monkeypatch, mode
):
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
    monkeypatch.setattr(
        "zcu_tools.program.v2.simulation.compiled.build_compiled_request", build
    )
    monkeypatch.setattr(soc, "install_request", install_delayed)
    monkeypatch.setattr(soc, "release_request", release_delayed)
    acquire = program.acquire if mode == "integrated" else program.acquire_decimated
    result = acquire(soc, progress=False)
    assert len(result) == 1
    assert len(observed) == 1
    outcome = observed[0]
    assert outcome.compile_wall_s == 2.0
    assert outcome.wall_s == 10.0
    assert outcome.complete is True
    assert len(outcome.rounds) == 1
    assert outcome.rounds[0].acquire.sequence == 0
    with pytest.raises(FrozenInstanceError):
        outcome.__setattr__("complete", False)


def test_binding_reports_compile_time_and_normal_return(monkeypatch):
    observed = []
    now = [10.0]
    # Acquire timing has no public clock injection; control its monotonic collaborator.
    monkeypatch.setattr("time.monotonic", lambda: now[0])
    binding = synthetic_binding(SimulationCallbacks(on_end=observed.append))
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


@pytest.mark.parametrize("body_error", [ValueError, KeyboardInterrupt])
def test_binding_records_body_exception_and_sums_failed_compile_scopes(
    monkeypatch, body_error
):
    observed = []
    now = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    binding = synthetic_binding(SimulationCallbacks(on_end=observed.append))

    def acquire():
        with binding.acquire():
            with binding.compiling():
                now[0] += 2.0
            with binding.compiling():
                now[0] += 3.0
                raise body_error("compile interrupted")

    with pytest.raises(body_error, match="compile interrupted"):
        acquire()
    assert observed[0].complete is False
    assert observed[0].compile_wall_s == 5.0
    assert observed[0].wall_s == 5.0
    with binding.acquire() as sequence:
        assert sequence == 1
    assert observed[1].compile_wall_s == 0.0


def test_completion_observer_sees_all_rounds_and_cannot_mask_execution_failure(caplog):
    observed = []

    def finish(outcome):
        observed.append(outcome)
        raise OSError("observer cannot write")

    def execute(request):
        if request.index == 1:
            raise ValueError("original execution failure")
        return RoundSignals(())

    binding = SimulationBinding(
        lambda: EnvironmentSnapshot("synthetic", "test", ()),
        execute,
        backend_version="test",
        seed=1,
        callbacks=SimulationCallbacks(on_end=finish),
    )

    def acquire():
        with binding.acquire() as sequence:
            request = AcquireRequest(
                "decimated",
                (1,),
                0,
                (),
                1,
                sequence,
                binding.rng_version,
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
    assert observed[1].complete is True
    assert [record.message for record in caplog.records] == [
        "simulation acquire observer failed"
    ]


@pytest.mark.parametrize("mode", ["integrated", "decimated"])
def test_builder_failure_reports_elapsed_compile_and_releases_call(
    program_seam, monkeypatch, mode
):
    program, soc, observed = program_seam
    now = [10.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])

    def build_then_fail(*args, **kwargs):
        build_compiled_request(*args, **kwargs)
        now[0] += 2.0
        raise ValueError("builder failed")

    # Builder has no failure-injection option; delegate real construction first.
    with monkeypatch.context() as patch:
        patch.setattr(
            "zcu_tools.program.v2.simulation.compiled.build_compiled_request",
            build_then_fail,
        )
        acquire = program.acquire if mode == "integrated" else program.acquire_decimated
        with pytest.raises(ValueError, match="builder failed"):
            acquire(soc, progress=False)
    assert len(observed) == 1
    assert observed[0].compile_wall_s == 2.0
    assert observed[0].wall_s == 2.0
    assert observed[0].rounds == ()
    assert observed[0].complete is False
    with pytest.raises(RuntimeError, match="no active acquisition"):
        soc.run_round()
    acquire(soc, progress=False)
    assert observed[1].complete is True
    assert observed[1].rounds[0].acquire.sequence == 1


@pytest.mark.parametrize("mode", ["integrated", "decimated"])
def test_execute_failure_and_success_inside_outer_except(program_seam, mode):
    program, soc, observed = program_seam
    execute = soc.binding.execute

    def fail(_request):
        raise ValueError("execute failed")

    soc.binding.execute = fail
    acquire = program.acquire if mode == "integrated" else program.acquire_decimated
    try:
        acquire(soc, progress=False)
    except ValueError as error:
        assert str(error) == "execute failed"
        soc.binding.execute = execute
        acquire(soc, progress=False)
    else:
        pytest.fail("execute failure must propagate")
    assert [outcome.complete for outcome in observed] == [False, True]
    assert [outcome.rounds[0].acquire.sequence for outcome in observed] == [0, 1]
    with pytest.raises(RuntimeError, match="no active acquisition"):
        soc.run_round()


@pytest.mark.parametrize("mode", ["integrated", "decimated"])
@pytest.mark.parametrize("set_flag", [False, True], ids=["flag-clear", "flag-set"])
@pytest.mark.parametrize("completed_rounds", [0, 2])
def test_stop_during_backend_keeps_only_completed_rounds(
    program_seam, mode, completed_rounds, set_flag
):
    program, soc, observed = program_seam
    cancel = Event()
    hook_rounds = []
    execute = soc.binding.execute
    acquire = program.acquire if mode == "integrated" else program.acquire_decimated

    def solve(request: RoundRequest) -> IntegratedRound | RoundSignals:
        assert request.cancel is cancel
        if request.index == completed_rounds:
            if set_flag:
                cancel.set()
            raise Cancelled()
        samples = request.acquire.readouts[0].samples
        iq = np.array([(2.0, -8.0), (8.0, -2.0)][request.index])
        if mode == "decimated":
            trace = np.broadcast_to(iq, (1, 1, samples, 2))
            return RoundSignals((ChannelSignals(0, trace),))
        result = execute(request)
        assert isinstance(result, IntegratedRound)
        (channel,) = result.statistics.channels
        # The integrated backend supplies sample sums, not normalized IQ.
        channel = replace(channel, sums=(samples * iq)[None, :])
        statistics = replace(result.statistics, channels=(channel,))
        return replace(result, statistics=statistics)

    def round_hook(count, _iq, flag: CancelFlagProtocol) -> None:
        assert flag is cancel
        hook_rounds.append(count)

    def run():
        return acquire(
            soc,
            rounds=4,
            cancel_flag=cancel,
            round_hook=round_hook,
            remove_offset=False,
            progress=False,
        )

    soc.binding.execute = solve
    if completed_rounds == 0:
        with pytest.raises(
            StoppedPartialAcquireError,
            match="^acquire stopped before the first round completed$",
        ):
            run()
    else:
        (iq,) = run()
        np.testing.assert_array_equal(iq[..., 0], 5.0)
        np.testing.assert_array_equal(iq[..., 1], -5.0)
    assert hook_rounds == list(range(1, completed_rounds + 1))
    assert cancel.is_set() is set_flag
    (outcome,) = observed
    assert outcome.complete is bool(completed_rounds)
    assert len(outcome.rounds) == completed_rounds + 1


@pytest.mark.parametrize("failure", ["first-round", "second-round", "unrelated"])
def test_stop_during_required_raw_preserves_only_completed_rounds(program_seam, failure):
    program, soc, observed = program_seam
    cancel = Event()
    hooks = []
    execute = soc.binding.execute
    failure_index = int(failure == "second-round")
    completed_raw = np.full((1, 1, 2), 8, dtype=np.int64)

    def solve(request: RoundRequest) -> IntegratedRound:
        result = execute(request)
        assert isinstance(result, IntegratedRound)

        class Raw:
            def materialize(self, reps, avg_level):
                if request.index == failure_index:
                    if failure == "unrelated":
                        raise ValueError("bad ADC data")
                    raise Cancelled()
                return {0: completed_raw.copy()}

        return replace(result, raw=Raw())

    def run():
        return program.acquire(
            soc,
            rounds=3,
            cancel_flag=cancel,
            threshold=0.0,
            remove_offset=False,
            progress=False,
            round_hook=lambda count, *_: hooks.append(count),
        )

    soc.binding.execute = solve
    if failure == "unrelated":
        with pytest.raises(ValueError, match="bad ADC data"):
            run()
    elif failure_index == 0:
        with pytest.raises(StoppedPartialAcquireError):
            run()
    else:
        (iq,) = run()
        np.testing.assert_array_equal(iq, [[1.0, 0.0]])
        np.testing.assert_array_equal(program.get_raw(), [completed_raw])
        np.testing.assert_array_equal(soc.get_accumulated(0, 0, 1), [[8, 8]])

    assert hooks == list(range(1, failure_index + 1))
    assert (program.get_raw() is None) is (failure_index == 0)
    assert not cancel.is_set()
    (outcome,) = observed
    assert outcome.complete is bool(failure_index)
    assert len(outcome.rounds) == failure_index + 1


def test_cleanup_failure_is_inside_binding_outcome(program_seam, monkeypatch):
    program, soc, observed = program_seam
    release = soc.release_request

    def release_then_fail():
        release()
        raise OSError("release failed")

    # Cleanup has no failure-injection option; release the real request first.
    monkeypatch.setattr(soc, "release_request", release_then_fail)
    with pytest.raises(OSError, match="release failed"):
        program.acquire(soc, progress=False)
    assert len(observed) == 1
    assert observed[0].complete is False
    assert len(observed[0].rounds) == 1
    with pytest.raises(RuntimeError, match="no active acquisition"):
        soc.run_round()
    with soc.binding.acquire() as sequence:
        assert sequence == 1


def test_observer_exception_warns_once_and_preserves_normal_return():
    observed = []

    def finish(outcome):
        observed.append(outcome)
        raise OSError("observer cannot write")

    binding = synthetic_binding(SimulationCallbacks(on_end=finish))
    returned = []
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        for _ in range(2):
            with binding.acquire() as sequence:
                returned.append(sequence)
    assert returned == [0, 1]
    assert [outcome.complete for outcome in observed] == [True, True]
    assert len(captured) == 1
    assert captured[0].category is RuntimeWarning
    assert "observer cannot write" in str(captured[0].message)


@pytest.mark.parametrize("failing_hook", ["wait_ready", "on_begin"])
def test_failed_preparation_consumes_no_sequence_or_outcome(failing_hook):
    observed = []
    events = []
    fail = [True]

    def readiness():
        events.append("ready")
        if failing_hook == "wait_ready" and fail[0]:
            raise ValueError("not ready")

    def begin():
        events.append("begin")
        if failing_hook == "on_begin" and fail[0]:
            raise ValueError("begin failed")

    binding = synthetic_binding(
        SimulationCallbacks(
            wait_ready=readiness, on_begin=begin, on_end=observed.append
        )
    )
    with pytest.raises(ValueError, match="not ready|begin failed"), binding.acquire():
        pytest.fail("preparation must fail before entering body")
    assert observed == []
    assert events == (["ready"] if failing_hook == "wait_ready" else ["ready", "begin"])
    fail[0] = False
    events.clear()
    with binding.acquire() as sequence:
        assert sequence == 0
        assert events == ["ready", "begin"]
        with pytest.raises(RuntimeError, match="reentry"), binding.acquire():
            pytest.fail("nested acquire must not enter")
    with binding.acquire() as sequence:
        assert sequence == 1
    assert len(observed) == 2
    assert observed[0].compile_wall_s == 0.0


@pytest.mark.parametrize("body_fails", [False, True])
def test_observer_base_exception_propagates_with_original_context(body_fails):
    observed = []
    interrupt = [True]
    original = ValueError("body failed")

    def finish(outcome):
        observed.append(outcome)
        if interrupt[0]:
            raise KeyboardInterrupt("observer interrupted")

    binding = synthetic_binding(SimulationCallbacks(on_end=finish))

    def acquire():
        with binding.acquire():
            if body_fails:
                raise original

    with pytest.raises(KeyboardInterrupt) as caught:
        acquire()
    assert caught.value.__context__ is (original if body_fails else None)
    assert observed[0].complete is not body_fails
    interrupt[0] = False
    with binding.acquire() as sequence:
        assert sequence == 1


def test_no_observer_acquire_and_compiling_read_no_clock(monkeypatch):
    def forbidden():
        pytest.fail("unobserved acquire must not read clock")

    monkeypatch.setattr(time, "monotonic", forbidden)
    binding = synthetic_binding()
    with binding.acquire() as sequence:
        assert sequence == 0
        with binding.compiling():
            pass
        binding.sample_round(
            AcquireRequest(
                "decimated",
                (1,),
                0,
                (),
                1,
                sequence,
                binding.rng_version,
                time_base=TimeBase(Fraction(1)),
            ),
            0,
        )
