"""QICK callable metadata and argument semantics at public simulation acquire."""

import gc
import inspect
import weakref
from collections.abc import Iterator

import numpy as np
import pytest
from qick.qick_asm import AcquireMixin
from zcu_tools.experiment.simulation import SimulationEnvironment, white_noise
from zcu_tools.program.v2 import ModularProgramV2
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg

from tests.program.v2.simulation.support import raw_of, request_of


class QickCallable:
    """External callable exposing observable Python signature metadata."""

    def __init__(self, signature: inspect.Signature):
        self.metadata = signature
        self.reads = 0

    @property
    def __signature__(self) -> inspect.Signature:
        self.reads += 1
        return self.metadata

    def __call__(self, *args, **kwargs):
        raise AssertionError("integrated simulation must not run QICK transport")


def program(env: SimulationEnvironment) -> ModularProgramV2:
    return ModularProgramV2(
        env.soccfg,
        modules=[
            PulseReadoutCfg(
                gen_ch=0,
                ro_ch=0,
                nqz=1,
                freq=100,
                gain=0.1,
                waveform=ConstWaveformCfg(length=0.1),
                ro_length=0.1,
            ).build("readout")
        ],
        reps=3,
    )


@pytest.fixture
def acquisition() -> Iterator[tuple[SimulationEnvironment, ModularProgramV2]]:
    env = white_noise(std_adc=3, seed=123456)
    prog = program(env)
    try:
        yield env, prog
    finally:
        # Re-enter with the restored external callable so shared metadata does
        # not retain a test double after this test.
        prog.acquire(
            env.soc,
            progress=False,
            rounds=2,
        )
        env.devices.close_all_devices()


def test_acquire_reuses_callable_metadata_across_programs_without_retaining_self(
    acquisition, monkeypatch: pytest.MonkeyPatch
):
    env, _ = acquisition
    callable = QickCallable(inspect.signature(AcquireMixin.acquire))
    with monkeypatch.context() as patch:
        patch.setattr(AcquireMixin, "acquire", callable)
        first = program(env)
        first.acquire(
            env.soc,
            progress=False,
            rounds=2,
        )
        retained = weakref.ref(first)
        del first
        gc.collect()
        assert retained() is None
        program(env).acquire(env.soc, progress=False)
        assert callable.reads == 1


def test_acquire_reinspects_replacement_and_only_retains_current_callable(
    acquisition, monkeypatch: pytest.MonkeyPatch
):
    env, prog = acquisition
    original_signature = inspect.signature(AcquireMixin.acquire)
    first = QickCallable(original_signature)
    replacement_signature = original_signature.replace(
        parameters=[
            *original_signature.parameters.values(),
            inspect.Parameter("binding_token", inspect.Parameter.KEYWORD_ONLY),
        ]
    )
    replacement = QickCallable(replacement_signature)
    with monkeypatch.context() as patch:
        patch.setattr(AcquireMixin, "acquire", first)
        prog.acquire(env.soc, progress=False)
        patch.setattr(AcquireMixin, "acquire", replacement)
        with pytest.raises(
            TypeError, match="missing a required argument: 'binding_token'"
        ):
            prog.acquire(env.soc, progress=False)
        prog.acquire(env.soc, progress=False, binding_token=True)
        assert replacement.reads == 1
        patch.setattr(AcquireMixin, "acquire", first)
        prog.acquire(env.soc, progress=False)
        assert first.reads == 2


@pytest.mark.parametrize("invalid", ["unknown", "duplicate", "excess"])
def test_acquire_preserves_bind_errors_sampling_identity_and_bitwise_output(
    acquisition, monkeypatch: pytest.MonkeyPatch, invalid: str
):
    env, prog = acquisition
    replay_env = white_noise(std_adc=3, seed=123456)
    replay = program(replay_env)
    callable = QickCallable(inspect.signature(AcquireMixin.acquire))

    def invalid_call(target, soc):
        if invalid == "unknown":
            return target.acquire(soc, unknown_argument=True)
        if invalid == "duplicate":
            return target.acquire(soc, 1, rounds=1)
        return target.acquire(soc, *range(20))

    try:
        with pytest.raises(TypeError) as original_error:
            invalid_call(prog, env.soc)
        expected = prog.acquire(env.soc, rounds=2, progress=False)
        with monkeypatch.context() as patch:
            patch.setattr(AcquireMixin, "acquire", callable)
            with pytest.raises(TypeError) as replay_error:
                invalid_call(replay, replay_env.soc)
            actual = replay.acquire(
                replay_env.soc,
                progress=False,
                rounds=2,
            )
            assert str(replay_error.value) == str(original_error.value)
            expected_request, actual_request = request_of(env), request_of(replay_env)
            # QICK argument binding happens after simulation admission.
            expected_sequence = 1
            assert (
                actual_request.sequence
                == expected_request.sequence
                == expected_sequence
            )
            assert actual_request.root_seed == expected_request.root_seed == 123456
            assert actual_request.rng_version == expected_request.rng_version
            assert np.asarray(actual).tobytes() == np.asarray(expected).tobytes()
            assert (
                np.asarray(raw_of(replay)).tobytes()
                == np.asarray(raw_of(prog)).tobytes()
            )
    finally:
        replay_env.devices.close_all_devices()
