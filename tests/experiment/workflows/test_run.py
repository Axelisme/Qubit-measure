"""Experiment Run integrates cfg, buffers, signals, and Schedule at its seam."""

from dataclasses import dataclass, field

import numpy as np
import pytest
from numpy.typing import NDArray
from qick import QickConfig
from zcu_tools.experiment.stop_signal import StopSignal
from zcu_tools.experiment.v2.runtime import SignalBuffer
from zcu_tools.experiment.workflows import MissingCapability, Run


@dataclass
class Cfg:
    value: float = 1.0
    nested: list[float] = field(default_factory=lambda: [2.0])


def test_run_cfg_and_schedule_cfg_are_independent_copies() -> None:
    original = Cfg()
    signal = StopSignal()
    run = Run(original, signal)
    buffer = run.buffer((2,), axes=(np.array([0.0, 1.0]),))

    run.cfg.nested.append(3.0)
    with run.schedule(buffer) as schedule:
        assert schedule.env is run
        assert schedule.stop is signal
        schedule.cfg.value = 10.0
        schedule.cfg.nested.append(4.0)
        buffer.set(np.array([1.0 + 2.0j, 3.0 + 4.0j]))

    assert original == Cfg()
    assert run.cfg == Cfg(value=1.0, nested=[2.0, 3.0])
    assert run.outcome.status == "completed"
    np.testing.assert_array_equal(buffer.array, [1.0 + 2.0j, 3.0 + 4.0j])


def test_absent_capabilities_fail_at_access() -> None:
    run = Run(Cfg(), StopSignal())
    with pytest.raises(MissingCapability, match="soc"):
        _ = run.soc
    with pytest.raises(MissingCapability, match="soc"):
        _ = run.soccfg
    with pytest.raises(MissingCapability, match="devices"):
        _ = run.devices


def test_soc_and_config_are_the_original_injected_handles() -> None:
    soc = object()
    soccfg = QickConfig({})
    run = Run(Cfg(), StopSignal(), soc=soc, soccfg=soccfg)
    assert run.soc is soc
    assert run.soccfg is soccfg


def test_buffer_starts_complex_with_unfilled_nan_slots() -> None:
    run = Run(Cfg(), StopSignal())
    buffer = run.buffer(
        (2, 3),
        axes=(np.array([0.0, 1.0]), np.array([1.0, 2.0, 3.0])),
    )
    assert buffer.array.dtype == np.complex128
    assert buffer.array.shape == (2, 3)
    assert np.isnan(buffer.array).all()


@pytest.mark.parametrize("shape", [(), (0,), (-1,), (True,)])
def test_invalid_buffer_dimensions_are_rejected(shape: tuple[int, ...]) -> None:
    run = Run(Cfg(), StopSignal())
    with pytest.raises(ValueError, match="positive integers"):
        run.buffer(shape, axes=tuple(np.ones(1) for _ in shape))


@pytest.mark.parametrize(
    "axes",
    [
        (),
        (np.array([0.0]),),
        (np.array([[0.0, 1.0]]),),
        (np.array([0.0, np.inf]),),
        (np.array([0.0, np.nan]),),
        (np.array([0.0, 1.0], dtype=np.float32),),
    ],
)
def test_buffer_axes_require_matching_finite_float64_vectors(
    axes: tuple[NDArray[np.float64], ...],
) -> None:
    run = Run(Cfg(), StopSignal())
    with pytest.raises(ValueError, match="axes"):
        run.buffer((2,), axes=axes)


def test_schedule_rejects_another_runs_buffer() -> None:
    first = Run(Cfg(), StopSignal())
    other = Run(Cfg(), StopSignal())
    buffer = other.buffer((1,), axes=(np.array([0.0]),))
    with pytest.raises(ValueError, match="belong to this Run"), first.schedule(buffer):
        pass


def test_schedule_body_errors_propagate_unchanged() -> None:
    run = Run(Cfg(), StopSignal())
    buffer = run.buffer((1,), axes=(np.array([0.0]),))
    cause = ValueError("experiment body failed")
    with (
        pytest.raises(ValueError, match="experiment body failed") as caught,
        run.schedule(buffer),
    ):
        raise cause
    assert caught.value is cause


def test_first_failed_schedule_is_not_erased_by_later_completed_schedule() -> None:
    run = Run(Cfg(), StopSignal())
    buffer = run.buffer((1,), axes=(np.array([0.0]),))
    cause = OSError("acquisition failed")

    def fail(_step: object) -> None:
        raise cause

    with run.schedule(buffer) as schedule:
        schedule.batch({"acquire": fail})
    first = run.outcome

    with run.schedule(buffer) as later:
        pass

    assert later.outcome.status == "completed"
    assert first.status == "failed"
    assert run.outcome.status == "failed"
    assert run.outcome.exception is cause
    assert run.cancel_signal.error is not None
    assert run.cancel_signal.error.exception is cause


def test_interrupted_schedule_retains_the_original_interrupt() -> None:
    run = Run(Cfg(), StopSignal())
    buffer = run.buffer((1,), axes=(np.array([0.0]),))
    interrupt = KeyboardInterrupt()

    def interrupt_acquire(_step: object) -> None:
        raise interrupt

    with run.schedule(buffer) as schedule:
        schedule.batch({"acquire": interrupt_acquire})

    assert run.outcome.status == "interrupted"
    assert run.outcome.exception is interrupt


def test_cancel_signal_is_explicit_and_produces_stopped_outcome() -> None:
    signal = StopSignal()
    run = Run(Cfg(), signal)
    buffer = run.buffer((1,), axes=(np.array([0.0]),))
    signal.set()
    with run.schedule(buffer) as schedule:
        seen = list(schedule.scan("point", (0,)))
    assert seen == []
    assert run.outcome.status == "stopped"


def test_non_dataclass_cfg_is_rejected() -> None:
    with pytest.raises(TypeError, match="dataclass instance"):
        Run(1.0, StopSignal())


def test_external_buffer_is_rejected() -> None:
    run = Run(Cfg(), StopSignal())
    buffer = SignalBuffer((1,))
    with pytest.raises(ValueError, match="belong to this Run"), run.schedule(buffer):
        pass
