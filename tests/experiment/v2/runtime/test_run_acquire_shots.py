"""Raw-shot acquisition through public Schedule and ScheduleStep entrypoints."""

from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from zcu_tools.experiment.scope import Scope
from zcu_tools.experiment.v2.runtime import RunOptions, Schedule, SignalBuffer
from zcu_tools.program.acquisition import (
    AcquisitionContractError,
    CancelFlagProtocol,
    StoppedPartialAcquireError,
)
from zcu_tools.progress_bar import use_pbar_factory

from tests.experiment.v2.runtime._flow_fakes import (
    RecordingProgressBar,
    recording_pbar_factory,
)


class ShotProducer:
    """Expose distinct averaged and retained shot data at the producer boundary."""

    def __init__(self) -> None:
        self.shots: list[NDArray[np.int64]] | None = [
            np.array([[11, -3], [7, 5]], dtype=np.int64)
        ]
        self.calls = 0
        self.raw_reads = 0
        self.action: Callable[[CancelFlagProtocol], None] | None = None
        self.errors: list[BaseException] = []
        self.kwargs: dict[str, object] = {}

    def acquire(
        self,
        soc: object,
        *,
        rounds: int,
        progress: bool,
        round_hook: Callable[
            [int, list[NDArray[np.float64]], CancelFlagProtocol], object
        ],
        cancel_flag: CancelFlagProtocol,
        **kwargs: object,
    ) -> list[NDArray[np.float64]]:
        self.calls += 1
        self.kwargs = dict(kwargs, rounds=rounds, progress=progress, soc=soc)
        if self.action is not None:
            self.action(cancel_flag)
        if self.errors:
            raise self.errors.pop(0)
        averaged = [np.array([[[4.0, 1.0]]])]
        round_hook(1, averaged, cancel_flag)
        return averaged

    def get_raw(self) -> list[NDArray[np.int64]] | None:
        self.raw_reads += 1
        return self.shots


ShotRun = tuple[Scope, Schedule[Any], ShotProducer]


@pytest.fixture
def active_run() -> Iterator[ShotRun]:
    root = Scope("run")
    with Schedule[Any](scope=root) as sched:
        yield root, sched, ShotProducer()


@pytest.mark.parametrize("use_step", [False, True])
def test_returns_retained_shots_without_a_slot_and_forwards_native_keywords(
    use_step: bool,
    active_run: ShotRun,
) -> None:
    root, sched, producer = active_run
    soc = object()
    owners = (
        (step for _, step in sched.scan("point", [0])) if use_step else iter([sched])
    )
    for owner in owners:
        result = owner.run_acquire_shots(
            producer, soc, options=RunOptions(progress=True), soft_avgs=3
        )
        assert result is producer.shots
        assert producer.calls == 1
        assert producer.raw_reads == 1
        assert producer.kwargs == dict(soc=soc, rounds=1, progress=True, soft_avgs=3)
        assert not root.cancel("acquire").cancelled
    assert sched.outcome.status == "completed"


@pytest.mark.parametrize("invalid", ["converter", "stop_condition", "rounds"])
def test_rejects_slot_policies_and_multiple_rounds_before_acquiring(
    invalid: str,
    active_run: ShotRun,
) -> None:
    _, sched, producer = active_run
    options: RunOptions[list[NDArray[np.float64]]] = RunOptions(
        converter=(lambda raw: raw[0]) if invalid == "converter" else None,
        stop_condition=(lambda: True) if invalid == "stop_condition" else None,
    )
    with pytest.raises(ValueError, match="rounds|convert|data stop"):
        sched.run_acquire_shots(
            producer, object(), options=options, rounds=2 if invalid == "rounds" else 1
        )
    assert producer.calls == 0


def test_missing_raw_buffer_fails_after_acquire_without_retry(
    active_run: ShotRun,
) -> None:
    _, sched, producer = active_run
    producer.shots = None
    with pytest.raises(RuntimeError, match="raw shot buffer"):
        sched.run_acquire_shots(producer, object(), options=RunOptions(retry=2))
    assert producer.calls == 1
    assert sched.outcome.status == "completed"


@pytest.mark.parametrize("target", ["acquire", "schedule"])
@pytest.mark.parametrize("partial_error", [False, True])
def test_external_cancel_returns_none_after_scope_and_progress_cleanup(
    target: str,
    partial_error: bool,
    active_run: ShotRun,
) -> None:
    root, sched, producer = active_run
    bars: list[RecordingProgressBar] = []
    observed: list[int] = []

    def cancel(flag: CancelFlagProtocol) -> None:
        assert not flag.is_set()
        assert root.cancel(target).cancelled
        assert flag.is_set()

    def on_stop() -> None:
        assert root.active_path == sched.scope.path
        assert bars and all(bar.closed for bar in bars)
        observed.append(producer.calls)

    producer.action = cancel
    if partial_error:
        producer.errors = [StoppedPartialAcquireError("before completed round")]
    with use_pbar_factory(recording_pbar_factory(bars)):
        result = sched.run_acquire_shots(
            producer, object(), options=RunOptions(retry=2, on_stop=on_stop)
        )
    assert result is None
    assert observed == [1]
    assert producer.calls == 1
    assert producer.raw_reads == 0
    assert not root.cancel("acquire").cancelled
    assert sched.outcome.status == ("stopped" if target == "schedule" else "completed")
    if target == "acquire":
        producer.action = None
        assert sched.run_acquire_shots(producer, object()) is producer.shots
        assert observed == [1]
        assert not sched.is_stop()


def test_pre_stopped_schedule_observes_once_without_acquiring(
    active_run: ShotRun,
) -> None:
    _, sched, producer = active_run
    observed: list[bool] = []
    sched.set_stop()
    result = sched.run_acquire_shots(
        producer, object(), options=RunOptions(on_stop=lambda: observed.append(True))
    )
    assert result is None
    assert observed == [True]
    assert producer.calls == producer.raw_reads == 0
    assert sched.outcome.status == "stopped"


@pytest.mark.parametrize("retry", [0, 1])
@pytest.mark.parametrize("overrun", [False, True])
def test_producer_errors_use_explicit_retry_budget(
    retry: int,
    overrun: bool,
    active_run: ShotRun,
    caplog: pytest.LogCaptureFixture,
) -> None:
    root, sched, producer = active_run
    cause = (
        AcquisitionContractError("too many shots")
        if overrun
        else RuntimeError("offline")
    )
    producer.errors = [cause]
    observed: list[bool] = []
    options: RunOptions[list[NDArray[np.float64]]] = RunOptions(
        retry=retry, on_stop=lambda: observed.append(True)
    )
    if retry:
        assert (
            sched.run_acquire_shots(producer, object(), options=options)
            is producer.shots
        )
        assert producer.calls == 2
        assert sched.outcome.status == "completed"
        assert ("shot_overrun" if overrun else "RuntimeError") in caplog.text
    else:
        with pytest.raises(type(cause)) as caught:
            sched.run_acquire_shots(producer, object(), options=options)
        assert caught.value is cause
        assert producer.calls == 1
        assert sched.outcome.status == "failed"
    assert observed == []
    assert not root.cancel("acquire").cancelled


@pytest.mark.parametrize("interrupt", [False, True])
def test_unrequested_partial_error_and_keyboard_interrupt_propagate_unchanged(
    interrupt: bool,
    active_run: ShotRun,
) -> None:
    root, sched, producer = active_run
    cause = (
        KeyboardInterrupt("interrupted")
        if interrupt
        else StoppedPartialAcquireError("not stopped")
    )
    producer.errors = [cause]
    observed: list[bool] = []
    with pytest.raises(type(cause)) as caught:
        sched.run_acquire_shots(
            producer,
            object(),
            options=RunOptions(
                retry=2 if interrupt else 0, on_stop=lambda: observed.append(True)
            ),
        )
    assert caught.value is cause
    assert producer.calls == 1
    assert observed == []
    assert not root.cancel("acquire").cancelled


def test_concurrent_schedule_stop_does_not_suppress_producer_failure(
    active_run: ShotRun,
) -> None:
    root, sched, producer = active_run
    cause = RuntimeError("transport failed")
    producer.errors = [cause]

    def cancel(flag: CancelFlagProtocol) -> None:
        assert root.cancel("schedule").cancelled
        assert flag.is_set()

    producer.action = cancel
    with pytest.raises(RuntimeError) as caught:
        sched.run_acquire_shots(producer, object(), options=RunOptions(retry=2))
    assert caught.value is cause
    assert producer.calls == 1
    assert sched.outcome.status == "stopped"


def test_raw_shots_leave_an_existing_slot_and_live_updates_untouched() -> None:
    updates: list[NDArray[np.float64]] = []
    buffer = SignalBuffer((1,), dtype=np.float64, on_update=updates.append)
    buffer.set(np.array([99.0]))
    updates.clear()
    with Schedule(buffer, scope=Scope("run")) as sched:
        producer = ShotProducer()
        assert sched.run_acquire_shots(producer, object()) is producer.shots
        np.testing.assert_array_equal(buffer.data, [99.0])
        assert updates == []
