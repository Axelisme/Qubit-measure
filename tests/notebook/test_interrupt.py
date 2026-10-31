"""Notebook SIGINT requests cooperate with a synchronous experiment core."""

import signal
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from types import FrameType

import pytest
from zcu_tools.device import DeviceManager
from zcu_tools.experiment.context import RunContext
from zcu_tools.notebook import NotebookAdapter, NotebookExperiment
from zcu_tools.plotting.plots import NonPresentingHost

from tests.notebook._adapter_fakes import Cfg, RecordingCore


class InterruptCore(RecordingCore):
    """Run an authored action before producing the ordinary recording result."""

    def __init__(self, action: Callable[[RunContext], None]) -> None:
        super().__init__()
        self.action = action

    def run(self, config: Cfg, *, context: RunContext) -> float:
        self.action(context)
        return super().run(config, context=context)


def notebook(
    core: InterruptCore, *, interrupt_timeout: float = 10.0
) -> NotebookExperiment[InterruptCore]:
    return NotebookAdapter(
        soc=object(),
        soccfg=object(),
        device_manager=DeviceManager(),
        host=NonPresentingHost(),
        interrupt_timeout=interrupt_timeout,
    )(core)


class ManualTimer:
    """Deliver the external timer deadline only when the test advances it."""

    def __init__(self, interval: float, function: Callable[[], None]) -> None:
        self.interval = interval
        self.function = function
        self.started = False
        self.cancelled = False
        self.on_cancel: Callable[[], None] = lambda: None

    def start(self) -> None:
        self.started = True

    def cancel(self) -> None:
        self.on_cancel()
        self.cancelled = True

    def join(self) -> None:
        pass

    def expire(self) -> None:
        if self.started and not self.cancelled:
            self.function()


@pytest.fixture
def timers(monkeypatch: pytest.MonkeyPatch) -> list[ManualTimer]:
    created: list[ManualTimer] = []

    def timer(interval: float, function: Callable[[], None]) -> ManualTimer:
        instance = ManualTimer(interval, function)
        created.append(instance)
        return instance

    monkeypatch.setattr(threading, "Timer", timer)
    return created


def test_timeout_interrupts_core_and_retains_previous_record(
    timers: list[ManualTimer],
) -> None:
    core = InterruptCore(lambda context: None)
    adapter = notebook(core)
    previous = adapter.run(Cfg())
    previous_evidence = adapter.acquisition_for(previous)
    previous_handler = signal.getsignal(signal.SIGINT)

    def blocked_author(context: RunContext) -> None:
        signal.raise_signal(signal.SIGINT)
        assert context.scope.stop.is_set()
        assert timers and timers[-1].started, "First SIGINT must arm a deadline"
        timers[-1].expire()
        pytest.fail("The deadline must interrupt the core")

    core.action = blocked_author
    with pytest.raises(KeyboardInterrupt):
        adapter.run(Cfg())
    assert adapter.last_run is previous
    assert adapter.acquisition_for(previous) == previous_evidence
    assert signal.getsignal(signal.SIGINT) is previous_handler
    assert timers[-1].cancelled


def test_return_cancels_deadline_before_restoring_handler(
    timers: list[ManualTimer],
) -> None:
    received: list[int] = []

    def previous(signum: int, frame: FrameType | None) -> None:
        received.append(signum)

    original = signal.signal(signal.SIGINT, previous)
    try:

        def check_cancel() -> None:
            assert signal.getsignal(signal.SIGINT) is not previous

        def author(context: RunContext) -> None:
            signal.raise_signal(signal.SIGINT)
            timers[-1].on_cancel = check_cancel

        adapter = notebook(InterruptCore(author))
        adapter.run(Cfg())
        assert timers and timers[-1].cancelled
        assert signal.getsignal(signal.SIGINT) is previous
        timers[-1].expire()
        assert received == []
        signal.raise_signal(signal.SIGINT)
        assert received == [signal.SIGINT]
    finally:
        signal.signal(signal.SIGINT, original)


def test_second_sigint_interrupts_without_publishing(timers: list[ManualTimer]) -> None:
    def author(context: RunContext) -> None:
        signal.raise_signal(signal.SIGINT)
        signal.raise_signal(signal.SIGINT)

    adapter = notebook(InterruptCore(author))
    with pytest.raises(KeyboardInterrupt):
        adapter.run(Cfg())
    assert adapter.last_run is None
    assert timers[-1].cancelled


@pytest.mark.parametrize("timeout", [0.0, -1.0])
def test_nonpositive_timeout_is_rejected_at_adapter_entry(timeout: float) -> None:
    with pytest.raises(ValueError, match="interrupt_timeout"):
        NotebookAdapter(interrupt_timeout=timeout)
    with pytest.raises(ValueError, match="interrupt_timeout"):
        NotebookExperiment(RecordingCore(), interrupt_timeout=timeout)


def test_custom_timeout_is_forwarded_to_deadline(timers: list[ManualTimer]) -> None:
    def author(context: RunContext) -> None:
        signal.raise_signal(signal.SIGINT)
        assert timers[-1].interval == 0.25

    notebook(InterruptCore(author), interrupt_timeout=0.25).run(Cfg())


def test_timeout_fallback_delivers_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch, timers: list[ManualTimer]
) -> None:
    monkeypatch.delattr(signal, "pthread_kill", raising=False)

    def author(context: RunContext) -> None:
        signal.raise_signal(signal.SIGINT)
        timers[-1].expire()

    adapter = notebook(InterruptCore(author))
    with pytest.raises(KeyboardInterrupt):
        adapter.run(Cfg())
    assert adapter.last_run is None


@pytest.mark.skipif(not hasattr(signal, "pthread_kill"), reason="POSIX signal contract")
def test_controlled_deadline_targets_main_thread_during_blocking_wait(
    monkeypatch: pytest.MonkeyPatch,
    timers: list[ManualTimer],
) -> None:
    send = signal.pthread_kill
    main_thread = threading.get_ident()
    delivered: list[tuple[int, int, int]] = []

    def deliver(thread_id: int, signum: int) -> None:
        delivered.append((threading.get_ident(), thread_id, signum))
        send(thread_id, signum)

    monkeypatch.setattr(signal, "pthread_kill", deliver)

    condition = threading.Condition()
    armed = threading.Event()
    abort = threading.Event()
    entered_wait: list[bool] = []
    errors: list[BaseException] = []
    previous_handler = signal.getsignal(signal.SIGINT)

    def expire_when_waiting() -> None:
        try:
            # The submitted author arms the timer while holding condition.
            assert armed.wait(30)
            with condition:
                if abort.is_set():
                    return
                # Acquiring this lock proves Condition.wait released it inside
                # its blocking protocol, not merely that the author will call wait.
                # No notifier releases the wait before the deadline is delivered.
                entered_wait.append(True)
                timers[-1].expire()
        except BaseException as error:
            errors.append(error)
            with condition:
                condition.notify_all()

    def author(context: RunContext) -> None:
        with condition:
            signal.raise_signal(signal.SIGINT)
            armed.set()
            # Only the controlled deadline can end this wait; 30s guards deadlock.
            condition.wait(30)
        pytest.fail("The deadline must interrupt the blocking wait")

    worker = threading.Thread(target=expire_when_waiting)
    adapter = notebook(InterruptCore(author))
    worker.start()
    try:
        with pytest.raises(KeyboardInterrupt):
            adapter.run(Cfg())
    finally:
        abort.set()
        armed.set()
        with condition:
            condition.notify_all()
        worker.join(30)
        assert not worker.is_alive(), "Deadline delivery must finish before teardown"
        if errors:
            raise errors[0]
    assert entered_wait == [True]
    assert timers[-1].cancelled
    assert signal.getsignal(signal.SIGINT) is previous_handler
    assert adapter.last_run is None
    assert len(delivered) == 1
    sender, target, signum = delivered[0]
    assert sender != main_thread
    assert target == main_thread
    assert signum == signal.SIGINT


def test_worker_thread_run_keeps_process_sigint_handler() -> None:
    previous = signal.getsignal(signal.SIGINT)

    def author(context: RunContext) -> None:
        assert signal.getsignal(signal.SIGINT) is previous

    adapter = notebook(InterruptCore(author))
    with ThreadPoolExecutor(max_workers=1) as executor:
        # Submitted synchronous core returns; timeout is only a deadlock watchdog.
        record = executor.submit(adapter.run, Cfg()).result(timeout=30)
    assert adapter.acquisition_for(record).completion == "complete"
    assert signal.getsignal(signal.SIGINT) is previous


def test_first_sigint_requests_stop_and_publishes_returned_record() -> None:
    def author(context: RunContext) -> None:
        signal.raise_signal(signal.SIGINT)
        assert context.scope.stop.is_set()
        assert context.scope.stop.reason

    adapter = notebook(InterruptCore(author))
    previous_handler = signal.getsignal(signal.SIGINT)
    try:
        record = adapter.run(Cfg(scale=7.0))
    except KeyboardInterrupt:
        pytest.fail("The first SIGINT must request a cooperative stop")
    assert record.result == 7.0
    assert adapter.last_run is record
    assert adapter.acquisition_for(record).completion == "stopped"
    assert signal.getsignal(signal.SIGINT) is previous_handler
