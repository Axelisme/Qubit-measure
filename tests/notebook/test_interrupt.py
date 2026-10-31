"""Notebook SIGINT requests cooperate with a synchronous experiment core."""

import signal
import threading
from collections.abc import Callable
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


def notebook(core: InterruptCore) -> NotebookExperiment[InterruptCore]:
    return NotebookAdapter(
        soc=object(), soccfg=object(), device_manager=DeviceManager(), host=NonPresentingHost()
    )(core)


class ManualTimer:
    """Deliver the external timer deadline only when the test advances it."""

    def __init__(self, interval: float, function: Callable[[], None]) -> None:
        self.interval = interval
        self.function = function
        self.started = False
        self.cancelled = False

    def start(self) -> None:
        self.started = True

    def cancel(self) -> None:
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


def test_timeout_interrupts_core_and_retains_previous_record(timers: list[ManualTimer]) -> None:
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


def test_return_cancels_deadline_before_restoring_handler(timers: list[ManualTimer]) -> None:
    received: list[int] = []

    def previous(signum: int, frame: FrameType | None) -> None:
        received.append(signum)

    original = signal.signal(signal.SIGINT, previous)
    try:
        adapter = notebook(InterruptCore(lambda context: signal.raise_signal(signal.SIGINT)))
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
