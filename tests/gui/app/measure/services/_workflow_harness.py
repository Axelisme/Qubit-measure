"""Recording operation boundaries shared by workflow startup contract tests."""

import logging
from collections.abc import Callable
from threading import Event, Thread
from time import monotonic

from zcu_tools.device import DeviceManager, FakeDevice
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.session.adapters.manual_owner_scheduler import ManualOwnerScheduler
from zcu_tools.gui.session.hardware_gate import RunBlocksHardwareGate
from zcu_tools.gui.session.operation_handles import OperationHandles
from zcu_tools.gui.session.operation_runner import (
    OperationInfrastructure,
    OperationRunner,
)
from zcu_tools.gui.session.ports import ProgressEvent
from zcu_tools.gui.session.services.build import build_session_services
from zcu_tools.gui.session.services.progress import ProgressService
from zcu_tools.gui.session.state import SessionState
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase


class DeferredBackground:
    """Separate submit, worker return and owner delivery for lifecycle assertions.

    Set fail_submit to reject only the next submission with OSError; later
    submissions, including terminal readback, retain normal execution/delivery.
    on_submit observes each submission until the caller clears it.
    """

    def __init__(self) -> None:
        self.fail_submit = False
        self.on_submit: Callable[[], None] | None = None
        self.work: Callable[[], object] | None = None
        self.on_done: Callable[[object], None] | None = None
        self.on_error: Callable[[Exception], None] | None = None
        self.result: object = None
        self.error: Exception | None = None
        self.worker: Thread | None = None
        self.completed = Event()

    def submit(
        self,
        work: Callable[[], object],
        *,
        run_in_pool: bool,
        on_done: Callable[[object], None],
        on_error: Callable[[Exception], None],
    ) -> None:
        # Observe continuation admission before a producer exists.
        if self.on_submit is not None:
            self.on_submit()
        if self.fail_submit:
            self.fail_submit = False
            raise OSError("submit failed")
        if self.work is not None:
            raise RuntimeError("another deferred worker is pending")
        self.work, self.on_done, self.on_error = work, on_done, on_error
        self.result, self.error = None, None

    def execute(self) -> None:
        if self.work is None:
            raise RuntimeError("no submitted worker")
        try:
            self.result = self.work()
        except Exception as error:
            logging.exception("Deferred test worker failed")
            self.error = error
        finally:
            self.completed.set()

    def start(self) -> None:
        """Start the retained invocation without delivering owner terminal callbacks."""
        if self.worker is not None:
            raise RuntimeError("a retained worker already exists")
        self.completed.clear()
        self.worker = Thread(target=self.execute)
        self.worker.start()

    def deliver(self) -> None:
        if self.on_done is None or self.on_error is None:
            raise RuntimeError("no terminal callbacks")
        done, failed = self.on_done, self.on_error
        self.work = self.on_done = self.on_error = None
        if self.error is not None:
            failed(self.error)
        else:
            done(self.result)


class OwnerProgressTransport:
    """Deliver real native progress through the explicitly pumped owner."""

    def __init__(self, owner: ManualOwnerScheduler) -> None:
        self.owner = owner
        self.receiver: Callable[[ProgressEvent], None] | None = None

    def set_receiver(self, receiver: Callable[[ProgressEvent], None]) -> None:
        self.receiver = receiver

    def emit(self, event: ProgressEvent) -> None:
        receiver = self.receiver
        if receiver is None:
            raise RuntimeError("no progress receiver")
        self.owner.post(lambda: receiver(event))


class StartHarness:
    """Real session owners with a deferred executor, no connected hardware."""

    def __init__(self) -> None:
        self.owner = ManualOwnerScheduler()
        self.bus = BaseEventBus()
        self.gate = RunBlocksHardwareGate(run_kind="run", bus=self.bus)
        self.handles = OperationHandles()
        self.background = DeferredBackground()
        self.progress = ProgressService(OwnerProgressTransport(self.owner))
        self.runner = OperationRunner(
            self.gate,
            self.handles,
            self.progress,
            self.background,
            self.bus,
        )
        self.state = SessionState(SessionEnv(parameters=None, soc=None, soccfg=None))
        self.devices = DeviceManager()
        self.services = build_session_services(
            state=self.state,
            bus=self.bus,
            operations=OperationInfrastructure(
                gate=self.gate,
                background=self.background,
                runner=self.runner,
                handles=self.handles,
            ),
            progress=self.progress,
            bind_assets=ArbWaveformDatabase.info_at,
            device_registry=self.devices,
        )

    def publish_devices(self, *names: str) -> None:
        """Publish synthetic startup drivers through the session's public owner."""
        for name in names:
            self.devices.register_device(name, FakeDevice(address=name, fast_mode=True))
        self.services.device.publish_initial_devices()

    def wait_for(self, completed: Callable[[], bool], *, label: str) -> None:
        """Pump the manual owner until a public condition holds, with a watchdog."""
        deadline = monotonic() + 30
        while not completed():
            if monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for {label}")
            self.owner.pump_once(block=True, timeout=0.01)

    def finish_worker(self) -> None:
        """Wait for the retained worker, then deliver its terminal callbacks once."""
        worker = self.background.worker
        if worker is None:
            raise RuntimeError("no retained worker")
        self.wait_for(self.background.completed.is_set, label="workflow worker return")
        worker.join(timeout=30)
        if worker.is_alive():
            raise TimeoutError("Workflow worker did not exit")
        self.background.worker = None
        self.owner.pump_all()
        self.background.deliver()

    def complete_segment(self) -> None:
        """Run the deferred worker, pump its progress, then deliver owner terminal."""
        self.background.execute()
        self.owner.pump_all()
        self.background.deliver()
