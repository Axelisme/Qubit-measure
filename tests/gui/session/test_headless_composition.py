"""Executable definition of the Qt-free session-core composition."""

from __future__ import annotations

import importlib.abc
import os
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

_CHILD_FLAG = "--headless-child"
_QT_ROOTS = frozenset({"qtpy", "PyQt6", "PySide6"})


class _BlockQtImports(importlib.abc.MetaPathFinder):
    def find_spec(
        self,
        fullname: str,
        path: object = None,
        target: object = None,
    ) -> None:
        del path, target
        if fullname.partition(".")[0] in _QT_ROOTS:
            raise ImportError(f"Qt import blocked in headless smoke: {fullname}")
        return None


def _install_qt_import_blocker() -> None:
    sys.meta_path.insert(0, _BlockQtImports())


def _run_headless_smoke() -> None:
    from zcu_tools.device.fake import FakeDevice
    from zcu_tools.gui.event_bus import BaseEventBus
    from zcu_tools.gui.session.adapters.manual_owner_scheduler import (
        ManualOwnerScheduler,
    )
    from zcu_tools.gui.session.adapters.thread_pool_background import (
        ThreadPoolBackgroundExecutor,
    )
    from zcu_tools.gui.session.events import (
        ConnectionFinishedPayload,
        GateChangedPayload,
    )
    from zcu_tools.gui.session.hardware_gate import RunBlocksHardwareGate
    from zcu_tools.gui.session.operation_handles import (
        OperationHandles,
        OperationOutcome,
    )
    from zcu_tools.gui.session.operation_runner import (
        BgResult,
        ExclusionRequest,
        OperationRunner,
        OperationSpec,
        SettleFn,
    )
    from zcu_tools.gui.session.ports import ProgressEvent
    from zcu_tools.gui.session.services.build import build_session_services
    from zcu_tools.gui.session.services.connection import ConnectMockRequest
    from zcu_tools.gui.session.services.io_manager import IOManager
    from zcu_tools.gui.session.services.progress import ProgressService
    from zcu_tools.gui.session.state import (
        DeviceState,
        DeviceStatus,
        SessionState,
        StartupPrefs,
    )
    from zcu_tools.gui.session.types import ExpContext
    from zcu_tools.meta_tool import MetaDict, ModuleLibrary

    owner_id = threading.get_ident()
    owner = ManualOwnerScheduler()

    class OwnerProgressTransport:
        def __init__(self) -> None:
            self._receiver: Callable[[ProgressEvent], None] | None = None

        def emit(self, event: ProgressEvent) -> None:
            receiver = self._receiver
            if receiver is None:
                raise RuntimeError("progress receiver is not installed")
            owner.post(lambda: receiver(event))

        def set_receiver(self, receiver: Callable[[ProgressEvent], None]) -> None:
            if self._receiver is not None:
                raise RuntimeError("progress receiver is already installed")
            self._receiver = receiver

    class InMemoryDeviceRegistry:
        def __init__(self) -> None:
            self._devices: dict[str, Any] = {}

        def register_device(self, name: str, device: Any) -> None:
            self._devices[name] = device

        def drop_device(self, name: str, ignore_error: bool = False) -> None:
            if name not in self._devices:
                if ignore_error:
                    return
                raise ValueError(f"unknown device: {name}")
            del self._devices[name]

        def get_device(self, name: str) -> Any:
            return self._devices[name]

        def get_all_devices(self) -> dict[str, Any]:
            return dict(self._devices)

        def get_info(self, name: str) -> Any:
            return self.get_device(name).get_info()

    def unexpected_driver_factory(type_name: str, address: str) -> Any:
        raise AssertionError(
            f"headless smoke unexpectedly requested {type_name} at {address}"
        )

    def pump_until(predicate: Callable[[], bool], *, timeout: float = 5.0) -> None:
        deadline = time.monotonic() + timeout
        while not predicate():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AssertionError("owner loop did not reach the expected state")
            owner.pump_once(block=True, timeout=min(0.05, remaining))

    state = SessionState(
        ExpContext(
            md=MetaDict(),
            ml=ModuleLibrary(),
            soc=None,
            soccfg=None,
        )
    )
    bus = BaseEventBus()
    gate = RunBlocksHardwareGate(run_kind="run", bus=bus)
    handles = OperationHandles()
    progress = ProgressService(OwnerProgressTransport())
    background = ThreadPoolBackgroundExecutor(owner, max_pool_workers=2)
    runner = OperationRunner(gate, handles, progress, background, bus)
    registry = InMemoryDeviceRegistry()
    fake_flux = FakeDevice(fast_mode=True)
    registry.register_device("fake_flux", fake_flux)
    state.put_device(
        DeviceState(
            name="fake_flux",
            type_name="FakeDevice",
            address="none",
            status=DeviceStatus.CONNECTED,
            remember=True,
            info=fake_flux.get_info(),
        )
    )
    session = build_session_services(
        state=state,
        bus=bus,
        gate=gate,
        handles=handles,
        background=background,
        progress=progress,
        io_manager=IOManager(),
        runner=runner,
        driver_factory=unexpected_driver_factory,
        device_registry=registry,
    )

    gate_events: list[tuple[int, tuple[object, ...]]] = []
    bus.subscribe(
        GateChangedPayload,
        lambda payload: gate_events.append((threading.get_ident(), payload.active)),
    )
    terminal_threads: list[int] = []
    connection_events: list[ConnectionFinishedPayload] = []

    def record_connection(payload: ConnectionFinishedPayload) -> None:
        terminal_threads.append(threading.get_ident())
        connection_events.append(payload)

    bus.subscribe(ConnectionFinishedPayload, record_connection)
    worker_threads: list[int] = []

    success_token = session.soc_connection.start_connect(ConnectMockRequest())
    assert handles.live_count() == 1
    assert handles.poll(success_token) is None
    assert len(gate.snapshot()) == 1
    assert gate.snapshot()[0].note == "connect SoC (mock)"
    pump_until(lambda: handles.poll(success_token) is not None)
    success_outcome = handles.poll(success_token)
    assert success_outcome is not None and success_outcome.status == "finished"
    assert session.soc_connection.has_soc()
    assert session.soc_connection.is_mock_soc()
    assert state.exp_context.soc is not None
    assert state.exp_context.soccfg is not None
    assert state.version.get("soc") == 1
    assert len(connection_events) == 1 and connection_events[0].success
    assert progress.bars_for_operation(success_token) == ()
    assert gate.snapshot() == ()

    cancel_requested = threading.Event()
    cancel_started = threading.Event()

    def cancel_work(factory: Any) -> str:
        worker_threads.append(threading.get_ident())
        bar = factory(desc="headless-cancel", total=1)
        bar.update()
        cancel_started.set()
        if not cancel_requested.wait(timeout=5.0):
            raise TimeoutError("cancel hook was not called")
        return "cancelled"

    def cancel_terminal(result: BgResult, settle: SettleFn) -> None:
        terminal_threads.append(threading.get_ident())
        if not result.ok or result.result != "cancelled":
            raise AssertionError(f"unexpected cancel result: {result!r}")
        state.set_startup_prefs(StartupPrefs(chip_name="cancelled"))
        settle(OperationOutcome("cancelled"))

    cancel_token = runner.begin(
        OperationSpec(
            exclusion=ExclusionRequest(
                kind="run",
                owner_id="cancel-owner",
                note="headless cancel",
            ),
            owner_id="cancel-owner",
            wants_progress=True,
            cancel_hook=cancel_requested.set,
            work=cancel_work,
            run_in_pool=False,
            on_terminal=cancel_terminal,
        )
    )
    assert handles.live_count() == 1
    assert len(gate.snapshot()) == 1
    assert gate.snapshot()[0].note == "headless cancel"
    assert cancel_started.wait(timeout=3.0)
    pump_until(lambda: bool(progress.bars_for_operation(cancel_token)))
    handles.stop(cancel_token, "headless stop")
    pump_until(lambda: handles.poll(cancel_token) is not None)

    cancel_outcome = handles.poll(cancel_token)
    assert cancel_outcome is not None and cancel_outcome.status == "cancelled"
    awaited = handles.await_outcome(cancel_token, timeout=0.0)
    assert awaited is not None and awaited.feedback == "headless stop"
    assert state.startup_prefs.chip_name == "cancelled"
    assert progress.bars_for_operation(cancel_token) == ()
    assert gate.snapshot() == ()
    assert handles.live_count() == 0
    assert background.quiesce(timeout=5.0)

    assert terminal_threads == [owner_id, owner_id]
    assert worker_threads and all(thread_id != owner_id for thread_id in worker_threads)
    assert len(gate_events) == 4
    assert all(thread_id == owner_id for thread_id, _active in gate_events)
    assert [len(active) for _thread_id, active in gate_events] == [1, 0, 1, 0]
    assert not any(name.partition(".")[0] in _QT_ROOTS for name in sys.modules)


def test_session_core_runs_without_qt_imports_or_qapplication() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    env = os.environ.copy()
    python_path = str(repo_root / "lib")
    if current := env.get("PYTHONPATH"):
        python_path = os.pathsep.join((python_path, current))
    env["PYTHONPATH"] = python_path

    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), _CHILD_FLAG],
        cwd=repo_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=20.0,
    )
    assert completed.returncode == 0, (
        f"headless child failed with code {completed.returncode}\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )


if __name__ == "__main__" and sys.argv[1:] == [_CHILD_FLAG]:
    _install_qt_import_blocker()
    with tempfile.TemporaryDirectory(prefix="zcu-headless-mpl-") as mpl_config:
        os.environ["MPLCONFIGDIR"] = mpl_config
        _run_headless_smoke()
