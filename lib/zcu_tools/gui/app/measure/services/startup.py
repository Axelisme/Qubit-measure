"""Trusted startup-script lifecycle, sharing the app's operation runner and drain."""

from __future__ import annotations

import logging
import runpy
import traceback
from collections.abc import Callable
from pathlib import Path

from zcu_tools.gui.app.measure.events.startup import StartupChangedPayload
from zcu_tools.gui.app.measure.services.operation_gate import OperationKind
from zcu_tools.gui.app.measure.startup_context import StartupContext
from zcu_tools.gui.app.measure.state import State
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.owner import OwnerThreadGuard
from zcu_tools.gui.session.operation_handles import OperationOutcome
from zcu_tools.gui.session.operation_runner import (
    BgResult,
    ExclusionRequest,
    OperationRunner,
    OperationSpec,
    SettleFn,
)

logger = logging.getLogger(__name__)


def _execute_script(path: Path, context: StartupContext) -> None:
    namespace = runpy.run_path(str(path))
    startup: object = namespace.get("startup")
    if not callable(startup):
        raise TypeError(f"{path} must define callable startup(ctx)")
    startup(context)


def _describe_failure(path: Path, error: BaseException) -> str:
    frames = traceback.extract_tb(error.__traceback__)
    script_frames = [frame for frame in frames if Path(frame.filename) == path]
    line = script_frames[-1].lineno if script_frames else None
    if isinstance(error, SyntaxError):
        line = error.lineno
    location = str(path) if line is None else f"{path}, line {line}"
    return f"Startup failed: {location}: {type(error).__name__}: {error}"


class StartupService:
    """Own script admission, progress and reverse-order close callbacks.

    runner is the application's shared operation runner, so script work belongs
    to Controller.close's normal background drain. bus publishes owner-thread
    StartupChangedPayload facts. state owns the live startup-activity projection
    read by tab controls; it is never persisted. All public methods run on the constructing
    owner thread. Only script execution runs off-owner. The service never owns
    session data, disk cache formats, devices or caller-owned board resources.
    """

    def __init__(
        self, runner: OperationRunner, bus: BaseEventBus, state: State
    ) -> None:
        self._owner = OwnerThreadGuard()
        self._runner = runner
        self._bus = bus
        self._started = False
        self._state = state
        self._closing = False
        self._callbacks: list[Callable[[], None]] = []

    def start(self, path: Path, context: StartupContext) -> int:
        """Run trusted path's startup(context) once in a dedicated background worker.

        Return the normal operation token. Missing files, load errors and script
        exceptions settle failed and publish path/line diagnostics; successful
        commands and callbacks remain installed. Run is excluded until terminal
        delivery, but script hardware commands retain their normal gates.
        Repeat start or start after callback teardown raises RuntimeError.
        Synchronous submission failures propagate and release the startup lease.
        """
        self._owner.assert_owner()
        if self._started or self._closing:
            raise RuntimeError("startup can only run once before close")
        path = path.resolve()
        self._started = True

        def opened(_token: int) -> None:
            self._state.set_startup_active(active=True)

        def terminal(result: BgResult, settle: SettleFn) -> None:
            self._state.set_startup_active(active=False)
            error = None
            if result.error is not None:
                error = _describe_failure(path, result.error)
                logger.error(error, exc_info=result.error)
            settle(OperationOutcome("finished" if error is None else "failed", error))
            message = f"Startup failed: {path.name}" if error else "Startup complete"
            self._bus.emit(StartupChangedPayload(False, message, error))

        try:
            token = self._runner.begin(
                OperationSpec(
                    exclusion=ExclusionRequest(
                        OperationKind.STARTUP, "startup", f"Startup: {path.name}"
                    ),
                    owner_id="startup",
                    wants_progress=False,
                    cancel_hook=None,
                    work=lambda _factory: _execute_script(path, context),
                    run_in_pool=False,
                    on_terminal=terminal,
                    on_opened=opened,
                )
            )
        except Exception:
            self._state.set_startup_active(active=False)
            raise
        self._bus.emit(StartupChangedPayload(True, f"Startup: {path.name}"))
        return token

    def progress(self, message: str) -> None:
        """Publish nonblank script progress on-owner; reject after terminal delivery."""
        self._owner.assert_owner()
        if not self._state.is_startup_active():
            raise RuntimeError("startup is not running")
        if not message.strip():
            raise ValueError("startup progress must not be blank")
        self._bus.emit(StartupChangedPayload(True, message))

    def on_close(self, callback: Callable[[], None]) -> None:
        """Retain callback for close_callbacks, including after partial script failure.

        Owner-only. callback takes no arguments; noncallable values raise
        TypeError. Registration once teardown begins raises RuntimeError.
        """
        self._owner.assert_owner()
        if self._closing:
            raise RuntimeError("startup callbacks are already closing")
        if not callable(callback):
            raise TypeError("on_close callback must be callable")
        self._callbacks.append(callback)

    def close_callbacks(self) -> None:
        """After successful application drain, run callbacks in reverse order once.

        Owner-only; a live script raises RuntimeError before consuming callbacks.
        Each ordinary callback failure is logged and isolated. Repeated calls
        do nothing. This does not drain workers or close hardware; the Controller
        owns that ordering. Callbacks may use synchronous script facets on-owner.
        """
        self._owner.assert_owner()
        if self._state.is_startup_active():
            raise RuntimeError("cannot close callbacks while startup is running")
        if self._closing:
            return
        self._closing = True
        while self._callbacks:
            callback = self._callbacks.pop()
            try:
                callback()
            except Exception:
                logger.exception("Startup on_close callback failed")
