"""ProgressService — the single owner of live progress-bar telemetry.

Qt-free, lock-free, main-thread-only. It receives ``ProgressEvent``s already
marshalled to the main thread by a ``ProgressTransport`` (the Qt marshal is a
driven adapter), so it never touches Qt and never needs a lock.

Model:
  - one ``ProgressContainer`` per operation (keyed by ``operation_id``); a
    worker may open several bars, each a ``handle_id`` within its container.
  - container lifetime is bound to the *operation* (created at ``make_factory``,
    destroyed at ``discard_operation``), **not** to any View — so closing a
    dialog mid-setup does not destroy progress; re-attaching shows it again.
  - Views attach by ``owner_id`` (a tab_id / device_name they already hold). The
    service maps owner → current live operation_id, so a View attaches once and
    automatically follows the owner across successive operations.

This service shares nothing with ``OperationGate`` but the ``operation_id``
integer (which RunService/DeviceService hand to both); zero import, zero call,
zero lifecycle coupling between the two.
"""

from __future__ import annotations

import itertools
import time
from typing import Any, Callable

from zcu_tools.gui.pbar_host import ProgressBar, ProgressBarModel
from zcu_tools.gui.services.ports import (
    ProgressEvent,
    ProgressEventKind,
    ProgressTransport,
)

Listener = Callable[[], None]
Disposer = Callable[[], None]


class ProgressContainer:
    """All live bars of one operation. Qt-free; mutated only on the main thread."""

    def __init__(self) -> None:
        self._bars: dict[int, ProgressBarModel] = {}

    def apply(self, event: ProgressEvent, *, now: float) -> None:
        if event.kind is ProgressEventKind.CREATE:
            self._bars[event.handle_id] = ProgressBarModel(
                event.label, event.total, now
            )
        elif event.kind is ProgressEventKind.UPDATE:
            model = self._bars.get(event.handle_id)
            if model is None:
                # update-before-create (e.g. a throttled-away CREATE): build now.
                model = ProgressBarModel(event.label, event.total, now)
                self._bars[event.handle_id] = model
            model.set_label(event.label)
            model.set_n(event.n)
        elif event.kind is ProgressEventKind.CLOSE:
            self._bars.pop(event.handle_id, None)

    def bars(self) -> tuple[tuple[int, ProgressBarModel], ...]:
        return tuple(self._bars.items())


class BoundProgressFactory:
    """A ``make_pbar`` factory bound to one operation; mints handle ids locally.

    Injected into a worker via the ``use_pbar_factory`` ContextVar. The worker
    calls it like tqdm; it never sees the transport-level ids beyond what the
    bar carries.
    """

    def __init__(self, transport: ProgressTransport, operation_id: int) -> None:
        self._transport = transport
        self._operation_id = operation_id
        self._handles = itertools.count()

    def __call__(self, *args: Any, **kwargs: Any) -> ProgressBar:
        label = kwargs.pop("desc", "") or (args[1] if len(args) > 1 else "")
        total = kwargs.pop("total", None) or (args[2] if len(args) > 2 else None)
        leave = kwargs.pop("leave", True)
        disabled = bool(kwargs.pop("disable", False))
        return ProgressBar(
            self._transport,
            self._operation_id,
            next(self._handles),
            label=str(label) if label else "",
            total=total,
            leave=leave,
            disabled=disabled,
        )


class ProgressService:
    """Owns ``operation_id -> ProgressContainer`` and ``owner_id -> live op``.

    Does not hold the OperationGate (decision 2): RunService/DeviceService tell
    it the owner_id at ``make_factory`` time, so it never reverse-looks-up the
    gate.
    """

    def __init__(self, transport: ProgressTransport) -> None:
        self._transport = transport
        self._containers: dict[int, ProgressContainer] = {}
        self._live_op: dict[str, int] = {}  # owner_id -> current live operation_id
        # Listeners registered per owner (not per container): a container is
        # swapped out as the owner's operation rotates, but the View's listener
        # must survive that — it re-queries bars_for_owner on each notify.
        self._listeners: dict[str, list[Listener]] = {}
        self._owner_of: dict[int, str] = {}  # operation_id -> owner_id (for notify)
        transport.set_receiver(self._on_event)

    # -- worker factory minting (called by RunService/DeviceService after acquire)
    def make_factory(self, operation_id: int, owner_id: str) -> BoundProgressFactory:
        self._containers[operation_id] = ProgressContainer()
        self._live_op[owner_id] = operation_id
        self._owner_of[operation_id] = owner_id
        self._notify_owner(owner_id)
        return BoundProgressFactory(self._transport, operation_id)

    # -- main-thread deliver (receiver guarantees this runs on the main thread)
    def _on_event(self, event: ProgressEvent) -> None:
        container = self._containers.get(event.operation_id)
        if container is None:
            # Event for an operation we never registered / already discarded.
            return
        container.apply(event, now=time.monotonic())
        owner = self._owner_of.get(event.operation_id)
        if owner is not None:
            self._notify_owner(owner)

    # -- reads (RPC + UI)
    def bars_for_owner(self, owner_id: str) -> tuple[tuple[int, ProgressBarModel], ...]:
        op = self._live_op.get(owner_id)
        if op is None:
            return ()
        container = self._containers.get(op)
        return container.bars() if container is not None else ()

    # -- UI attach (View attaches once by its own owner_id; follows op rotation)
    def attach_by_owner(self, owner_id: str, listener: Listener) -> Disposer:
        self._listeners.setdefault(owner_id, []).append(listener)

        def _dispose() -> None:
            listeners = self._listeners.get(owner_id)
            if listeners is not None and listener in listeners:
                listeners.remove(listener)
                if not listeners:
                    del self._listeners[owner_id]

        return _dispose

    # -- operation terminal cleanup (RunService/DeviceService on finish/fail)
    def discard_operation(self, operation_id: int) -> None:
        self._containers.pop(operation_id, None)
        owner = self._owner_of.pop(operation_id, None)
        if owner is not None:
            # Only clear the live mapping if it still points at this operation
            # (a newer operation for the same owner may already have replaced it).
            if self._live_op.get(owner) == operation_id:
                del self._live_op[owner]
            self._notify_owner(owner)

    def _notify_owner(self, owner_id: str) -> None:
        for listener in tuple(self._listeners.get(owner_id, ())):
            listener()
