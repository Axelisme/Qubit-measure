"""Recording gate, progress and background adapters for runner seam tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

from zcu_tools.gui.event_bus import BaseEventBus, OriginKind
from zcu_tools.gui.session.operation_handles import OperationHandles
from zcu_tools.gui.session.operation_runner import OperationRunner
from zcu_tools.gui.session.ports import OperationConflictError


class FakeGate:
    """Minimal ExclusionGate: records calls, optionally raises on ensure."""

    def __init__(self, *, conflict: bool = False) -> None:
        self._conflict = conflict
        self.ensure_calls: list[tuple[str, str | None]] = []
        self.register_calls: list[
            tuple[int, str, str, OriginKind, str, str | None]
        ] = []
        self.release_calls: list[int] = []

    def ensure_can_start(self, kind: str, *, resource_id: str | None = None) -> None:
        self.ensure_calls.append((kind, resource_id))
        if self._conflict:
            raise OperationConflictError("conflict")

    def register(
        self,
        token: int,
        kind: str,
        *,
        owner_id: str,
        origin_kind: OriginKind,
        note: str,
        resource_id: str | None = None,
    ) -> None:
        self.register_calls.append(
            (token, kind, owner_id, origin_kind, note, resource_id)
        )

    def release(self, token: int) -> None:
        self.release_calls.append(token)

    def is_device_mutating(self, name: str) -> bool:
        return False


class FakeBg:
    """Synchronous background executor stub.

    When ``fail_submit`` is True, submit() raises immediately (simulating a
    synchronous submit failure before any work starts).
    Otherwise, submit captures the callbacks; call ``deliver_result`` or
    ``deliver_error`` to drive the on_done / on_error path.
    """

    def __init__(self, *, fail_submit: bool = False) -> None:
        self._fail_submit = fail_submit
        self._on_done: Callable[[Any], None] | None = None
        self._on_error: Callable[[Exception], None] | None = None

    def submit(
        self,
        work: Callable[[], Any],
        *,
        run_in_pool: bool,
        on_done: Callable[[Any], None],
        on_error: Callable[[Exception], None],
    ) -> None:
        if self._fail_submit:
            raise RuntimeError("submit boom")
        # Keep work and callbacks so tests can drive them.
        self._work = work
        self._on_done = on_done
        self._on_error = on_error

    def deliver_result(self) -> None:
        """Execute the captured work thunk and pass its return value to on_done."""
        assert self._work is not None and self._on_done is not None
        result = self._work()
        self._on_done(result)

    def deliver_error(self, exc: Exception) -> None:
        assert self._on_error is not None
        self._on_error(exc)


class FakeProgress:
    """Tracks make_factory and discard_operation calls."""

    def __init__(self, *, fail_discard: bool = False) -> None:
        self._fail_discard = fail_discard
        self.factories: dict[int, MagicMock] = {}
        self.discard_calls: list[int] = []

    def make_factory(self, operation_id: int, owner_id: str) -> MagicMock:
        factory = MagicMock(name=f"factory-{operation_id}")
        self.factories[operation_id] = factory
        return factory

    def discard_operation(self, operation_id: int) -> None:
        self.discard_calls.append(operation_id)
        if self._fail_discard:
            raise RuntimeError("discard boom")


def make_runner(
    *,
    gate: FakeGate | None = None,
    handles: OperationHandles | None = None,
    progress: FakeProgress | None = None,
    bg: FakeBg | None = None,
    bus: BaseEventBus | None = None,
) -> tuple[OperationRunner, FakeGate, OperationHandles, FakeProgress, FakeBg]:
    g = gate or FakeGate()
    h = handles or OperationHandles()
    p = progress or FakeProgress()
    b = bg or FakeBg()
    runner = OperationRunner(g, h, p, b, bus or BaseEventBus())
    return runner, g, h, p, b
