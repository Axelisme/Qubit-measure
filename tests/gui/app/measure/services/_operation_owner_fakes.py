"""Typed operation-owner fakes for OperationControlFacet tests."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from zcu_tools.gui.app.measure.services.run_analyze_control import ActiveTabOperation
from zcu_tools.gui.session.pbar_host import ProgressBarModel


@dataclass(frozen=True)
class DeviceOperation:
    token: int
    device_name: str = "device"


@dataclass
class TabOperationOwner:
    """Lists fixed tab operations; cancelling is an error unless a hook is given."""

    operations: tuple[ActiveTabOperation, ...] = ()
    on_cancel_run: Callable[[], bool] | None = None

    def active_tab_operations(self) -> tuple[ActiveTabOperation, ...]:
        return self.operations

    def cancel_run(self) -> bool:
        if self.on_cancel_run is None:
            raise AssertionError("unexpected cancel_run")
        return self.on_cancel_run()

    def cancel_analyze(self, tab_id: str) -> bool:
        raise AssertionError(f"unexpected cancel_analyze: {tab_id}")


@dataclass
class DeviceOperationOwner:
    """Lists fixed device operations; cancelling is an error unless a hook is given."""

    operations: tuple[DeviceOperation, ...] = ()
    on_cancel: Callable[[str], None] | None = None

    def get_active_device_operations(self) -> tuple[DeviceOperation, ...]:
        return self.operations

    def cancel_device_operation(self, name: str) -> None:
        if self.on_cancel is None:
            raise AssertionError(f"unexpected cancel_device_operation: {name}")
        self.on_cancel(name)


class SaveOperationOwner:
    def active_save_operations(self) -> tuple[()]:
        return ()


class UnusedProgress:
    def bars_for_operation(
        self, operation_id: int
    ) -> tuple[tuple[int, ProgressBarModel], ...]:
        raise AssertionError(f"unexpected progress read: {operation_id}")
