"""Focused tests for autofluxdep shared-session run locks."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import pytest
from zcu_tools.gui.app.autofluxdep.run_locks import GuardedDeviceControl
from zcu_tools.gui.session.device_control import DeviceControlPort


class _RecordingDeviceControl:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.try_poll_result = True

    def poll_device_info(self, name: str) -> None:
        self.calls.append(("poll_device_info", (name,)))

    def try_poll_device_info(self, name: str) -> bool:
        self.calls.append(("try_poll_device_info", (name,)))
        return self.try_poll_result

    def start_connect_device(self, req: object) -> int:
        self.calls.append(("start_connect_device", (req,)))
        return 1

    def start_disconnect_device(self, req: object) -> int:
        self.calls.append(("start_disconnect_device", (req,)))
        return 2

    def start_reconnect_device(self, name: str) -> int:
        self.calls.append(("start_reconnect_device", (name,)))
        return 3

    def start_setup_device(self, req: object) -> int:
        self.calls.append(("start_setup_device", (req,)))
        return 4

    def forget_device(self, name: str) -> None:
        self.calls.append(("forget_device", (name,)))

    def cancel_device_operation(self, name: str) -> None:
        self.calls.append(("cancel_device_operation", (name,)))


def _locked_guard(kind: str) -> None:
    raise RuntimeError(f"{kind} is locked while a run is active")


@pytest.mark.parametrize(
    "mutation",
    (
        lambda dev: dev.start_connect_device(cast(Any, object())),
        lambda dev: dev.start_disconnect_device(cast(Any, object())),
        lambda dev: dev.start_reconnect_device("flux"),
        lambda dev: dev.start_setup_device(cast(Any, object())),
        lambda dev: dev.forget_device("flux"),
        lambda dev: dev.cancel_device_operation("flux"),
    ),
)
def test_guarded_device_refresh_delegates_while_mutations_stay_locked(
    mutation: Callable[[GuardedDeviceControl], object],
) -> None:
    inner = _RecordingDeviceControl()
    guarded = GuardedDeviceControl(cast(DeviceControlPort, inner), _locked_guard)

    guarded.poll_device_info("flux")
    assert guarded.try_poll_device_info("flux") is True

    assert inner.calls == [
        ("poll_device_info", ("flux",)),
        ("try_poll_device_info", ("flux",)),
    ]
    with pytest.raises(RuntimeError, match="device is locked while a run is active"):
        mutation(guarded)
    assert inner.calls == [
        ("poll_device_info", ("flux",)),
        ("try_poll_device_info", ("flux",)),
    ]


def test_guarded_device_try_poll_preserves_inner_skip_result() -> None:
    inner = _RecordingDeviceControl()
    inner.try_poll_result = False
    guarded = GuardedDeviceControl(cast(DeviceControlPort, inner), _locked_guard)

    assert guarded.try_poll_device_info("flux") is False

    assert inner.calls == [("try_poll_device_info", ("flux",))]
