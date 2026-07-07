from __future__ import annotations

from typing import TypeAlias

from .base import BaseDevice, BaseDeviceInfo, DeviceBusyError
from .cancel_scope import (
    current_device_setup_cancel_signal,
    device_setup_cancel_scope,
)
from .fake import FakeDevice, FakeDeviceInfo
from .manager import GlobalDeviceManager
from .mg3692 import AnritsuMG3692, AnritsuMG3692Info
from .sgs100a import RohdeSchwarzSGS100A, RohdeSchwarzSGS100AInfo
from .yoko import YOKOGS200, YOKOGS200Info

DeviceInfo: TypeAlias = (
    YOKOGS200Info | RohdeSchwarzSGS100AInfo | FakeDeviceInfo | AnritsuMG3692Info
)


__all__ = [
    # base
    "BaseDevice",
    "BaseDeviceInfo",
    "DeviceBusyError",
    "current_device_setup_cancel_signal",
    "device_setup_cancel_scope",
    # manager
    "GlobalDeviceManager",
    # devices
    "YOKOGS200",
    "YOKOGS200Info",
    "RohdeSchwarzSGS100A",
    "RohdeSchwarzSGS100AInfo",
    "FakeDevice",
    "FakeDeviceInfo",
    "AnritsuMG3692",
    "AnritsuMG3692Info",
    # other
    "DeviceInfo",
]
