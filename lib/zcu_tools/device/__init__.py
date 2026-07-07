from __future__ import annotations

from typing import TypeAlias

from .base import BaseDevice, BaseDeviceInfo, DeviceBusyError
from .fake import FakeDevice, FakeDeviceInfo
from .manager import GlobalDeviceManager
from .sgs100a import RohdeSchwarzSGS100A, RohdeSchwarzSGS100AInfo
from .yoko import YOKOGS200, YOKOGS200Info
from .mg3692 import AnritsuMG3692, AnritsuMG3692Info

DeviceInfo: TypeAlias = (
    YOKOGS200Info | RohdeSchwarzSGS100AInfo | FakeDeviceInfo | AnritsuMG3692Info
)


__all__ = [
    # base
    "BaseDevice",
    "BaseDeviceInfo",
    "DeviceBusyError",
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
