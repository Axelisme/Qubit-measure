from __future__ import annotations

from typing import TypeAlias, Union

from .base import BaseDevice, BaseDeviceInfo
from .fake import FakeDevice, FakeDeviceInfo
from .manager import GlobalDeviceManager
from .sgs100a import RohdeSchwarzSGS100A, RohdeSchwarzSGS100AInfo
from .yoko import YOKOGS200, YOKOGS200Info

DeviceInfo: TypeAlias = YOKOGS200Info | RohdeSchwarzSGS100AInfo | FakeDeviceInfo


__all__ = [
    # base
    "BaseDevice",
    "BaseDeviceInfo",
    # manager
    "GlobalDeviceManager",
    # devices
    "YOKOGS200",
    "YOKOGS200Info",
    "RohdeSchwarzSGS100A",
    "RohdeSchwarzSGS100AInfo",
    "FakeDevice",
    "FakeDeviceInfo",
    # other
    "DeviceInfo",
]
