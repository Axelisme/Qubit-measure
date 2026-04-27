from __future__ import annotations

from typing_extensions import Union, TypeAlias

from .base import BaseDevice, BaseDeviceInfo
from .yoko import YOKOGS200, YOKOGS200Info
from .sgs100a import RohdeSchwarzSGS100A, RohdeSchwarzSGS100AInfo
from .fake import FakeDevice, FakeDeviceInfo
from .manager import GlobalDeviceManager

DeviceInfo: TypeAlias = Union[YOKOGS200Info, RohdeSchwarzSGS100AInfo, FakeDeviceInfo]


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
