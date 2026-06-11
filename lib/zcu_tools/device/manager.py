from __future__ import annotations

import threading
import warnings
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar

from .base import BaseDevice

if TYPE_CHECKING:
    from . import DeviceInfo


class GlobalDeviceManager:
    _devices: ClassVar[dict[str, BaseDevice]] = {}
    _lock: ClassVar[threading.RLock] = threading.RLock()

    @classmethod
    def register_device(cls, name: str, device: Any) -> None:
        with cls._lock:
            if name in cls._devices:
                warnings.warn(f"Device {name} already registered, overwriting")
            cls._devices[name] = device

    @classmethod
    def drop_device(cls, name: str, ignore_error: bool = False) -> None:
        with cls._lock:
            if name not in cls._devices:
                if ignore_error:
                    return
                raise ValueError(f"Device {name} not found")
            del cls._devices[name]

    @classmethod
    def get_device(cls, name: str) -> BaseDevice:
        with cls._lock:
            if name not in cls._devices:
                raise ValueError(f"Device {name} not found")
            return cls._devices[name]

    @classmethod
    def get_all_devices(cls) -> dict[str, BaseDevice]:
        with cls._lock:
            return dict(cls._devices)

    @classmethod
    def setup_devices(
        cls, dev_cfg: Mapping[str, DeviceInfo], *, progress: bool = True
    ) -> None:
        # Validate all names and snapshot references under the registry lock so
        # that the check-then-act is atomic with respect to concurrent
        # register/drop calls.  Fast-fail: any unknown name aborts the whole
        # batch before any setup begins.
        with cls._lock:
            for name in dev_cfg:
                if name not in cls._devices:
                    raise ValueError(f"Device {name} not found")
            # Snapshot instance references; registry mutations after this point
            # do not affect which instances we are about to configure.
            snapshot: list[tuple[BaseDevice, DeviceInfo]] = [
                (cls._devices[name], cfg) for name, cfg in dev_cfg.items()
            ]

        # Per-instance lock (BaseDevice._lock) serializes each setup() call.
        # Busy devices raise DeviceBusyError immediately (fail-fast); we do not
        # swallow that error.
        for device, cfg in snapshot:
            device.setup(cfg, progress=progress)

    @classmethod
    def get_info(cls, name: str) -> DeviceInfo:
        # Resolve the instance under the registry lock; call get_info() outside
        # it so a long-running setup() on another device cannot block this read.
        device = cls.get_device(name)
        return device.get_info()  # type: ignore[return-value]

    @classmethod
    def get_all_info(cls) -> dict[str, DeviceInfo]:
        # Snapshot the registry under the lock, then query each device outside
        # it so concurrent setup() calls on individual devices do not block
        # the whole registry for the duration of their I/O.
        snapshot = cls.get_all_devices()
        return {name: device.get_info() for name, device in snapshot.items()}  # type: ignore[return-value]
