from __future__ import annotations

import sys
import threading
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, Optional, Self, TypeVar

from zcu_tools.cfg_model import ConfigBase

if TYPE_CHECKING:
    from pyvisa import ResourceManager


class DeviceBusyError(RuntimeError):
    """Raised by BaseDevice.setup() when the device lock is already held by another
    thread.  setup() is fail-fast: it never queues behind an in-progress operation.
    Use is_busy() to inspect status for display/diagnostic purposes only."""


class BaseDeviceInfo(ConfigBase):
    address: str
    type: str
    label: str | None = None

    def with_updates(self, **kwargs) -> Self:
        protected_fields = {"type", "address"}
        for key in kwargs:
            if key in protected_fields:
                raise ValueError(
                    f"Cannot update protected field '{key}' in DeviceInfo."
                )
            if key not in type(self).model_fields:
                raise ValueError(
                    f"Unknown field '{key}' for {self.__class__.__name__}."
                )
        return super().with_updates(**kwargs)


T_DeviceInfo = TypeVar("T_DeviceInfo", bound=BaseDeviceInfo)


class BaseDevice(ABC, Generic[T_DeviceInfo]):
    """
    Base class for all devices.
    """

    info_model: type[BaseDeviceInfo] = BaseDeviceInfo

    def __init__(self, address: str, rm: ResourceManager) -> None:
        self.address = address

        # Per-instance reentrant lock serializing every public operation on this
        # device. RLock (not Lock) is required because public entry points nest on
        # the same thread: setup() -> _setup() -> set_current()/set_voltage() ->
        # output_on()/query(). The GlobalDeviceManager lock only guards the registry
        # (which instance answers to a name), not concurrent access to one instance's
        # SCPI session, so a separate per-instance lock is needed here.
        self._lock = threading.RLock()

        import pyvisa as visa

        try:
            self.session = rm.open_resource(address)
            self.session.read_termination = "\n"  # type: ignore
            self.session.write_termination = "\n"  # type: ignore
        except visa.Error:
            sys.stderr.write(f"Couldn't connect to {address}")
            raise

        # Print IDN message on connection
        self.connect_message()

    # ----- helper methods -----

    def connect_message(self) -> None:
        """Queries and prints the IDN to confirm connection."""
        import pyvisa

        with self._lock:
            try:
                idn = self.query("*IDN?")
                print(f"Connected to: {idn}")
            except pyvisa.Error as e:
                print(f"Could not query IDN. Error: {e}")

    def close(self) -> None:
        with self._lock:
            print(f"Disconnecting from {self.session.resource_name}")
            self.session.close()

    def write(self, cmd: str) -> None:
        with self._lock:
            self.session.write(cmd)  # type: ignore

    def query(self, cmd: str) -> str:
        with self._lock:
            return self.session.query(cmd).strip()  # type: ignore

    # ----- abstract methods -----

    @abstractmethod
    def _setup(
        self,
        cfg: T_DeviceInfo,
        *,
        progress: bool = True,
        stop_event: threading.Event | None = None,
    ) -> None: ...

    def setup(
        self,
        cfg: T_DeviceInfo,
        *,
        progress: bool = True,
        stop_event: threading.Event | None = None,
    ) -> None:
        """Setup the device with the given configuration.

        Fail-fast: if another thread is already operating this device (i.e. holds
        ``self._lock``), raises ``DeviceBusyError`` immediately instead of queuing.
        RLock semantics guarantee that the owning thread can still call ``setup()``
        from within its own operation sequence (reentrant), so nested calls are safe.
        """

        if cfg.address != self.address:
            raise RuntimeError(
                f"Trying to setup device at address {self.address} with cfg for address {cfg.address}"
            )

        if not isinstance(cfg, self.info_model):
            raise RuntimeError(
                f"Trying to setup device of type {self.__class__.__name__} with cfg of type {type(cfg)}"
            )

        # Non-blocking acquire: fails immediately if another thread holds the lock.
        # RLock allows reentry by the same thread, so nested setup() within an
        # already-locked operation sequence is not affected.
        acquired = self._lock.acquire(blocking=False)
        if not acquired:
            raise DeviceBusyError(
                f"{self.__class__.__name__} at {self.address!r}: another thread is "
                "operating this device"
            )
        try:
            self._setup(cfg, progress=progress, stop_event=stop_event)
        finally:
            self._lock.release()

    def is_busy(self) -> bool:
        """Return True when another thread is currently operating this device.

        This is a TOCTOU snapshot suitable only for display or diagnostic use.
        Do NOT use the result to gate a ``setup()`` call — the device state can
        change between ``is_busy()`` and ``setup()``.  Control flow must rely on
        catching ``DeviceBusyError`` from ``setup()`` instead.

        Note: a thread that already holds the lock (i.e. is the current operator)
        will see False here, because RLock reentry by the owner succeeds.
        """
        acquired = self._lock.acquire(blocking=False)
        if acquired:
            self._lock.release()
            return False
        return True

    @abstractmethod
    def get_info(self) -> T_DeviceInfo:
        """
        Get the current configuration of the device.
        """
