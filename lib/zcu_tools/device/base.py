from __future__ import annotations

import functools
import logging
import sys
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Generic, Literal, Self, TypeVar, cast

from zcu_tools.cfg_model import ConfigBase

if TYPE_CHECKING:
    from pyvisa import ResourceManager


class DeviceBusyError(RuntimeError):
    """Raised when a device operation lock is already held by another thread.

    Mutating public operations are fail-fast: they never queue behind an
    in-progress operation. Use is_busy() to inspect status for display/diagnostic
    purposes only.
    """


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

    # ----- experiment-knob setters -----
    #
    # The experiment layer drives device configs through these knob setters
    # instead of reaching into device-specific fields. Each subclass overrides
    # only the knobs it supports; unsupported knobs fail fast via the base raise.

    def set_flux(self, value: float) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} does not support setting flux"
        )

    def set_freq(self, freq_Hz: float) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} does not support setting freq"
        )

    def set_power(self, power_dBm: float) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} does not support setting power"
        )

    def set_output(self, output: Literal["on", "off"]) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} does not support setting output"
        )


T_DeviceInfo = TypeVar("T_DeviceInfo", bound=BaseDeviceInfo)
T_Return = TypeVar("T_Return")


def device_operation(method: Callable[..., T_Return]) -> Callable[..., T_Return]:
    """Guard a public mutating device operation with op_lock and logging."""

    @functools.wraps(method)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> T_Return:
        operation = f"{self.__class__.__name__}.{method.__name__}"
        acquired = self._op_lock.acquire(blocking=False)
        if not acquired:
            message = (
                f"{self.__class__.__name__} at {self.address!r}: another thread is "
                "operating this device"
            )
            self._logger.info("device operation busy: %s", operation)
            raise DeviceBusyError(message)

        outer = self._op_depth == 0
        start: float | None = None
        if outer:
            start = time.monotonic()
            self._logger.info("device operation started: %s", operation)

        self._op_depth += 1
        try:
            result = method(self, *args, **kwargs)
        except BaseException:
            if outer and start is not None:
                elapsed = time.monotonic() - start
                self._logger.exception(
                    "device operation failed: %s elapsed=%.3fs",
                    operation,
                    elapsed,
                )
            raise
        else:
            if outer and start is not None:
                elapsed = time.monotonic() - start
                self._logger.info(
                    "device operation finished: %s elapsed=%.3fs",
                    operation,
                    elapsed,
                )
            return result
        finally:
            self._op_depth -= 1
            self._op_lock.release()

    return cast(Callable[..., T_Return], wrapper)


class BaseDevice(ABC, Generic[T_DeviceInfo]):
    """
    Base class for all devices.
    """

    info_model: type[BaseDeviceInfo] = BaseDeviceInfo

    def __init__(self, address: str, rm: ResourceManager) -> None:
        self.address = address
        self._init_locks()

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

    def _init_locks(self) -> None:
        # op_lock guards logical mutating operations and must be reentrant because
        # setup() calls public setters. io_lock guards only one session transaction.
        self._op_lock = threading.RLock()
        self._io_lock = threading.Lock()
        self._op_depth = 0
        self._logger = logging.getLogger(type(self).__module__)

    def connect_message(self) -> None:
        """Queries and prints the IDN to confirm connection."""
        import pyvisa

        try:
            idn = self.query("*IDN?")
            print(f"Connected to: {idn}")
        except pyvisa.Error as e:
            print(f"Could not query IDN. Error: {e}")

    @device_operation
    def close(self) -> None:
        with self._io_lock:
            print(f"Disconnecting from {self.session.resource_name}")
            self.session.close()

    def write(self, cmd: str) -> None:
        with self._io_lock:
            self.session.write(cmd)  # type: ignore

    def query(self, cmd: str) -> str:
        with self._io_lock:
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

    @device_operation
    def setup(
        self,
        cfg: T_DeviceInfo,
        *,
        progress: bool = True,
        stop_event: threading.Event | None = None,
    ) -> None:
        """Setup the device with the given configuration.

        Fail-fast: if another thread is already operating this device (i.e. holds
        ``self._op_lock``), raises ``DeviceBusyError`` immediately instead of
        queuing. RLock semantics guarantee that the owning thread can still call
        ``setup()`` from within its own operation sequence (reentrant), so nested
        calls are safe.
        """

        if cfg.address != self.address:
            raise RuntimeError(
                f"Trying to setup device at address {self.address} with cfg for address {cfg.address}"
            )

        if not isinstance(cfg, self.info_model):
            raise RuntimeError(
                f"Trying to setup device of type {self.__class__.__name__} with cfg of type {type(cfg)}"
            )

        self._setup(cfg, progress=progress, stop_event=stop_event)

    def is_busy(self) -> bool:
        """Return True when another thread is currently operating this device.

        This is a TOCTOU snapshot suitable only for display or diagnostic use.
        Do NOT use the result to gate a ``setup()`` call — the device state can
        change between ``is_busy()`` and ``setup()``.  Control flow must rely on
        catching ``DeviceBusyError`` from ``setup()`` instead.

        Note: a thread that already holds op_lock (i.e. is the current operator)
        will see False here, because RLock reentry by the owner succeeds.
        """
        acquired = self._op_lock.acquire(blocking=False)
        if acquired:
            self._op_lock.release()
            return False
        return True

    @abstractmethod
    def get_info(self) -> T_DeviceInfo:
        """
        Get the current configuration of the device.
        """
