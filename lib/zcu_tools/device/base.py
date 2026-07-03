from __future__ import annotations

import functools
import logging
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    Literal,
    Protocol,
    Self,
    TypeVar,
    cast,
)

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


class _DeviceSession(Protocol):
    resource_name: str
    read_termination: str
    write_termination: str

    def write(self, cmd: str) -> object: ...

    def query(self, cmd: str) -> str: ...

    def close(self) -> None: ...


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

    info_model: ClassVar[type[BaseDeviceInfo]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        if "info_model" not in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} must declare a concrete info_model class attribute"
            )

        info_model = cls.__dict__["info_model"]
        if not isinstance(info_model, type) or not issubclass(
            info_model, BaseDeviceInfo
        ):
            raise TypeError(
                f"{cls.__name__}.info_model must be a BaseDeviceInfo subclass"
            )

        if info_model is BaseDeviceInfo:
            raise TypeError(
                f"{cls.__name__}.info_model must be a concrete BaseDeviceInfo subclass"
            )

    def __init__(self, address: str, rm: ResourceManager | None) -> None:
        self.address = address
        self._init_locks()

        self.session: _DeviceSession | None = self._open_session(rm)
        if self.session is not None:
            self.session.read_termination = "\n"
            self.session.write_termination = "\n"
            self.connect_message()

    # ----- helper methods -----

    def _init_locks(self) -> None:
        # op_lock guards logical mutating operations and must be reentrant because
        # setup() calls public setters. io_lock guards only one session transaction.
        self._op_lock = threading.RLock()
        self._io_lock = threading.Lock()
        self._op_depth = 0
        self._logger = logging.getLogger(type(self).__module__)

    def _open_session(self, rm: ResourceManager | None) -> _DeviceSession | None:
        if rm is None:
            raise TypeError(
                f"{self.__class__.__name__} requires a VISA ResourceManager"
            )

        import pyvisa as visa

        try:
            return cast(_DeviceSession, rm.open_resource(self.address))
        except visa.Error:
            self._logger.exception("could not connect to %s", self.address)
            raise

    def _require_session(self) -> _DeviceSession:
        if self.session is None:
            raise RuntimeError(
                f"{self.__class__.__name__} at {self.address!r} has no VISA session"
            )
        return self.session

    def connect_message(self) -> None:
        """Queries and logs the IDN to confirm connection."""
        import pyvisa

        self._require_session()
        try:
            idn = self.query("*IDN?")
            self._logger.info("connected to device: %s", idn)
        except pyvisa.Error:
            self._logger.warning("could not query IDN", exc_info=True)

    @device_operation
    def close(self) -> None:
        session = self._require_session()
        with self._io_lock:
            self._logger.info("disconnecting from %s", session.resource_name)
            session.close()

    def write(self, cmd: str) -> None:
        session = self._require_session()
        with self._io_lock:
            session.write(cmd)

    def query(self, cmd: str) -> str:
        session = self._require_session()
        with self._io_lock:
            return session.query(cmd).strip()

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
