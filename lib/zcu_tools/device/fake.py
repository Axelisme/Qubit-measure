from __future__ import annotations

import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import ClassVar, Literal, Self, override

from pydantic import Field, field_validator

from ._ramp import RampProgress, ramp_linear
from .base import BaseDevice, BaseDeviceInfo, device_operation

DEFAULT_RAMPSTEP = 0.01
RAMP_INTERVAL = 0.01  # seconds between steps (skipped in fast_mode)


class FakeDeviceInfo(BaseDeviceInfo):
    """Memory-device settings and readback in fixed, unconverted coordinates.

    Construction requires address. value_unit defaults to native, is stored in
    dumps; V and A are also supported without conversion. value is a
    strict real scalar; output controls enable state and rampstep is the native
    ramp increment.
    """

    type: Literal["FakeDevice"] = "FakeDevice"
    value_unit: Literal["native", "V", "A"] = Field(
        default="native", description="Fixed value/rampstep coordinate; no conversion."
    )
    output: Literal["on", "off"] = "off"
    value: float = 0.0
    rampstep: float = DEFAULT_RAMPSTEP

    @field_validator("value", mode="before")
    @classmethod
    def _validate_value(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                "value must be a real numeric scalar (int or float); "
                "bool and strings are not accepted"
            )
        return value


class FakeDevice(BaseDevice[FakeDeviceInfo]):
    """Memory source with optional process-wide address identity.

    An explicit address reuses the same instance, including unit, output, value,
    ramp settings and simulation bindings. First construction sets fast_mode and
    value_unit; subsequent construction does not reinitialize them. close is a
    no-op and does not release this identity. Omitted address (None) creates an
    independent anonymous instance whose info address is 'none'; it is not cached.
    The class container retains at most 1024 addressed instances to prevent
    unbounded process-lifetime growth, not as a normal operating quota. At
    capacity, new addresses raise RuntimeError; existing addresses remain usable.
    No instances are evicted. Real device drivers never use this container.
    """

    info_model = FakeDeviceInfo
    _instances: ClassVar[dict[str, FakeDevice]] = {}
    _instance_lock: ClassVar[threading.RLock] = threading.RLock()

    def __new__(
        cls,
        *,
        address: str | None = None,
        fast_mode: bool = False,
        value_unit: Literal["native", "V", "A"] = "native",
    ) -> Self:
        """Allocate anonymous memory or retrieve the class-retained address.

        Invalid address types raise TypeError; invalid initial units or an
        address owned by a different FakeDevice subclass raise ValueError.
        Capacity failure raises RuntimeError before creating another instance.
        """
        del fast_mode  # __init__ applies this only for a new instance.
        if address is not None and type(address) is not str:
            raise TypeError("address must be a string or None")
        with FakeDevice._instance_lock:
            if address is not None and address in FakeDevice._instances:
                instance = FakeDevice._instances[address]
                if not isinstance(instance, cls):
                    raise ValueError(
                        f"address {address!r} belongs to another FakeDevice class"
                    )
                return instance
            if value_unit not in ("native", "V", "A"):
                raise ValueError("value_unit must be native, V or A")
            if address is not None and len(FakeDevice._instances) >= 1024:
                raise RuntimeError(
                    "FakeDevice instance capacity 1024 reached; cannot create a new address"
                )
            instance = super().__new__(cls)
            instance._initialized = False
            if address is not None:
                FakeDevice._instances[address] = instance
            return instance

    def __init__(
        self,
        *,
        address: str | None = None,
        fast_mode: bool = False,
        value_unit: Literal["native", "V", "A"] = "native",
    ) -> None:
        """Create or retrieve the address identity described by the class contract.

        address=None means anonymous, not the explicit string 'none'. fast_mode
        skips ramp sleeps; value_unit is native, V or A without conversion.
        Only the first construction applies these settings. Invalid initial units
        raise ValueError. Addressed construction is serialized across threads.
        """
        with FakeDevice._instance_lock:
            if self._initialized:
                return
            super().__init__("none" if address is None else address, rm=None)
            self._value_unit: Literal["native", "V", "A"] = value_unit
            self.output: Literal["on", "off"] = "off"
            self.value = 0.0
            self._rampstep = DEFAULT_RAMPSTEP
            self._fast_mode = fast_mode
            self._initialized = True

    @classmethod
    def instance_addresses(cls) -> tuple[str, ...]:
        """Return retained explicit addresses without reading or changing devices."""
        with FakeDevice._instance_lock:
            return tuple(FakeDevice._instances)

    @classmethod
    @contextmanager
    def isolated_instances(cls) -> Generator[None]:
        """Use an empty address scope, then discard it and restore the prior scope.

        This process-wide lifecycle boundary does not close borrowed instances.
        Call only while no device constructors or operations are active, including
        workers. Hosts/tests must drain their work before leaving the scope.
        Nested scopes restore the enclosing identities; anonymous devices are
        unaffected. Production normally keeps one scope for the process lifetime.
        """
        with FakeDevice._instance_lock:
            previous = FakeDevice._instances
            FakeDevice._instances = {}
        try:
            yield
        finally:
            with FakeDevice._instance_lock:
                FakeDevice._instances.clear()
                FakeDevice._instances = previous

    @override
    def _open_session(self, rm: object | None) -> None:
        return None

    def get_output(self) -> Literal["on", "off"]:
        return self.output

    @device_operation
    def set_output(self, status: Literal["on", "off"]) -> None:
        self.output = status

    @device_operation
    def output_on(self) -> None:
        self.set_output("on")

    @device_operation
    def output_off(self) -> None:
        self.set_output("off")

    # ==========================================================================#

    def get_value(self) -> float:
        return self.value

    @device_operation
    def set_value(self, value: float) -> float:
        self.value = value
        return self.value

    @device_operation
    def close(self) -> None:
        """Fake devices own no external session."""

    def _set_value_smart(
        self,
        value: float,
        stop_event: threading.Event | None = None,
        *,
        progress: bool = False,
    ) -> None:
        current_value = self.get_value()

        def apply_value(target: float) -> None:
            self.value = float(target)
            if not self._fast_mode:
                time.sleep(RAMP_INTERVAL)

        ramp_linear(
            start=current_value,
            target=value,
            step=self._rampstep,
            apply_value=apply_value,
            progress=RampProgress(enabled=progress, desc="Ramp value", decimals=6),
            stop_event=stop_event,
            include_start=False,
        )

    # ==========================================================================#

    def _setup(
        self,
        cfg: FakeDeviceInfo,
        *,
        progress: bool = True,
        stop_event: threading.Event | None = None,
    ) -> None:
        if cfg.value_unit != self._value_unit:
            raise ValueError("setup value_unit differs from fixed device unit")
        self.set_output(cfg.output)
        self._rampstep = cfg.rampstep
        self._set_value_smart(cfg.value, progress=progress, stop_event=stop_event)

    def get_info(self) -> FakeDeviceInfo:
        """Return fresh memory settings with the fixed unit; no I/O or mutation."""
        return FakeDeviceInfo(
            address=self.address,
            output=self.output,
            value=self.value,
            rampstep=self._rampstep,
            value_unit=self._value_unit,
        )
