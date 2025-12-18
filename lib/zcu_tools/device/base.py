from __future__ import annotations

import sys
from abc import ABC, abstractmethod

from typing_extensions import TYPE_CHECKING, Generic, NotRequired, TypedDict, TypeVar

if TYPE_CHECKING:
    from pyvisa import ResourceManager


class DeviceInfo(TypedDict):
    type: str
    address: str

    label: NotRequired[str]


T_DeviceInfo = TypeVar("T_DeviceInfo", bound=DeviceInfo)


class BaseDevice(ABC, Generic[T_DeviceInfo]):
    """
    Base class for all devices.
    """

    def __init__(self, address: str, rm: ResourceManager) -> None:
        self.address = address

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

        try:
            idn = self.query("*IDN?")
            print(f"Connected to: {idn}")
        except pyvisa.Error as e:
            print(f"Could not query IDN. Error: {e}")

    def close(self) -> None:
        print(f"Disconnecting from {self.session.resource_name}")
        self.session.close()

    def write(self, cmd: str) -> None:
        self.session.write(cmd)  # type: ignore

    def query(self, cmd: str) -> str:
        return self.session.query(cmd).strip()  # type: ignore

    # ----- abstract methods -----

    @abstractmethod
    def _setup(self, cfg: T_DeviceInfo, *, progress: bool = True) -> None: ...

    def setup(self, cfg: T_DeviceInfo, *, progress: bool = True) -> None:
        """
        Setup the device with the given configuration.
        """
        # sanity checks
        if cfg["type"] != self.__class__.__name__:
            raise RuntimeError(
                f"Trying to setup device of type {self.__class__.__name__} with cfg of type {cfg['type']}"
            )

        if cfg["address"] != self.address:
            raise RuntimeError(
                f"Trying to setup device at address {self.address} with cfg for address {cfg['address']}"
            )

        # private method to setup
        self._setup(cfg, progress=progress)

    @abstractmethod
    def get_info(self) -> T_DeviceInfo:
        """
        Get the current configuration of the device.
        """
