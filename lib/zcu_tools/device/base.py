from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, TypedDict

from typing_extensions import NotRequired

if TYPE_CHECKING:
    from pyvisa import ResourceManager


class DeviceInfo(TypedDict):
    type: str
    address: str

    label: NotRequired[str]


class BaseDevice(ABC):
    """
    Base class for all devices.
    """

    def __init__(self, address: str, rm: ResourceManager) -> None:
        self.address = address

        import pyvisa as visa

        try:
            self.session = rm.open_resource(address)
            self.session.read_termination = "\n"
            self.session.write_termination = "\n"
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
            idn = self.session.query("*IDN?")
            print(f"Connected to: {idn.strip()}")
        except pyvisa.Error as e:
            print(f"Could not query IDN. Error: {e}")

    def close(self) -> None:
        print(f"Disconnecting from {self.session.resource_name}")
        self.session.close()

    def write(self, cmd: str) -> None:
        self.session.write(cmd)

    def query(self, cmd: str) -> str:
        return self.session.query(cmd).strip()

    # ----- abstract methods -----

    @abstractmethod
    def _setup(self, cfg: Dict[str, Any], *, progress: bool = True) -> None:
        pass

    def setup(self, cfg: Dict[str, Any], *, progress: bool = True) -> None:
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
    def get_info(self) -> DeviceInfo:
        """
        Get the current configuration of the device.
        """
        pass
