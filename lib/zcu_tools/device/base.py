import sys
from abc import ABC, abstractmethod
from typing import Any, TypedDict, Dict

try:
    from pyvisa import ResourceManager
except ImportError:
    # sometimes pyvisa is not installed, e.g. in pure analysis use cases
    # we just define a dummy ResourceManager type
    ResourceManager = object


class DeviceInfo(TypedDict):
    type: str
    VISAaddress: str


class BaseDevice(ABC):
    """
    Base class for all devices.
    """

    def __init__(self, VISAaddress: str, rm: ResourceManager) -> None:
        self.VISAaddress = VISAaddress

        import pyvisa as visa

        try:
            self.session = rm.open_resource(VISAaddress)
        except visa.Error:
            sys.stderr.write("Couldn't connect to '%s', exiting now..." % VISAaddress)

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

        if cfg["VISAaddress"] != self.VISAaddress:
            raise RuntimeError(
                f"Trying to setup device at address {self.VISAaddress} with cfg for address {cfg['VISAaddress']}"
            )

        # private method to setup
        self._setup(cfg, progress=progress)

    @abstractmethod
    def get_info(self) -> DeviceInfo:
        """
        Get the current configuration of the device.
        """
        pass
