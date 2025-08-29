import sys
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    try:
        from pyvisa import ResourceManager
    except ImportError:
        ResourceManager = object


class BaseDevice(ABC):
    """
    Base class for all devices.
    """

    def __init__(self, VISAaddress: str, rm: "ResourceManager") -> None:
        self.VISAaddress = VISAaddress

        import pyvisa as visa

        try:
            self.session = rm.open_resource(VISAaddress)
        except visa.Error:
            sys.stderr.write("Couldn't connect to '%s', exiting now..." % VISAaddress)
            sys.exit()

    @abstractmethod
    def setup(self, cfg: Dict[str, Any], *, progress: bool = True) -> None:
        """
        Setup the device with the given configuration.
        """
        pass

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """
        Get the current configuration of the device.
        """
        pass
