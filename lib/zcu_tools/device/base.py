import sys
from abc import ABC, abstractmethod
from typing import Any, Dict

import pyvisa as visa


class BaseDevice(ABC):
    """
    Base class for all devices.
    """

    def __init__(self, VISAaddress: str, rm: visa.ResourceManager) -> None:
        self.VISAaddress = VISAaddress

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
