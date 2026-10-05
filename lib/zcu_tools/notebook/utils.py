from __future__ import annotations

import gc
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from zcu_tools.device import DeviceInfo, DeviceManager
from zcu_tools.device.sgs100a import RohdeSchwarzSGS100A
from zcu_tools.device.yoko import YOKOGS200

if TYPE_CHECKING:
    try:
        # in case pyvisa is not installed, use Any as ResourceManager to pass type checking
        from pyvisa import ResourceManager
    except ImportError:
        from typing import Any as ResourceManager


def gc_collect(verbose: bool = True) -> None:
    """Clear all matplotlib figures and run garbage collection to free memory."""
    plt.close("all")
    gc_num = gc.collect()
    if verbose:
        print(f"Garbage collection done. Collected {gc_num} objects.")


def get_ip_address(iface: str) -> str:
    """
    獲取指定網路介面的 IP 位址，支援 Linux 與 Windows 系統。

    Args:
        iface (str): 網路介面的名稱。

    Returns:
        str: 該介面的 IP 位址。

    Raises:
        OSError: 當無法獲取 IP 位址時拋出。
    """
    import platform
    import socket

    if platform.system() == "Windows":
        # Windows 系統
        import psutil

        for nic, addrs in psutil.net_if_addrs().items():
            if nic == iface:
                for addr in addrs:
                    if addr.family == socket.AF_INET:
                        return addr.address
        raise OSError(f"Interface {iface} not found or has no IPv4 address.")
    else:
        # Linux 系統
        import fcntl
        import struct

        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            return socket.inet_ntoa(
                fcntl.ioctl(  # type: ignore
                    s.fileno(),
                    0x8915,  # SIOCGIFADDR
                    struct.pack("256s", bytes(iface[:15], "utf-8")),
                )[20:24]
            )
        except OSError:
            raise OSError(f"Interface {iface} not found or has no IPv4 address.")


def savefig(fig: Figure, filepath: str, close_after: bool = True, **kwargs) -> None:
    """Save a matplotlib figure, creating parent directories if necessary. close the figure after saving to free memory."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(filepath, **kwargs)
    if close_after:
        plt.close(fig)


def dump_device_info(path: str | Path, device_manager: DeviceManager) -> None:
    info_snapshot = {
        name: info.to_dict() for name, info in device_manager.get_all_info().items()
    }
    with open(str(path), "w") as f:
        json.dump(info_snapshot, f, indent=2)


def reconnect_devices(
    dev_info: Mapping[str, DeviceInfo], device_manager: DeviceManager
) -> ResourceManager:
    from pyvisa import ResourceManager

    resource_manager = ResourceManager()
    for name, info in dev_info.items():
        if info.type == "YOKOGS200":
            device = YOKOGS200(info.address, resource_manager)
        elif info.type == "RohdeSchwarzSGS100A":
            device = RohdeSchwarzSGS100A(info.address, resource_manager)
        else:
            raise ValueError(f"Not supported device type: {info.type}")
        device_manager.register_device(name, device)

    return resource_manager
