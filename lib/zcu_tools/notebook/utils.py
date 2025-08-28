import json
from copy import deepcopy
from typing import Dict, Optional, Union


def make_sweep(
    start: Union[int, float],
    stop: Optional[Union[int, float]] = None,
    expts: Optional[int] = None,
    step: Optional[Union[int, float]] = None,
    force_int: bool = False,
) -> Dict[str, Union[int, float]]:
    """
    建立一個掃描參數的字典，包含起始值、結束值、步長與實驗次數。

    Args:
        start (Union[int, float]): 掃描的起始值。
        stop (Optional[Union[int, float]], optional): 掃描的結束值。
        expts (Optional[int], optional): 掃描的實驗次數。
        step (Optional[Union[int, float]], optional): 掃描的步長。
        force_int (bool, optional): 是否將所有值強制轉為整數。預設為 False。

    Raises:
        AssertionError: 當參數不足以定義掃描時拋出。
        AssertionError: 當 `expts` 小於等於 0 或 `step` 為 0 時拋出。

    Returns:
        dict: 包含掃描參數的字典，鍵為 "start", "stop", "expts", "step"。
    """
    err_str = "Not enough information to define a sweep."
    if expts is None:
        assert stop is not None, err_str
        assert step is not None, err_str
        expts = int((stop - start) / step + 0.99)
    elif step is None:
        assert stop is not None, err_str
        assert expts is not None, err_str
        step = (stop - start) / expts

    if force_int:
        start = int(start)
        step = int(step)
        expts = int(expts)

    stop = start + step * expts

    assert expts > 0, f"expts must be greater than 0, but got {expts}"
    assert step != 0, f"step must not be zero, but got {step}"

    return {"start": start, "stop": stop, "expts": expts, "step": step}


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
                fcntl.ioctl(
                    s.fileno(),
                    0x8915,  # SIOCGIFADDR
                    struct.pack("256s", bytes(iface[:15], "utf-8")),
                )[20:24]
            )
        except OSError:
            raise OSError(f"Interface {iface} not found or has no IPv4 address.")


def make_comment(cfg: dict, comment: str = "") -> str:
    """
    Generate a formatted comment string from a configuration dictionary.

    Args:
        cfg (dict): Configuration dictionary to be converted to a string.
        prepend (str, optional): Additional string to prepend to the comment. Defaults to "".

    Returns:
        str: A formatted comment string.
    """
    # pretty convert cfg to string
    cfg = deepcopy(cfg)
    cfg["comment"] = comment

    return json.dumps(cfg, indent=2)
