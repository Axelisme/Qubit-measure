import sys
import time
import traceback
from copy import deepcopy
from typing import Any, Callable, Dict, Literal, Optional, Union


def deepupdate(
    d: Dict[str, Any],
    u: Dict[str, Any],
    behavior: Literal["error", "force", "ignore"] = "error",
):
    """
    深度更新字典 `d`，將字典 `u` 的內容合併進去。

    Args:
        d (Dict[str, Any]): 目標字典，將被更新。
        u (Dict[str, Any]): 更新內容的來源字典。
        behavior (Literal["error", "force", "ignore"], optional):
            定義當鍵衝突時的行為：
            - "error": 拋出 KeyError。
            - "force": 覆蓋目標字典中的值。
            - "ignore": 保持目標字典中的值不變。
            預設為 "error"。

    Raises:
        KeyError: 當 `behavior` 為 "error" 且鍵衝突時拋出。

    Returns:
        None: 此函數直接修改輸入的字典 `d`。
    """

    def conflict_handler(d: Dict[str, Any], u: Dict[str, Any], k: Any):
        if behavior == "error":
            raise KeyError(f"Key {k} already exists in {d}.")
        elif behavior == "force":
            d[k] = u[k]

    for k, v in u.items():
        if k not in d:
            d[k] = v
        elif isinstance(v, dict) and isinstance(d[k], dict):
            deepupdate(d[k], v, behavior)
        else:
            conflict_handler(d, u, k)


# type cast all numpy types to python types
def numpy2number(obj: Any) -> Any:
    """
    將所有 numpy 資料型別轉換為對應的 Python 資料型別。

    Args:
        obj (Any): 任意物件，可能包含 numpy 型別。

    Returns:
        Any: 將 numpy 型別轉換為 Python 型別後的物件。
    """
    if hasattr(obj, "tolist"):
        obj = obj.tolist()
    if isinstance(obj, dict):
        return {k: numpy2number(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [numpy2number(v) for v in obj]
    if hasattr(obj, "item"):
        return obj.item()
    return obj


def make_sweep(
    start: Union[int, float],
    stop: Optional[Union[int, float]] = None,
    expts: Optional[int] = None,
    step: Optional[Union[int, float]] = None,
    force_int: bool = False,
) -> dict:
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
        expts = int((stop - start) / step + 0.99)  # pyright: ignore[reportOperatorIssue]
    elif step is None:
        assert stop is not None, err_str
        assert expts is not None, err_str
        step = (stop - start) / expts  # pyright: ignore[reportOperatorIssue]

    if force_int:
        start = int(start)  # pyright: ignore
        step = int(step)  # pyright: ignore
        expts = int(expts)

    stop = start + step * expts  # pyright: ignore[reportOperatorIssue]

    assert expts > 0, f"expts must be greater than 0, but got {expts}"
    assert step != 0, f"step must not be zero, but got {step}"

    return {"start": start, "stop": stop, "expts": expts, "step": step}


def get_ip_address(iface):
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


def print_traceback():
    """
    印出當前的異常追蹤訊息。如果異常包含 `_pyroTraceback`，則印出該追蹤訊息。
    """
    err_msg = sys.exc_info()[1]
    if hasattr(err_msg, "_pyroTraceback"):
        print("".join(err_msg._pyroTraceback))
    else:
        print(traceback.format_exc())


class AsyncFunc:
    """
    一個用於執行非同步函數的工具類別，支援最小間隔時間與索引參數。

    Attributes:
        func (Optional[Callable]): 要執行的函數。
        min_interval (float): 兩次執行之間的最小間隔時間。
        include_idx (bool): 是否在執行函數時包含索引參數。
    """

    def __init__(
        self,
        func: Optional[Callable],
        min_interval: float = 0.1,
        include_idx: bool = True,
    ):
        """
        初始化 AsyncFunc 類別。

        Args:
            func (Optional[Callable]): 要執行的函數。
            min_interval (float, optional): 兩次執行之間的最小間隔時間。預設為 0.1 秒。
            include_idx (bool, optional): 是否在執行函數時包含索引參數。預設為 True。

        Raises:
            ValueError: 當 `min_interval` 小於等於 0 時拋出。
        """
        self.func = func
        self.min_interval = min_interval
        self.include_idx = include_idx

        if min_interval <= 0:
            raise ValueError("min_interval must be greater than 0")

    def __enter__(self):
        """
        啟動非同步執行環境，並初始化相關資源。

        Returns:
            AsyncFunc: 返回自身以供使用。
        """
        if self.func is None:
            return None  # do nothing

        import threading

        self.lock = threading.Lock()

        # these variables are protected by lock
        self.acquiring = True
        self.have_new_job = threading.Event()
        self.last_ir = -1  # initial to -1 to accept the first job
        self.last_job = None

        # start worker thread
        self.worker_t = threading.Thread(target=self.work_loop, daemon=True)
        self.worker_t.start()  # start worker thread

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        結束非同步執行環境，釋放相關資源。
        """
        if self.func is None:
            return  # do nothing

        self.last_job = None
        with self.lock:
            self.acquiring = False
            self.have_new_job.set()  # notify worker thread to exit
        self.worker_t.join()

    def work_loop(self):
        """
        工作執行緒的主迴圈，負責執行非同步任務。
        """
        assert self.func is not None, "This method should not be called if func is None"
        prev_start = time.time() - 2 * self.min_interval
        while True:
            self.have_new_job.wait()  # wait for new job

            # this don't need to be protected by lock
            # because it is only be set before event is set
            if not self.acquiring:
                break  # if not acquiring, exit

            # check if min_interval is satisfied
            if time.time() - prev_start < self.min_interval:
                time.sleep(self.min_interval / 10)
                continue

            with self.lock:  # get job
                job, self.last_job = self.last_job, None
                self.have_new_job.clear()  # clear flag

            # do not raise exception in this thread
            try:
                assert job is not None, "Job should not be None"
                ir, args, kwargs = job
                if self.include_idx:
                    self.func(ir, *args, **kwargs)
                else:
                    self.func(*args, **kwargs)
            except Exception:
                print("Error in callback:")
                print_traceback()
            finally:
                prev_start = time.time()

    def __call__(self, ir: int, *args, **kwargs):
        """
        將任務加入執行佇列，並確保僅保留最新的任務。

        Args:
            ir (int): 任務的索引。
            *args: 傳遞給函數的參數。
            **kwargs: 傳遞給函數的關鍵字參數。
        """
        # this method may be called concurrently, so we need to protect it
        # also, make it executed in worker thread, to avoid blocking main thread
        with self.lock:
            # only keep the latest job
            if ir > self.last_ir and self.acquiring:
                self.last_ir = ir
                self.last_job = deepcopy((ir, args, kwargs))
                self.have_new_job.set()  # notify worker thread
