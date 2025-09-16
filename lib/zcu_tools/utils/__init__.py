from typing import Any, Dict, Literal

from . import datasaver, fitting, process, tomography
from .async_func import AsyncFunc


def deepupdate(
    d: Dict[str, Any],
    u: Dict[str, Any],
    behavior: Literal["error", "force", "ignore"] = "error",
) -> None:
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

    def conflict_handler(d: Dict[str, Any], u: Dict[str, Any], k: Any) -> None:
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


def numpy2number(obj: Any) -> Any:
    """
    將所有 numpy 資料型別轉換為對應的 Python 資料型別。

    Args:
        obj (Any): 任意物件，可能包含 numpy 型別。

    Returns:
        Any: 將 numpy 型別轉換為 Python 型別後的物件。
    """
    if hasattr(obj, "tolist") and callable(obj.tolist):
        obj = obj.tolist()
    if isinstance(obj, dict):
        return {k: numpy2number(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [numpy2number(v) for v in obj]
    if hasattr(obj, "item") and callable(obj.item):
        return obj.item()
    return obj


__all__ = [
    "deepupdate",
    "numpy2number",
    "AsyncFunc",
    "datasaver",
    "fitting",
    "process",
]
