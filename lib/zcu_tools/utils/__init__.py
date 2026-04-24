from __future__ import annotations

from typing_extensions import Any, Literal, Mapping, MutableMapping

from . import datasaver, fitting, math, process, tomography


def deepupdate(
    d: MutableMapping[str, Any],
    u: Mapping[str, Any],
    behavior: Literal["error", "force", "ignore"] = "error",
) -> None:
    """
    深度更新字典 `d`，將字典 `u` 的內容合併進去。

    Args:
        d (dict[str, Any]): 目標字典，將被更新。
        u (dict[str, Any]): 更新內容的來源字典。
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

    def conflict_handler(
        d: MutableMapping[str, Any], u: Mapping[str, Any], k: Any
    ) -> None:
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


def format_obj(obj: Any) -> Any:
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        obj = obj.to_dict()  # work for pydantic model
    if hasattr(obj, "model_dump") and callable(obj.model_dump):
        obj = obj.model_dump(mode="python")  # work for pydantic model
    if hasattr(obj, "tolist") and callable(obj.tolist):
        obj = obj.tolist()  # work for numpy array
    if hasattr(obj, "item") and callable(obj.item):
        obj = obj.item()  # work for numpy scalar

    if isinstance(obj, dict):
        obj = {k: format_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        obj = [format_obj(v) for v in obj]

    return obj


__all__ = [
    # modules
    "fitting",
    "math",
    "process",
    "tomography",
    # utils
    "deepupdate",
    # datasaver
    "datasaver",
    "format_obj",
]
