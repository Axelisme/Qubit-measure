from __future__ import annotations

from collections.abc import Hashable, Mapping, MutableMapping
from numbers import Number
from typing import Any

import numpy as np
from numpy.typing import NDArray


def get_path(root: Any, path: tuple[Hashable, ...]) -> Any:
    target = root
    for depth, seg in enumerate(path):
        if isinstance(target, Mapping):
            target = target[seg]
        elif isinstance(target, list):
            if not isinstance(seg, int):
                raise ValueError(f"Expected int index for list, got {type(seg)}")
            target = target[seg]
        elif isinstance(target, np.ndarray):
            return writable_view(target, path[depth:])
        else:
            raise ValueError(f"Expected Mapping, list, or NDArray, got {type(target)}")
    return target


def set_target(target: Any, value: Any) -> None:
    if isinstance(target, MutableMapping):
        if not isinstance(value, Mapping):
            raise ValueError(f"Expected Mapping, got {type(value)}")
        target.update(value)
    elif isinstance(target, list):
        if not isinstance(value, list):
            raise ValueError(f"Expected list, got {type(value)}")
        target.clear()
        target.extend(value)
    elif isinstance(target, np.ndarray):
        if isinstance(value, np.ndarray):
            np.copyto(dst=target, src=value)
        elif isinstance(value, Number):
            np.copyto(dst=target, src=np.asarray(value))
        else:
            raise ValueError(f"Expected NDArray or number, got {type(value)}")
    else:
        raise ValueError(f"Expected Mapping, list, or NDArray, got {type(target)}")


def writable_view(array: NDArray[Any], index: tuple[Any, ...]) -> NDArray[Any]:
    if not index:
        return array

    direct = array[index]
    if isinstance(direct, np.ndarray):
        if not np.shares_memory(direct, array):
            raise ValueError("NDArray path indexing must select a writable view")
        return direct

    scalar_index: list[slice] = []
    for axis, part in enumerate(index):
        if not isinstance(part, int):
            raise ValueError("Scalar NDArray path indexing only supports integer axes")
        axis_size = array.shape[axis]
        normalized = part + axis_size if part < 0 else part
        if normalized < 0 or normalized >= axis_size:
            raise IndexError(f"index {part} is out of bounds for axis {axis}")
        scalar_index.append(slice(normalized, normalized + 1))

    return array[tuple(scalar_index)].reshape(())
