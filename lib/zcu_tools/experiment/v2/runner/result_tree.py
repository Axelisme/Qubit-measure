from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from numbers import Number
from typing import Any, Generic

import numpy as np
from numpy.typing import DTypeLike, NDArray
from typing_extensions import TypeVar

from zcu_tools.experiment.v2.utils import Result, merge_result_list
from zcu_tools.utils.func_tools import min_interval

from .schedule import SignalBuffer

T_Env = TypeVar("T_Env", default=dict[str, Any])
T_Result = TypeVar("T_Result", bound=Result, default=Result)


@dataclass(frozen=True, slots=True)
class ResultUpdateEvent(Generic[T_Env, T_Result]):
    measurement_name: str
    outer_index: int | None
    outer_value: Any
    env: T_Env
    node: ResultNode[T_Env]
    result: T_Result
    flush: bool


class _Subscription(Generic[T_Env]):
    def __init__(
        self,
        callback: Callable[[ResultUpdateEvent[T_Env, Any]], None],
        *,
        update_interval: float | None,
    ) -> None:
        self.callback = callback
        self.throttled_callback = min_interval(callback, update_interval)

    def emit(self, event: ResultUpdateEvent[T_Env, Any]) -> None:
        if event.flush or self.throttled_callback is None:
            self.callback(event)
            return
        self.throttled_callback(event)


class ResultTree(Generic[T_Env]):
    """Executor-owned structured result tree with per-measurement updates."""

    def __init__(
        self,
        data: list[dict[str, Result]],
        *,
        outer_values: Sequence[Any] | NDArray[Any] | None = None,
        update_interval: float | None = 0.1,
    ) -> None:
        self._data = data
        self._outer_values = outer_values
        self._update_interval = update_interval
        self._subscriptions: dict[str, list[_Subscription[T_Env]]] = {}
        self._result_cache: dict[str, Result] = {}

    @property
    def data(self) -> list[dict[str, Result]]:
        return self._data

    def at(self, index: int) -> ResultNode[T_Env]:
        return ResultNode(self, (index,))

    def measurement_node(self, name: str) -> ResultNode[T_Env]:
        return ResultNode(self, (name,), measurement_name=name)

    def subscribe(
        self,
        measurement_name: str,
        callback: Callable[[ResultUpdateEvent[T_Env, Any]], None],
    ) -> None:
        self._subscriptions.setdefault(measurement_name, []).append(
            _Subscription(callback, update_interval=self._update_interval)
        )

    def measurement_result(self, measurement_name: str) -> Result:
        if measurement_name not in self._result_cache:
            task_rows = [row[measurement_name] for row in self._data]
            self._result_cache[measurement_name] = merge_result_list(task_rows)
        return self._result_cache[measurement_name]

    def trigger_update(
        self,
        step: Any | None = None,
        *,
        flush: bool = False,
    ) -> None:
        if step is None:
            return

        names = self._measurement_names_for_path(step.path)
        for name in names:
            self._result_cache.pop(name, None)
            event = ResultUpdateEvent(
                measurement_name=name,
                outer_index=self._outer_index_for_path(step.path),
                outer_value=self._outer_value_for_path(step.path),
                env=step.env,
                node=self.measurement_node(name),
                result=self.measurement_result(name),
                flush=flush,
            )
            for subscription in self._subscriptions.get(name, ()):
                subscription.emit(event)

    def get_path(self, path: tuple[Hashable, ...]) -> Any:
        if self._is_measurement_view_path(path):
            measurement_name = path[0]
            if not isinstance(measurement_name, str):
                raise ValueError(
                    f"Expected measurement name, got {type(measurement_name)}"
                )
            return self.measurement_result(measurement_name)
        return _get_path(self._data, path)

    def set_path(self, path: tuple[Hashable, ...], value: Any) -> None:
        target = self.get_path(path)
        _set_target(target, value)
        measurement_name = self._measurement_name_for_path(path)
        if measurement_name is not None:
            self._result_cache.pop(measurement_name, None)

    def _measurement_names_for_path(self, path: tuple[Hashable, ...]) -> list[str]:
        measurement_name = self._measurement_name_for_path(path)
        if measurement_name is not None:
            return [measurement_name]
        return list(self._subscriptions)

    def _measurement_name_for_path(self, path: tuple[Hashable, ...]) -> str | None:
        if self._is_measurement_view_path(path):
            measurement_name = path[0]
            if not isinstance(measurement_name, str):
                raise ValueError(
                    f"Expected measurement name, got {type(measurement_name)}"
                )
            return measurement_name
        if len(path) >= 2 and isinstance(path[1], str):
            return path[1]
        return None

    def _outer_index_for_path(self, path: tuple[Hashable, ...]) -> int | None:
        if path and isinstance(path[0], int):
            return path[0]
        return None

    def _outer_value_for_path(self, path: tuple[Hashable, ...]) -> Any:
        outer_index = self._outer_index_for_path(path)
        if outer_index is None or self._outer_values is None:
            return None
        return self._outer_values[outer_index]

    def _is_measurement_view_path(self, path: tuple[Hashable, ...]) -> bool:
        return len(path) == 1 and isinstance(path[0], str)


class ResultNode(Generic[T_Env]):
    def __init__(
        self,
        tree: ResultTree[T_Env],
        path: tuple[Hashable, ...],
        *,
        measurement_name: str | None = None,
    ) -> None:
        self._tree = tree
        self.path = path
        self.measurement_name = measurement_name

    @property
    def data(self) -> Any:
        return self._tree.get_path(self.path)

    def child(self, addr: Hashable) -> ResultNode[T_Env]:
        if self.measurement_name is not None:
            raise ValueError("measurement subscription nodes do not have data children")
        return ResultNode(self._tree, self.path + (addr,))

    def subscribe(
        self,
        callback: Callable[[ResultUpdateEvent[T_Env, Any]], None],
    ) -> None:
        if self.measurement_name is None:
            raise ValueError("subscriptions require a measurement node")
        self._tree.subscribe(self.measurement_name, callback)

    def set(self, value: Any, *, flush: bool = False) -> None:
        if self.measurement_name is not None:
            raise ValueError("measurement subscription nodes are read-only")
        self._tree.set_path(self.path, value)
        self._tree.trigger_update(_ResultNodeStep(self.path), flush=flush)

    def buffer(
        self,
        shape: int | Sequence[int],
        *,
        dtype: DTypeLike = np.complex128,
    ) -> SignalBuffer:
        target = self.data
        if not isinstance(target, np.ndarray):
            raise ValueError(
                "ResultNode.buffer target must point to an NDArray in result data, "
                f"got {type(target)} at path {self.path}"
            )
        buffer = SignalBuffer(
            shape,
            dtype=dtype,
            on_update=lambda data: self.set(data),
            update_interval=None,
        )
        if target.shape != buffer.array.shape:
            raise ValueError(
                "ResultNode.buffer shape must match result data target shape; "
                f"target shape={target.shape}, buffer shape={buffer.array.shape}"
            )
        return buffer


class _ResultNodeStep:
    def __init__(self, path: tuple[Hashable, ...]) -> None:
        self.path = path
        self.env: Any = None


def _get_path(root: Any, path: tuple[Hashable, ...]) -> Any:
    target = root
    for depth, seg in enumerate(path):
        if isinstance(target, Mapping):
            target = target[seg]
        elif isinstance(target, list):
            if not isinstance(seg, int):
                raise ValueError(f"Expected int index for list, got {type(seg)}")
            target = target[seg]
        elif isinstance(target, np.ndarray):
            return _writable_view(target, path[depth:])
        else:
            raise ValueError(f"Expected Mapping, list, or NDArray, got {type(target)}")
    return target


def _set_target(target: Any, value: Any) -> None:
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


def _writable_view(array: NDArray[Any], path: tuple[Hashable, ...]) -> NDArray[Any]:
    if not path:
        return array
    index_parts: list[int] = []
    for part in path:
        if not isinstance(part, int):
            raise ValueError(
                f"NDArray result paths must be integer-indexed; got path suffix {path}"
            )
        index_parts.append(part)
    index = tuple(index_parts)
    return array[index]
