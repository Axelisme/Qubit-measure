from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from numbers import Number

import numpy as np
from numpy.typing import NDArray
from typing_extensions import (
    Any,
    Callable,
    Generic,
    Hashable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    TypeAlias,
    TypeVar,
    Union,
    cast,
    overload,
)

Result: TypeAlias = Union[Sequence["Result"], Mapping[Any, "Result"], NDArray]


T_Result = TypeVar("T_Result", bound=Result)
T_RootResult = TypeVar("T_RootResult", bound=Result)
T_ChildResult = TypeVar("T_ChildResult", bound=Result)


@dataclass(frozen=True)
class TaskContext(Generic[T_Result]):
    data: T_Result
    env_dict: MutableMapping[str, Any] = field(default_factory=dict)

    def view(
        self,
        cfg: MutableMapping[str, Any],
        update_hook: Optional[
            Callable[[TaskContextView[Result, T_Result]], Any]
        ] = None,
    ) -> TaskContextView[T_Result, T_Result]:
        return TaskContextView(self, cfg, update_hook)

    def set_data(
        self, value: Result, addr_stack: Optional[List[Union[int, Hashable]]] = None
    ) -> None:
        if addr_stack is None:
            addr_stack = []

        target = self.get_data(addr_stack)
        if isinstance(target, dict):
            if not isinstance(value, dict):
                raise ValueError(f"Expected dict, got {type(value)}")
            target.update(value)
        elif isinstance(target, list):
            if not isinstance(value, list):
                raise ValueError(f"Expected list, got {type(value)}")
            target.clear()
            target.extend(value)
        elif isinstance(target, np.ndarray):
            if not isinstance(value, (np.ndarray, Number)):
                raise ValueError(f"Expected NDArray or number, got {type(value)}")
            np.copyto(dst=target, src=value)

    def get_data(
        self, addr_stack: Optional[List[Union[int, Hashable]]] = None
    ) -> Result:
        if addr_stack is None:
            addr_stack = []

        target = self.data
        for seg in addr_stack:
            if isinstance(target, dict):
                target = target[seg]
            elif isinstance(target, list):
                if not isinstance(seg, int):
                    raise ValueError(f"Expected int, got {type(seg)}")
                target = target[seg]
            else:
                raise ValueError(f"Expected dict or list, got {type(target)}")
        return target


@dataclass(frozen=True)
class TaskContextView(Generic[T_Result, T_RootResult]):
    context: TaskContext[T_RootResult]
    cfg: MutableMapping[str, Any]
    update_hook: Optional[Callable[[TaskContextView[Result, T_RootResult]], Any]] = None

    _addr_stack: List[Union[int, Hashable]] = field(default_factory=list)
    _root: Optional[TaskContextView[T_RootResult, T_RootResult]] = field(default=None)

    @overload
    def __call__(
        self: TaskContextView[Sequence[T_ChildResult], T_RootResult],
        addr: int,
        new_cfg: Optional[MutableMapping[str, Any]] = None,
    ) -> TaskContextView[T_ChildResult, T_RootResult]: ...

    @overload
    def __call__(
        self: TaskContextView[Mapping[Any, T_ChildResult], T_RootResult],
        addr: Hashable,
        new_cfg: Optional[MutableMapping[str, Any]] = None,
    ) -> TaskContextView[T_ChildResult, T_RootResult]: ...

    def __call__(
        self,
        addr: Union[int, Hashable],
        new_cfg: Optional[MutableMapping[str, Any]] = None,
    ) -> TaskContextView:
        actual_cfg = deepcopy(self.cfg) if new_cfg is None else new_cfg

        root = self._root
        if root is None:
            root = self
        root = cast(TaskContextView[T_RootResult, T_RootResult], root)

        return TaskContextView(
            self.context,
            actual_cfg,
            update_hook=self.update_hook,
            _addr_stack=self._addr_stack + [addr],
            _root=root,
        )

    @property
    def root(self) -> TaskContextView[T_RootResult, T_RootResult]:
        if self._root is None:
            return cast(TaskContextView[T_RootResult, T_RootResult], self)
        return self._root

    def trigger_hook(self) -> None:
        if self.update_hook is not None:
            self.update_hook(self)  # type: ignore[arg-type]

    def set_data(self, value: T_Result) -> None:
        self.context.set_data(value, self._addr_stack)
        self.trigger_hook()

    def get_data(self) -> T_Result:
        result = self.context.get_data(self._addr_stack)
        return cast(T_Result, result)

    @property
    def env_dict(self) -> MutableMapping[str, Any]:
        return self.context.env_dict

    @property
    def data(self) -> T_RootResult:
        return self.context.data
