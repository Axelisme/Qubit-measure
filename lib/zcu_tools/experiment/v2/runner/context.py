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
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
    TypeVar,
    Union,
    overload,
)

from zcu_tools.device import DeviceInfo

AddrType = Union[str, int, Tuple[str, str]]
T_KeyType = TypeVar("T_KeyType", bound=AddrType)

ResultType = Union[Mapping[T_KeyType, "ResultType"], Sequence["ResultType"], NDArray]
T_ResultType = TypeVar("T_ResultType", bound=ResultType)


class TaskConfig(TypedDict):
    dev: Mapping[str, DeviceInfo]


T_TaskConfigType = TypeVar("T_TaskConfigType", bound=TaskConfig)
T_NewTaskConfigType = TypeVar("T_NewTaskConfigType", bound=TaskConfig)


@dataclass(frozen=True)
class TaskContext(Generic[T_ResultType, T_TaskConfigType]):
    cfg: T_TaskConfigType
    data: T_ResultType
    update_hook: Optional[Callable[[TaskContext], None]] = None
    env_dict: MutableMapping[str, Any] = field(default_factory=dict)
    addr_stack: List[AddrType] = field(default_factory=list)

    @overload
    def __call__(
        self, addr: AddrType
    ) -> TaskContext[T_ResultType, T_TaskConfigType]: ...

    @overload
    def __call__(
        self, addr: AddrType, new_cfg: T_NewTaskConfigType
    ) -> TaskContext[T_ResultType, T_NewTaskConfigType]: ...

    def __call__(
        self, addr: AddrType, new_cfg: Optional[T_TaskConfigType] = None
    ) -> TaskContext:
        actual_cfg = self.cfg if new_cfg is None else new_cfg

        return TaskContext(
            deepcopy(actual_cfg),
            self.data,
            self.update_hook,
            self.env_dict,
            self.addr_stack + [addr],
        )

    def is_empty_stack(self) -> bool:
        return len(self.addr_stack) == 0

    def current_task(self) -> Optional[AddrType]:
        if self.is_empty_stack():
            return None
        return self.addr_stack[-1]

    def trigger_hook(self) -> None:
        if self.update_hook is not None:
            self.update_hook(self)

    def set_data(self, value: ResultType, addr_stack: List[T_KeyType] = []) -> None:
        target = self.get_data(addr_stack)

        if isinstance(target, dict):
            if not isinstance(value, dict):
                raise ValueError(f"Expected dict, got {type(value)}")
            target.update(value)  # not deep update, intentionally
        elif isinstance(target, list):
            if not isinstance(value, list):
                raise ValueError(f"Expected list, got {type(value)}")
            target.clear()
            target.extend(value)
        elif isinstance(target, np.ndarray):
            if not isinstance(value, (np.ndarray, Number)):
                raise ValueError(f"Expected NDArray or number, got {type(value)}")
            np.copyto(target, value)

        self.trigger_hook()

    def set_current_data(
        self, value: ResultType, append_addr: List[T_KeyType] = []
    ) -> None:
        self.set_data(value, self.addr_stack + append_addr)

    def get_data(self, addr_stack: List[T_KeyType] = []) -> ResultType:
        target = self.data

        # Navigate dict keys from address
        for seg in addr_stack:
            if isinstance(target, list):
                assert isinstance(seg, int)
                target = target[seg]
            elif isinstance(target, dict):
                target = target[seg]
            else:
                raise ValueError(f"Expected dict or list, got {type(target)}")

        return target

    def get_current_data(self, append_addr: List[T_KeyType] = []) -> ResultType:
        return self.get_data(self.addr_stack + append_addr)
