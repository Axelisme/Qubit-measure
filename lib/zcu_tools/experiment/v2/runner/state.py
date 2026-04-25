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
    Mapping,
    Optional,
    Sequence,
    TypeAlias,
    TypeVar,
    Union,
    cast,
    overload,
)

from zcu_tools.experiment.cfg_model import ExpCfgModel

Result: TypeAlias = Union[Sequence["Result"], Mapping[Any, "Result"], NDArray]

T_Cfg = TypeVar("T_Cfg", bound=ExpCfgModel)
T_ChildCfg = TypeVar("T_ChildCfg", bound=ExpCfgModel)

T_Result = TypeVar("T_Result", bound=Result)
T_RootResult = TypeVar("T_RootResult", bound=Result)
T_ChildResult = TypeVar("T_ChildResult", bound=Result)
T_MappingResult = TypeVar("T_MappingResult", bound=Mapping[Any, Any])


@dataclass
class TaskState(Generic[T_Result, T_RootResult, T_Cfg]):
    """State object passed to tasks in runner.

    - `root_data` holds the full result tree for the entire task hierarchy.
    - `path` tracks the current position inside `root_data`.
    - `cfg` is the configuration mapping visible at this level.
    - `env` is a shared mutable mapping for environment / side-channel data.
    - `on_update` is called whenever `set_value` updates the current slice.
    """

    root_data: T_RootResult
    cfg: T_Cfg
    env: dict[str, Any] = field(default_factory=dict)
    on_update: Optional[Callable[["TaskState[Any, T_RootResult, Any]"], Any]] = None
    path: tuple[Union[int, Hashable], ...] = field(default_factory=tuple)

    # ------------------------------------------------------------------
    # Navigation helpers
    # ------------------------------------------------------------------

    @overload
    def child(
        self: "TaskState[list[T_ChildResult], T_RootResult, T_Cfg]",
        addr: int,
        child_type: None = None,
    ) -> "TaskState[T_ChildResult, T_RootResult, T_Cfg]": ...

    @overload
    def child(
        self: "TaskState[T_MappingResult, T_RootResult, T_Cfg]",
        addr: Hashable,
        child_type: type[T_ChildResult],
    ) -> "TaskState[T_ChildResult, T_RootResult, T_Cfg]": ...

    def child(
        self,
        addr: Union[int, Hashable],
        child_type: Optional[type[T_ChildResult]] = None,
    ) -> TaskState[T_ChildResult, T_RootResult, T_Cfg]:
        return TaskState(
            root_data=self.root_data,
            cfg=deepcopy(self.cfg),
            env=self.env,
            on_update=self.on_update,
            path=self.path + (addr,),
        )

    @overload
    def child_with_cfg(
        self: "TaskState[list[T_ChildResult], T_RootResult, T_Cfg]",
        addr: int,
        new_cfg: T_ChildCfg,
        child_type: None = None,
    ) -> "TaskState[T_ChildResult, T_RootResult, T_ChildCfg]": ...

    @overload
    def child_with_cfg(
        self: "TaskState[T_MappingResult, T_RootResult, T_Cfg]",
        addr: Hashable,
        new_cfg: T_ChildCfg,
        child_type: type[T_ChildResult],
    ) -> "TaskState[T_ChildResult, T_RootResult, T_ChildCfg]": ...

    def child_with_cfg(
        self,
        addr: Union[int, Hashable],
        new_cfg: T_ChildCfg,
        child_type: Optional[type[T_ChildResult]] = None,
    ) -> TaskState[T_ChildResult, T_RootResult, T_ChildCfg]:
        return TaskState(
            root_data=self.root_data,
            cfg=deepcopy(new_cfg),
            env=self.env,
            on_update=self.on_update,
            path=self.path + (addr,),
        )

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def _get_target(self) -> Result:
        target = self.root_data
        for seg in self.path:
            if isinstance(target, Mapping):
                target = target[seg]  # type: ignore[index]
            elif isinstance(target, list):
                if not isinstance(seg, int):
                    raise ValueError(f"Expected int index for list, got {type(seg)}")
                target = target[seg]
            else:
                raise ValueError(f"Expected Mapping or list, got {type(target)}")
        return target

    @property
    def value(self) -> T_Result:
        result = self._get_target()
        return cast(T_Result, result)

    def set_value(self, value: T_Result) -> None:
        """Update the current slice in-place and trigger the update hook.

        The update rules mirror the original TaskState.set_value():
        - dict: update keys
        - list: replace contents
        - ndarray: copy values (or broadcast from scalar)
        """
        target = self._get_target()

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

        self._trigger_update_hook()

    def _trigger_update_hook(self) -> None:
        if self.on_update is not None:
            # Expose a read-only view of the root result to the callback.
            snapshot = TaskState[Result, T_RootResult, T_Cfg](
                root_data=self.root_data, cfg=self.cfg, env=self.env, path=self.path
            )
            self.on_update(snapshot)
