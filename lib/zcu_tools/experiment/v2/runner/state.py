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
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    TypeAlias,
    TypeVar,
    Union,
    cast,
)

Result: TypeAlias = Union[Sequence["Result"], Mapping[Any, "Result"], NDArray]


T_Result = TypeVar("T_Result", bound=Result)
T_RootResult = TypeVar("T_RootResult", bound=Result)
T_ChildResult = TypeVar("T_ChildResult", bound=Result)


@dataclass
class TaskState(Generic[T_Result, T_RootResult]):
    """State object passed to tasks in runner.

    - `root_data` holds the full result tree for the entire task hierarchy.
    - `path` tracks the current position inside `root_data`.
    - `cfg` is the configuration mapping visible at this level.
    - `env` is a shared mutable mapping for environment / side-channel data.
    - `on_update` is called whenever `set_value` updates the current slice.
    """

    root_data: T_RootResult
    cfg: MutableMapping[str, Any]
    env: MutableMapping[str, Any] = field(default_factory=dict)
    on_update: Optional[Callable[["TaskState[Result, T_RootResult]"], Any]] = None
    path: Tuple[Union[int, Hashable], ...] = field(default_factory=tuple)

    # ------------------------------------------------------------------
    # Navigation helpers
    # ------------------------------------------------------------------

    def child(
        self,
        addr: Union[int, Hashable],
        new_cfg: Optional[Mapping[str, Any]] = None,
    ) -> "TaskState[T_ChildResult, T_RootResult]":
        """Return a new TaskState pointing to a child result.

        Strategy A: `cfg` is deep-copied when `new_cfg` is None, so that
        modifications in the child do not affect the parent's cfg.
        """
        actual_cfg = self.cfg if new_cfg is None else new_cfg

        return TaskState(
            root_data=self.root_data,
            cfg=deepcopy(actual_cfg),  # type: ignore
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

        self._trigger_update()

    def _trigger_update(self) -> None:
        if self.on_update is not None:
            # Expose a read-only view of the root result to the callback.
            snapshot = TaskState[Result, T_RootResult](
                root_data=self.root_data,
                cfg=self.cfg,
                env=self.env,
                path=self.path,
            )
            self.on_update(snapshot)
