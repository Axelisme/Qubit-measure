from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from copy import deepcopy

from typing_extensions import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Mapping,
    MutableMapping,
    NotRequired,
    Optional,
    Sequence,
    TypedDict,
    TypeVar,
    cast,
)

from zcu_tools.device import DeviceInfo
from zcu_tools.utils.debug import print_traceback
from zcu_tools.utils.func_tools import min_interval

from .state import Result, TaskState

if TYPE_CHECKING:
    from .repeat import RepeatOverTime, ReTryIfFail
    from .soft import Scan, T_Value

logger = logging.getLogger(__name__)

T_Result = TypeVar("T_Result", bound=Result)
T_RootResult = TypeVar("T_RootResult", bound=Result)


class TaskCfg(TypedDict, closed=False):
    dev: NotRequired[dict[str, DeviceInfo]]


class AbsTask(ABC, Generic[T_Result, T_RootResult]):
    def init(
        self,
        state: TaskState[T_Result, T_RootResult],
        dynamic_pbar: bool = False,
    ) -> None:
        """Initialize the task with the current state.

        If dynamic_pbar is True, the progress bar will only show up in the run() method.
        NOTE: This method may be called multiple times during the task execution.
        """

    @abstractmethod
    def run(self, state: TaskState[T_Result, T_RootResult]) -> None:
        """Run the task with the current state."""

    def cleanup(self) -> None: ...

    @abstractmethod
    def get_default_result(self) -> T_Result: ...

    def scan(
        self,
        name: str,
        values: Sequence[T_Value],
        before_each: Callable[
            [int, TaskState[list[T_Result], T_RootResult], T_Value], Any
        ],
    ) -> Scan[T_Result, T_RootResult]:
        """Scan a task over a sequence of values."""
        from .soft import Scan

        return Scan(name, values, before_each, task=self)

    def repeat(
        self, name: str, times: int, interval: float = 0.0
    ) -> RepeatOverTime[T_Result, T_RootResult]:
        """Repeat a task over a fixed number of times at a fixed interval."""
        from .repeat import RepeatOverTime

        return RepeatOverTime(name, times, task=self, interval=interval)

    def auto_retry(self, max_retries: int) -> ReTryIfFail[T_Result, T_RootResult]:
        """Automatically retry a task if it fails."""
        from .repeat import ReTryIfFail

        return ReTryIfFail(task=self, max_retries=max_retries)


def run_task(
    task: AbsTask[T_Result, T_Result],
    init_cfg: Mapping[str, Any],
    env_dict: Optional[MutableMapping[str, Any]] = None,
    on_update: Optional[Callable[[TaskState[Any, T_Result]], Any]] = None,
    update_interval: Optional[float] = 0.1,
) -> T_Result:
    """Run a task with a fresh TaskState.

    - Deep-copies `init_cfg` so that the task can mutate it safely.
    - Initializes the result via `task.get_default_result()`.
    - Wraps `on_update` with a min-interval throttler.
    """
    cfg = cast(MutableMapping[str, Any], deepcopy(init_cfg))
    init_result = task.get_default_result()

    if env_dict is None:
        env_dict = dict()

    on_update = min_interval(on_update, update_interval)

    state: TaskState[T_Result, T_Result] = TaskState(
        root_data=init_result,
        cfg=cfg,
        env=env_dict,
        on_update=on_update,
    )

    logger.debug(
        "run_task: task=%s, cfg_keys=%s, env_keys=%s",
        type(task).__name__, list(cfg.keys()), list(env_dict.keys()),
    )

    try:
        task.init(state, dynamic_pbar=False)
        logger.debug("run_task: init done, starting run")
        task.run(state)
        logger.debug("run_task: run done, cleanup")
    except KeyboardInterrupt:
        logger.warning("run_task: KeyboardInterrupt, early stopping")
    except Exception:
        logger.exception("run_task: error during measurement")
        print_traceback()
        raise
    finally:
        task.cleanup()

    return state.root_data
