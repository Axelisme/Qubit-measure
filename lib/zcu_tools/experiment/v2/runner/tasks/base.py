from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy

from typing_extensions import Any, Callable, Generic, MutableMapping, Optional, TypeVar

from zcu_tools.utils.debug import print_traceback
from zcu_tools.utils.func_tools import min_interval

from ..context import Result, TaskConfig, TaskContext, TaskContextView

T_Result = TypeVar("T_Result", bound=Result)
T_RootResult = TypeVar("T_RootResult", bound=Result)
T_TaskConfig = TypeVar("T_TaskConfig", bound=TaskConfig)


class AbsTask(ABC, Generic[T_Result, T_RootResult, T_TaskConfig]):
    def init(
        self,
        ctx: TaskContextView[T_Result, T_RootResult, T_TaskConfig],
        dynamic_pbar: bool = False,
    ) -> None:
        """Initialize the task with the current context. If dynamic_pbar is True, the pbar will only show up in the run() method."""

    @abstractmethod
    def run(self, ctx: TaskContextView[T_Result, T_RootResult, T_TaskConfig]) -> None:
        """Run the task with the current context."""

    def cleanup(self) -> None: ...

    @abstractmethod
    def get_default_result(self) -> T_Result: ...


def run_task(
    task: AbsTask[T_Result, T_Result, T_TaskConfig],
    init_cfg: T_TaskConfig,
    env_dict: Optional[MutableMapping[str, Any]] = None,
    update_hook: Optional[
        Callable[[TaskContextView[Result, T_Result, TaskConfig]], Any]
    ] = None,
    update_interval: Optional[float] = 0.1,
) -> T_Result:
    cfg = deepcopy(init_cfg)
    init_result = task.get_default_result()

    if env_dict is None:
        env_dict = dict()

    update_hook = min_interval(update_hook, update_interval)

    ctx = TaskContext(init_result, env_dict)
    ctx_view = ctx.view(cfg, update_hook)

    try:
        task.init(ctx_view, dynamic_pbar=False)
        task.run(ctx_view)
        task.cleanup()
    except KeyboardInterrupt:
        print("Received KeyboardInterrupt, early stopping the program")
    except Exception:
        print("Error during measurement:")
        print_traceback()

    return ctx.data
