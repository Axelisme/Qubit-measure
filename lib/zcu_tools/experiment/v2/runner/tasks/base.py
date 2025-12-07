from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy

from typing_extensions import Any, Callable, Generic, MutableMapping, Optional, TypeVar

from zcu_tools.device import GlobalDeviceManager
from zcu_tools.utils.debug import print_traceback
from zcu_tools.utils.func_tools import min_interval

from ..context import AddrType, ResultType, TaskConfig, TaskContext

T_KeyType = TypeVar("T_KeyType", bound=AddrType)

T_ResultType = TypeVar("T_ResultType", bound=ResultType)
T_TaskConfigType = TypeVar("T_TaskConfigType", bound=TaskConfig)
T_TaskContextType = TypeVar("T_TaskContextType", bound=TaskContext)


class AbsTask(ABC, Generic[T_ResultType, T_TaskContextType]):
    def init(self, ctx: T_TaskContextType, dynamic_pbar: bool = False) -> None:
        """Initialize the task with the current context. If dynamic_pbar is True, the pbar will only show up in the run() method."""

    @abstractmethod
    def run(self, ctx: T_TaskContextType) -> None:
        """Run the task with the current context."""

    def cleanup(self) -> None: ...

    @abstractmethod
    def get_default_result(self) -> T_ResultType: ...


def run_task(
    task: AbsTask[T_ResultType, T_TaskContextType[T_ResultType, T_TaskConfigType]],
    init_cfg: T_TaskConfigType,
    env_dict: Optional[MutableMapping[str, Any]] = None,
    update_hook: Optional[
        Callable[[T_TaskContextType[T_ResultType, T_TaskConfigType]], Any]
    ] = None,
    update_interval: Optional[float] = 0.1,
) -> T_ResultType:
    cfg = deepcopy(init_cfg)
    init_result = task.get_default_result()

    if env_dict is None:
        env_dict = dict()

    update_hook = min_interval(update_hook, update_interval)

    # initialize devices with progress bar
    GlobalDeviceManager.setup_devices(cfg["dev"], progress=True)

    ctx = TaskContext(cfg, init_result, update_hook, env_dict)

    try:
        task.init(ctx, dynamic_pbar=False)
        task.run(ctx)
        task.cleanup()
    except KeyboardInterrupt:
        print("Received KeyboardInterrupt, early stopping the program")
    except Exception:
        print("Error during measurement:")
        print_traceback()

    return ctx.data
