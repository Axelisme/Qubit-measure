from __future__ import annotations

import time
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
    TypedDict,
    TypeVar,
    Union,
)

import numpy as np
from numpy import ndarray
from tqdm.auto import tqdm

from zcu_tools.device import DeviceInfo, GlobalDeviceManager
from zcu_tools.utils.debug import print_traceback
from zcu_tools.utils.func_tools import min_interval

ResultType = Union[Dict[Any, "ResultType"], List["ResultType"], ndarray]
T_ResultType = TypeVar("T_ResultType")


class TaskConfig(TypedDict):
    dev: Dict[str, DeviceInfo]


@dataclass
class TaskContext:
    cfg: TaskConfig
    data: ResultType
    update_hook: Optional[Callable[[TaskContext], None]] = None
    env_dict: Dict[str, Any] = field(default_factory=dict)

    addr_stack: List[Union[int, str]] = field(default_factory=list)

    def __call__(
        self, addr: Union[int, str], new_cfg: Optional[TaskConfig] = None
    ) -> TaskContext:
        new_cfg = self.cfg if new_cfg is None else new_cfg

        return TaskContext(
            deepcopy(new_cfg),
            self.data,
            self.update_hook,
            self.env_dict,
            self.addr_stack + [addr],
        )

    def is_empty_stack(self) -> bool:
        return len(self.addr_stack) == 0

    def current_task(self) -> Any:
        if self.is_empty_stack():
            return None
        return self.addr_stack[-1]

    def set_data(
        self, value: ResultType, addr_stack: List[Union[int, str]] = []
    ) -> None:
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
        elif isinstance(target, ndarray):
            np.copyto(target, value)

        # update viewer
        if self.update_hook is not None:
            self.update_hook(self)

    def set_current_data(
        self, value: ResultType, append_addr: List[Union[int, str]] = []
    ) -> None:
        self.set_data(value, self.addr_stack + append_addr)

    def get_data(self, addr_stack: List[Union[int, str]] = []) -> ResultType:
        target = self.data

        # Navigate dict keys from address
        for seg in addr_stack:
            assert isinstance(target, (dict, list)), (
                f"target is not a dict or list: {target}"
            )
            target = target[seg]  # type: ignore

        return target

    def get_current_data(self, append_addr: List[Union[int, str]] = []) -> ResultType:
        return self.get_data(self.addr_stack + append_addr)


class AbsTask(ABC, Generic[T_ResultType]):
    def init(self, ctx: TaskContext, dynamic_pbar: bool = False) -> None:
        """Initialize the task with the current context. If dynamic_pbar is True, the pbar will only show up in the run() method."""
        pass

    @abstractmethod
    def run(self, ctx: TaskContext) -> None:
        """Run the task with the current context."""
        pass

    def cleanup(self) -> None: ...

    @abstractmethod
    def get_default_result(self) -> T_ResultType: ...


T_BatchKeyType = TypeVar("T_BatchKeyType")


class BatchTask(AbsTask[Dict[T_BatchKeyType, ResultType]]):
    def __init__(self, tasks: Dict[T_BatchKeyType, AbsTask[ResultType]]) -> None:
        self.tasks = tasks

    def make_pbar(self, leave: bool) -> tqdm:
        return tqdm(total=len(self.tasks), smoothing=0, leave=leave)

    def init(self, ctx: TaskContext, dynamic_pbar=False) -> None:
        self.dynamic_pbar = dynamic_pbar
        if dynamic_pbar:
            self.task_pbar = None
        else:
            self.task_pbar = self.make_pbar(leave=True)

        for task in self.tasks.values():
            task.init(ctx, dynamic_pbar=True)  # force dynamic pbar for each task

    def run(self, ctx: TaskContext) -> None:
        if self.dynamic_pbar:
            self.task_pbar = self.make_pbar(leave=False)
        else:
            assert self.task_pbar is not None
            self.task_pbar.reset()

        for name, task in self.tasks.items():
            self.task_pbar.set_description(desc=f"Task [{str(name)}]")

            task.run(ctx(addr=name))

            self.task_pbar.update()

        if self.dynamic_pbar:
            self.task_pbar.close()
            self.task_pbar = None

    def cleanup(self) -> None:
        for task in self.tasks.values():
            task.cleanup()

        if not self.dynamic_pbar:
            assert self.task_pbar is not None
            self.task_pbar.close()
            self.task_pbar = None

    def get_default_result(self) -> Dict[T_BatchKeyType, ResultType]:
        return {name: task.get_default_result() for name, task in self.tasks.items()}


class SoftTask(AbsTask[List[T_ResultType]]):
    def __init__(
        self,
        sweep_name: str,
        sweep_values: ndarray,
        update_cfg_fn: Callable[[int, TaskContext, float], None],
        sub_task: AbsTask[T_ResultType],
    ) -> None:
        self.sweep_values = sweep_values
        self.sweep_name = sweep_name
        self.update_cfg_fn = update_cfg_fn
        self.sub_task = sub_task

    def make_pbar(self, leave: bool) -> tqdm:
        return tqdm(
            total=len(self.sweep_values),
            smoothing=0,
            desc=self.sweep_name,
            leave=leave,
        )

    def init(self, ctx: TaskContext, dynamic_pbar=False) -> None:
        self.dynamic_pbar = dynamic_pbar
        if dynamic_pbar:
            self.sweep_pbar = None  # initialize in run()
        else:
            self.sweep_pbar = self.make_pbar(leave=True)

        self.update_cfg_fn(0, ctx, self.sweep_values[0])  # initialize the first value
        self.sub_task.init(ctx, dynamic_pbar=dynamic_pbar)

    def run(self, ctx: TaskContext) -> None:
        if self.dynamic_pbar:
            self.sweep_pbar = self.make_pbar(leave=False)
        else:
            assert self.sweep_pbar is not None
            self.sweep_pbar.reset()

        for i, v in enumerate(self.sweep_values):  # type: ignore
            self.update_cfg_fn(i, ctx, v)

            self.sub_task.run(ctx(addr=i))

            self.sweep_pbar.update()

        if self.dynamic_pbar:
            self.sweep_pbar.close()
            self.sweep_pbar = None

    def cleanup(self) -> None:
        self.sub_task.cleanup()

        if not self.dynamic_pbar:
            assert self.sweep_pbar is not None
            self.sweep_pbar.close()
            self.sweep_pbar = None

    def get_default_result(self) -> List[T_ResultType]:
        return [
            self.sub_task.get_default_result()
            for _ in range(len(self.sweep_values))  # type: ignore
        ]


def default_raw2signal_fn(raw: List[ndarray]) -> ndarray:
    return raw[0][0].dot([1, 1j])


T_RawType = TypeVar("T_RawType")


class HardTask(AbsTask[ndarray], Generic[T_RawType]):
    def __init__(
        self,
        measure_fn: Callable[
            [TaskContext, Callable[[int, T_RawType], None]], T_RawType
        ],
        raw2signal_fn: Callable[[T_RawType], ndarray] = default_raw2signal_fn,
        result_shape: Tuple[int, ...] = (),
    ) -> None:
        self.measure_fn = measure_fn
        self.raw2signal_fn = raw2signal_fn
        self.result_shape = result_shape

    def make_pbar(self, ctx: TaskContext, leave: bool) -> tqdm:
        total = ctx.cfg.get("rounds")
        return tqdm(
            total=total,
            smoothing=0,
            desc="rounds",
            leave=leave,
            disable=total == 1,
        )

    def init(self, ctx: TaskContext, dynamic_pbar=False) -> None:
        self.dynamic_pbar = dynamic_pbar
        if dynamic_pbar:
            self.avg_pbar = None  # initialize in run()
        else:
            self.avg_pbar = self.make_pbar(ctx, leave=True)

    def run(self, ctx: TaskContext) -> None:
        if self.dynamic_pbar:
            self.avg_pbar = self.make_pbar(ctx, leave=False)
        else:
            assert self.avg_pbar is not None
            self.avg_pbar.reset()

        GlobalDeviceManager.setup_devices(ctx.cfg["dev"], progress=False)

        def update_hook(ir: int, raw: Any) -> None:
            self.avg_pbar.update(ir - self.avg_pbar.n)

            ctx.set_current_data(self.raw2signal_fn(raw))

        signal = self.raw2signal_fn(self.measure_fn(ctx, update_hook))

        self.avg_pbar.update(ctx.cfg["rounds"] - self.avg_pbar.n)

        ctx.set_current_data(signal)

        if self.dynamic_pbar:
            assert self.avg_pbar is not None
            self.avg_pbar.close()
            self.avg_pbar = None

    def cleanup(self) -> None:
        if not self.dynamic_pbar:
            assert self.avg_pbar is not None
            self.avg_pbar.close()
            self.avg_pbar = None

    def get_default_result(self) -> ndarray:
        return np.full(self.result_shape, np.nan, dtype=complex)


class RepeatOverTime(AbsTask[List[T_ResultType]]):
    def __init__(
        self,
        name: str,
        num_times: int,
        interval: float,
        task: AbsTask[T_ResultType],
        fail_retry: int = 0,
    ) -> None:
        self.name = name
        self.num_times = num_times
        self.interval = interval
        self.task = task
        self.fail_retry = fail_retry

    def make_pbar(self, leave: bool) -> tqdm:
        return tqdm(total=self.num_times, smoothing=0, desc=self.name, leave=leave)

    def init(self, ctx: TaskContext, dynamic_pbar=False) -> None:
        self.dynamic_pbar = dynamic_pbar
        if dynamic_pbar:
            self.time_pbar = None  # initialize in run()
        else:
            self.time_pbar = self.make_pbar(leave=True)

        self.task.init(ctx, dynamic_pbar=dynamic_pbar)

    def run(self, ctx: TaskContext) -> None:
        if self.dynamic_pbar:
            self.time_pbar = self.make_pbar(leave=False)
        else:
            assert self.time_pbar is not None
            self.time_pbar.reset()

        start_t = time.time() - 2 * self.interval

        for i in range(self.num_times):
            while time.time() - start_t < self.interval:
                time.sleep(0.1)
            start_t = time.time()

            for attempt in range(self.fail_retry):
                try:
                    self.task.run(ctx(addr=i))
                except Exception:
                    print(
                        f"Failed to run task, retrying... ({attempt + 1}/{self.fail_retry})"
                    )
                    continue
                break
            else:
                self.task.run(ctx(addr=i))

            self.time_pbar.update()

        if self.dynamic_pbar:
            assert self.time_pbar is not None
            self.time_pbar.close()
            self.time_pbar = None

    def cleanup(self) -> None:
        self.task.cleanup()

        if not self.dynamic_pbar:
            assert self.time_pbar is not None
            self.time_pbar.close()
            self.time_pbar = None

    def get_default_result(self) -> List[T_ResultType]:
        return [self.task.get_default_result() for _ in range(self.num_times)]


class Runner(Generic[T_ResultType]):
    def __init__(
        self,
        task: AbsTask[T_ResultType],
        update_hook: Optional[Callable[[TaskContext], None]] = None,
        update_interval: Optional[float] = 0.1,
    ) -> None:
        self.task = task
        self.update_hook = min_interval(update_hook, update_interval)

    def run(
        self, init_cfg: TaskConfig, env_dict: Dict[str, Any] = dict()
    ) -> T_ResultType:
        cfg = deepcopy(init_cfg)
        init_result = self.task.get_default_result()

        # initialize devices with progress bar
        GlobalDeviceManager.setup_devices(cfg["dev"], progress=True)

        ctx = TaskContext(cfg, init_result, self.update_hook, env_dict)

        try:
            self.task.init(ctx, dynamic_pbar=False)
            self.task.run(ctx)
            self.task.cleanup()
        except KeyboardInterrupt:
            print("Received KeyboardInterrupt, early stopping the program")
        except Exception:
            print("Error during measurement:")
            print_traceback()

        return ctx.get_data()  # return all data
