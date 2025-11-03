import time
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy import ndarray
from tqdm.auto import tqdm

from zcu_tools.device import GlobalDeviceManager
from zcu_tools.utils.debug import print_traceback
from zcu_tools.utils.func_tools import min_interval

ResultType = Union[Dict[Any, "ResultType"], List["ResultType"], ndarray]


@dataclass
class TaskContext:
    cfg: Dict[str, Any]
    data: ResultType
    update_hook: Optional[Callable[["TaskContext"], None]] = None
    addr_stack: List[Union[int, Any]] = field(default_factory=list)

    def __call__(self, addr: Union[int, Any]) -> "TaskContext":
        return TaskContext(
            deepcopy(self.cfg),
            self.data,
            self.update_hook,
            self.addr_stack + [addr],
        )

    def is_empty_stack(self) -> bool:
        return len(self.addr_stack) == 0

    def set_data(
        self, value: ResultType, addr_stack: List[Union[int, Any]] = []
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

    def set_current_data(self, value: ResultType) -> None:
        self.set_data(value, self.addr_stack)

    def get_data(self, addr_stack: List[Union[int, str]] = []) -> ResultType:
        target = self.data

        # Navigate dict keys from address
        for seg in addr_stack:
            assert isinstance(target, (dict, list)), (
                f"target is not a dict or list: {target}"
            )
            target = target[seg]

        return target

    def get_current_data(self) -> ResultType:
        return self.get_data(self.addr_stack)


class AbsTask(ABC):
    def init(self, ctx: TaskContext, keep: bool = True) -> None:
        """Initialize the task with the current context. Do not modify the context."""
        pass

    @abstractmethod
    def run(self, ctx: TaskContext) -> None:
        """Run the task with the current context."""
        pass

    def cleanup(self) -> None:
        pass

    @abstractmethod
    def get_default_result(self) -> ResultType:
        pass


class BatchTask(AbsTask):
    def __init__(self, tasks: Dict[Any, AbsTask]) -> None:
        self.tasks = tasks

        self.task_pbar = None

    def init(self, ctx: TaskContext, keep=True) -> None:
        self.task_pbar = tqdm(total=len(self.tasks), smoothing=0, leave=keep)

    def run(self, ctx: TaskContext) -> None:
        assert self.task_pbar is not None
        self.task_pbar.reset()

        for name, task in self.tasks.items():
            self.task_pbar.set_description(desc=f"Task [{str(name)}]")

            task.init(ctx, keep=False)
            task.run(ctx(addr=name))
            task.cleanup()

            self.task_pbar.update()

    def cleanup(self) -> None:
        assert self.task_pbar is not None
        self.task_pbar.close()
        self.task_pbar = None

    def get_default_result(self) -> ResultType:
        return {name: task.get_default_result() for name, task in self.tasks.items()}


class SoftTask(AbsTask):
    def __init__(
        self,
        sweep_name: str,
        sweep_values: ndarray,
        update_cfg_fn: Callable[[int, TaskContext, float], None],
        sub_task: AbsTask,
    ) -> None:
        self.sweep_values = sweep_values
        self.sweep_name = sweep_name
        self.update_cfg_fn = update_cfg_fn
        self.sub_task = sub_task

        self.sweep_pbar = None

    def init(self, ctx: TaskContext, keep=True) -> None:
        self.sweep_pbar = tqdm(
            total=len(self.sweep_values), smoothing=0, desc=self.sweep_name, leave=keep
        )
        self.sub_task.init(ctx, keep=keep)

    def run(self, ctx: TaskContext) -> None:
        assert self.sweep_pbar is not None
        self.sweep_pbar.reset()

        for i, v in enumerate(self.sweep_values):
            self.update_cfg_fn(i, ctx, v)

            self.sub_task.run(ctx(addr=i))

            self.sweep_pbar.update()

    def cleanup(self) -> None:
        self.sub_task.cleanup()
        assert self.sweep_pbar is not None
        self.sweep_pbar.close()
        self.sweep_pbar = None

    def get_default_result(self) -> ResultType:
        return [
            self.sub_task.get_default_result() for _ in range(len(self.sweep_values))
        ]


class HardTask(AbsTask):
    def __init__(
        self,
        measure_fn: Callable[[TaskContext, Callable[[int, Any], None]], Any],
        raw2signal_fn: Callable[[Any], ndarray] = lambda raw: raw[0][0].dot([1, 1j]),
        result_shape: Tuple[int, ...] = (),
    ) -> None:
        self.measure_fn = measure_fn
        self.raw2signal_fn = raw2signal_fn
        self.result_shape = result_shape

        self.avg_pbar = None

    def init(self, ctx: TaskContext, keep=True) -> None:
        if ctx.cfg["rounds"] > 1:
            self.avg_pbar = tqdm(
                total=ctx.cfg["rounds"], smoothing=0, desc="rounds", leave=keep
            )

    def run(self, ctx: TaskContext) -> None:
        if self.avg_pbar is not None:
            self.avg_pbar.reset()

        GlobalDeviceManager.setup_devices(ctx.cfg["dev"], progress=False)

        def update_hook(ir: int, raw: Any) -> None:
            if self.avg_pbar is not None:
                self.avg_pbar.update(ir - self.avg_pbar.n)

            ctx.set_current_data(self.raw2signal_fn(raw))

        signal = self.raw2signal_fn(self.measure_fn(ctx, update_hook))

        if self.avg_pbar is not None:
            self.avg_pbar.update(ctx.cfg["rounds"] - self.avg_pbar.n)

        ctx.set_current_data(signal)

    def cleanup(self) -> None:
        if self.avg_pbar is not None:
            self.avg_pbar.close()
        self.avg_pbar = None

    def get_default_result(self) -> ResultType:
        return np.full(self.result_shape, np.nan, dtype=complex)


class RepeatOverTime(AbsTask):
    def __init__(
        self,
        name: str,
        num_times: int,
        interval: float,
        task: AbsTask,
        fail_retry: int = 0,
    ) -> None:
        self.name = name
        self.num_times = num_times
        self.interval = interval
        self.task = task
        self.fail_retry = fail_retry

        self.pbar = None

    def init(self, ctx: TaskContext, keep=True) -> None:
        self.pbar = tqdm(total=self.num_times, smoothing=0, desc=self.name, leave=keep)
        self.task.init(ctx, keep=keep)

    def run(self, ctx: TaskContext) -> None:
        assert self.pbar is not None
        self.pbar.reset()

        start_t = time.time()

        for i in range(self.num_times):
            while time.time() - start_t < self.interval:
                time.sleep(self.interval / 10)
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

            self.pbar.update()

    def cleanup(self) -> None:
        self.task.cleanup()
        assert self.pbar is not None
        self.pbar.close()
        self.pbar = None

    def get_default_result(self) -> ResultType:
        return [self.task.get_default_result() for _ in range(self.num_times)]


class Runner:
    def __init__(
        self,
        task: AbsTask,
        update_hook: Optional[Callable[[TaskContext], None]] = None,
        update_interval: float = 1.0,
    ) -> None:
        self.task = task
        self.update_hook = min_interval(update_hook, update_interval)

    def run(self, init_cfg: Dict[str, Any]) -> ResultType:
        cfg = deepcopy(init_cfg)
        init_result = self.task.get_default_result()

        # initialize devices with progress bar
        GlobalDeviceManager.setup_devices(cfg["dev"], progress=True)

        ctx = TaskContext(cfg, init_result, self.update_hook)

        try:
            self.task.init(ctx)
            self.task.run(ctx)
            self.task.cleanup()
        except KeyboardInterrupt:
            print("Received KeyboardInterrupt, early stopping the program")
        except Exception:
            print("Error during measurement:")
            print_traceback()

        if not ctx.is_empty_stack():
            warnings.warn("TaskContext is not empty, some data may be corrupted")

        return ctx.get_data()  # force return all data
