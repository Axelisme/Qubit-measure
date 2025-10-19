import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy import ndarray
from tqdm.auto import tqdm

from zcu_tools.device import GlobalDeviceManager
from zcu_tools.utils.debug import print_traceback

ResultType = Union[Dict[Any, "ResultType"], List["ResultType"], ndarray]


def default_raw2signal_fn(raw_signal: Any) -> ndarray:
    return raw_signal[0][0].dot([1, 1j])


class TaskContext:
    def __init__(
        self,
        cfg: Dict[str, Any],
        init_result: ResultType,
        update_hook: Optional[Callable[["TaskContext"], None]] = None,
    ) -> None:
        self.cfg = cfg
        self.data = init_result
        self.update_hook = update_hook

        self.addr_stack: List[Union[int, Any]] = []
        self.cfg_stack: List[Dict[str, Any]] = []
        self.last_call_addr = None

    def __call__(self, addr: Union[int, Any]) -> "TaskContext":
        self.last_call_addr = addr
        return self

    def __enter__(self) -> None:
        assert self.last_call_addr is not None
        self.addr_stack.append(self.last_call_addr)
        self.cfg_stack.append(deepcopy(self.cfg))  # record cfg

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.addr_stack.pop()
        self.cfg = self.cfg_stack.pop()  # restore cfg
        self.last_call_addr = None

    def is_empty(self) -> bool:
        return len(self.addr_stack) == 0

    def set_data(
        self, value: ResultType, addr_stack: Optional[List[Union[int, str]]] = None
    ) -> None:
        """Set data at the current address."""
        value = np.asarray(value)

        # default to current address stack
        if addr_stack is None:
            addr_stack = self.addr_stack

        target = self.get_data(addr_stack)

        if not isinstance(target, type(value)):
            raise TypeError(f"expected {type(target)}, got {type(value)}")

        if isinstance(target, dict):
            target.update(value)
        elif isinstance(target, list):
            target.extend(value)
        elif isinstance(target, ndarray):
            np.copyto(target, value)

        # update viewer
        if self.update_hook is not None:
            self.update_hook(self)

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
    def init(self, ctx: TaskContext, keep:bool = True) -> None:
        pass

    @abstractmethod
    def run(self, ctx: TaskContext) -> None:
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

    def init(self, ctx: TaskContext, keep = True) -> None:
        self.task_pbar = tqdm(total=len(self.tasks), smoothing=0, leave=keep)

    def run(self, ctx: TaskContext) -> None:
        assert self.task_pbar is not None
        self.task_pbar.reset()

        for name, task in self.tasks.items():
            self.task_pbar.set_description(desc=f"Task [{str(name)}]")

            task.init(ctx, keep=False)
            with ctx(addr=name):
                task.run(ctx)
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

    def init(self, ctx: TaskContext, keep = True) -> None:
        self.sweep_pbar = tqdm(
            total=len(self.sweep_values), smoothing=0, desc=self.sweep_name, leave=keep
        )
        self.sub_task.init(ctx, keep=keep)

    def run(self, ctx: TaskContext) -> None:
        assert self.sweep_pbar is not None
        self.sweep_pbar.reset()

        for i, v in enumerate(self.sweep_values):
            self.update_cfg_fn(i, ctx, v)

            with ctx(addr=i):
                self.sub_task.run(ctx)

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
        raw2signal_fn: Callable[[Any], ndarray] = default_raw2signal_fn,
        result_shape: Tuple[int, ...] = (),
    ) -> None:
        self.measure_fn = measure_fn
        self.raw2signal_fn = raw2signal_fn
        self.result_shape = result_shape

        self.avg_pbar = None

    def init(self, ctx: TaskContext, keep = True) -> None:
        self.avg_pbar = tqdm(total=ctx.cfg["rounds"], smoothing=0, desc="rounds", leave=keep)

    def run(self, ctx: TaskContext) -> None:
        assert self.avg_pbar is not None
        self.avg_pbar.reset()

        GlobalDeviceManager.setup_devices(ctx.cfg["dev"], progress=False)

        def update_hook(ir: int, raw: Any) -> None:
            self.avg_pbar.update(ir - self.avg_pbar.n)

            ctx.set_data(self.raw2signal_fn(raw))

        signal = self.raw2signal_fn(self.measure_fn(ctx, update_hook))

        self.avg_pbar.update(ctx.cfg["rounds"] - self.avg_pbar.n)

        ctx.set_data(signal)

    def cleanup(self) -> None:
        assert self.avg_pbar is not None
        self.avg_pbar.close()
        self.avg_pbar = None

    def get_default_result(self) -> ResultType:
        return np.full(self.result_shape, np.nan, dtype=complex)


class AnalysisTask(AbsTask):
    def __init__(
        self, analysis_fn: Callable[[TaskContext], ResultType], init_result: ResultType
    ) -> None:
        self.analysis_fn = analysis_fn
        self.init_result = init_result

    def run(self, ctx: TaskContext) -> None:
        ctx.set_data(self.analysis_fn(ctx))

    def get_default_result(self) -> ResultType:
        return deepcopy(self.init_result)


class Runner:
    def __init__(
        self,
        task: AbsTask,
        update_hook: Optional[Callable[[TaskContext], None]] = None,
    ) -> None:
        self.task = task
        self.update_hook = update_hook

    def run(self, init_cfg: Dict[str, Any]) -> ResultType:
        cfg = deepcopy(init_cfg)
        init_result = self.task.get_default_result()

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

        if not ctx.is_empty():
            warnings.warn("TaskContext is not empty, some data may be corrupted")

        return ctx.get_data()  # force return all data
