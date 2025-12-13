from __future__ import annotations

from time import time
from typing import Sequence

from tqdm.auto import tqdm
from typing_extensions import List, TypeVar

from .base import AbsTask, Result, TaskConfig

T_RootResult = TypeVar("T_RootResult", bound=Result)
T_ChildResult = TypeVar("T_ChildResult", bound=Result)
T_Result = TypeVar("T_Result", bound=Result)
T_TaskConfig = TypeVar("T_TaskConfig", bound=TaskConfig)


def run_with_retries(
    task: AbsTask,
    ctx,
    retry_time: int,
    dynamic_pbar: bool = False,
    raise_error: bool = True,
) -> None:
    for attempt in range(retry_time + 1):
        try:
            task.run(ctx)
        except Exception:
            if attempt == retry_time:
                if raise_error:
                    raise
            else:
                print(f"Failed to run task, retrying... ({attempt + 1}/{retry_time})")
                task.cleanup()  # cleanup and re-init
                task.init(ctx, dynamic_pbar=dynamic_pbar)
                continue
        break


class ReTryIfFail(AbsTask[T_Result, T_RootResult, T_TaskConfig]):
    def __init__(
        self, task: AbsTask[T_Result, T_RootResult, T_TaskConfig], max_retries: int
    ) -> None:
        self.task = task
        self.max_retries = max_retries

    def init(self, ctx, dynamic_pbar=False) -> None:
        self.dynamic_pbar = dynamic_pbar
        self.task.init(ctx, dynamic_pbar=dynamic_pbar)

    def run(self, ctx) -> None:
        run_with_retries(
            self.task, ctx, retry_time=self.max_retries, dynamic_pbar=self.dynamic_pbar
        )

    def cleanup(self) -> None:
        self.task.cleanup()

    def get_default_result(self) -> T_Result:
        return self.task.get_default_result()


class RepeatOverTime(AbsTask[Sequence[T_ChildResult], T_RootResult, T_TaskConfig]):
    def __init__(
        self,
        name: str,
        num_times: int,
        interval: float,
        task: AbsTask[T_ChildResult, T_RootResult, T_TaskConfig],
    ) -> None:
        self.name = name
        self.num_times = num_times
        self.interval = interval
        self.task = task

    def make_pbar(self, leave: bool) -> tqdm:
        return tqdm(total=self.num_times, smoothing=0, desc=self.name, leave=leave)

    def init(self, ctx, dynamic_pbar=False) -> None:
        self.dynamic_pbar = dynamic_pbar
        if dynamic_pbar:
            self.time_pbar = None  # initialize in run()
        else:
            self.time_pbar = self.make_pbar(leave=True)

        self.task.init(ctx(addr=0), dynamic_pbar=dynamic_pbar)

    def run(self, ctx) -> None:
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

            ctx.env_dict["repeat_idx"] = i

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

    def get_default_result(self) -> List[T_ChildResult]:
        return [self.task.get_default_result() for _ in range(self.num_times)]
