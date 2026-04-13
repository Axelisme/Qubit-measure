from __future__ import annotations

import logging
import time

from tqdm.auto import tqdm
from typing_extensions import Optional, Sequence, TypeVar

from zcu_tools.utils.func_tools import MinIntervalFunc

from .base import AbsTask
from .state import Result, TaskState

logger = logging.getLogger(__name__)

T_RootResult = TypeVar("T_RootResult", bound=Result)
T_ChildResult = TypeVar("T_ChildResult", bound=Result)
T_Result = TypeVar("T_Result", bound=Result)


def run_with_retries(
    task: AbsTask,
    state,
    retry_time: int,
    dynamic_pbar: bool = False,
    raise_error: bool = True,
) -> None:
    for attempt in range(retry_time + 1):
        try:
            task.run(state)
        except Exception as e:
            if attempt == retry_time:
                if raise_error:
                    raise
                logger.error("run_with_retries: final attempt failed: %s", e)
            else:
                logger.warning(
                    "run_with_retries: attempt %d/%d failed, retrying: %s",
                    attempt + 1, retry_time, e,
                )
                task.cleanup()
                task.init(state, dynamic_pbar=dynamic_pbar)
                continue
        break


class ReTryIfFail(AbsTask[T_Result, T_RootResult]):
    def __init__(self, task: AbsTask[T_Result, T_RootResult], max_retries: int) -> None:
        self.task = task
        self.max_retries = max_retries
        self.dynamic_pbar: bool = False

    def init(self, state, dynamic_pbar: bool = False) -> None:
        self.dynamic_pbar = dynamic_pbar
        self.task.init(state, dynamic_pbar=dynamic_pbar)

    def run(self, state) -> None:
        run_with_retries(
            self.task,
            state,
            retry_time=self.max_retries,
            dynamic_pbar=self.dynamic_pbar,
        )

    def cleanup(self) -> None:
        self.task.cleanup()

    def get_default_result(self) -> T_Result:
        return self.task.get_default_result()


class RepeatOverTime(AbsTask[list[T_ChildResult], T_RootResult]):
    def __init__(
        self,
        name: str,
        num_times: int,
        task: AbsTask[T_ChildResult, T_RootResult],
        interval: float = 0.0,
    ) -> None:
        self.name = name
        self.num_times = num_times
        self.interval = interval
        self.task = task

        self.iter_pbar: Optional[tqdm] = None
        self.time_pbar: Optional[tqdm] = None
        self.dynamic_pbar: bool = False

    def make_pbar(self, leave: bool) -> tuple[tqdm, tqdm]:
        return (
            tqdm(total=self.num_times, smoothing=0, desc=self.name, leave=leave),
            tqdm(
                total=self.interval,
                smoothing=0,
                desc="Passing Time",
                leave=leave,
                miniters=0.2,
                bar_format="{desc}: {bar} {n:.1f}/{total:.1f} s",
                disable=self.interval == 0.0,
            ),
        )

    def init(
        self,
        state: TaskState[list[T_ChildResult], T_RootResult],
        dynamic_pbar: bool = False,
    ) -> None:
        self.dynamic_pbar = dynamic_pbar

        if not dynamic_pbar:
            self.iter_pbar, self.time_pbar = self.make_pbar(leave=True)

        state.env["repeat_idx"] = 0

        self.task.init(state.child(0), dynamic_pbar=dynamic_pbar)

    def run(self, state: TaskState[list[T_ChildResult], T_RootResult]) -> None:
        if self.dynamic_pbar:
            self.iter_pbar, self.time_pbar = self.make_pbar(leave=False)
        else:
            assert self.iter_pbar is not None
            assert self.time_pbar is not None
            self.iter_pbar.reset()
            self.time_pbar.reset()

        start_t = time.time() - 2 * self.interval

        for i in range(self.num_times):
            while time.time() - start_t < self.interval:
                pass_time = round(time.time() - start_t, 1)
                self.time_pbar.update(pass_time - self.time_pbar.n)  # type: ignore[arg-type]

                time.sleep(0.1)
            self.time_pbar.reset()

            start_t = time.time()

            state.env["repeat_idx"] = i

            self.task.run(state.child(i))

            self.iter_pbar.update()

            # If have time left, force trigger hooks
            if time.time() - start_t < self.interval:
                with MinIntervalFunc.force_execute():
                    state._trigger_update()

        if self.dynamic_pbar:
            self.iter_pbar.close()
            self.time_pbar.close()
            self.iter_pbar = None
            self.time_pbar = None

    def cleanup(self) -> None:
        self.task.cleanup()

        if self.iter_pbar is not None:
            self.iter_pbar.close()
            self.iter_pbar = None

        if self.time_pbar is not None:
            self.time_pbar.close()
            self.time_pbar = None

    def get_default_result(self) -> list[T_ChildResult]:
        return [self.task.get_default_result() for _ in range(self.num_times)]
