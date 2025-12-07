from __future__ import annotations

from typing import Dict, Mapping

from tqdm.auto import tqdm
from typing_extensions import Generic

from .base import AbsTask, T_KeyType, T_ResultType, T_TaskContextType


class BatchTask(
    AbsTask[Dict[T_KeyType, T_ResultType], T_TaskContextType],
    Generic[T_ResultType, T_TaskContextType, T_KeyType],
):
    def __init__(
        self, tasks: Mapping[T_KeyType, AbsTask[T_ResultType, T_TaskContextType]]
    ) -> None:
        self.tasks = tasks

    def make_pbar(self, leave: bool) -> tqdm:
        return tqdm(total=len(self.tasks), smoothing=0, leave=leave)

    def init(self, ctx, dynamic_pbar=False) -> None:
        self.dynamic_pbar = dynamic_pbar
        if dynamic_pbar:
            self.task_pbar = None
        else:
            self.task_pbar = self.make_pbar(leave=True)

        for task in self.tasks.values():
            task.init(ctx, dynamic_pbar=True)  # force dynamic pbar for each task

    def run(self, ctx) -> None:
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

    def get_default_result(self) -> Dict[T_KeyType, T_ResultType]:
        return {name: task.get_default_result() for name, task in self.tasks.items()}
