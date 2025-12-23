from __future__ import annotations

from tqdm.auto import tqdm
from typing_extensions import Hashable, Mapping, TypeVar, Optional

from .base import AbsTask, Result, TaskConfig

T_Key = TypeVar("T_Key", bound=Hashable)

T_RootResult = TypeVar("T_RootResult", bound=Result)
T_ChildResult = TypeVar("T_ChildResult", bound=Result)
T_TaskConfig = TypeVar("T_TaskConfig", bound=TaskConfig)


class BatchTask(AbsTask[Mapping[T_Key, T_ChildResult], T_RootResult, T_TaskConfig]):
    def __init__(
        self,
        tasks: Mapping[T_Key, AbsTask[T_ChildResult, T_RootResult, T_TaskConfig]],
    ) -> None:
        self.tasks = tasks

        self.task_pbar: Optional[tqdm] = None

    def make_pbar(self, leave: bool) -> tqdm:
        return tqdm(total=len(self.tasks), smoothing=0, leave=leave)

    def init(self, ctx, dynamic_pbar=False) -> None:
        self.dynamic_pbar = dynamic_pbar

        if not dynamic_pbar:
            self.task_pbar = self.make_pbar(leave=True)

        # force dynamic pbar for each task
        for name, task in self.tasks.items():
            task.init(ctx(addr=name), dynamic_pbar=True)

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

        if self.task_pbar is not None:
            self.task_pbar.close()
            self.task_pbar = None

    def get_default_result(self) -> Mapping[T_Key, T_ChildResult]:
        return {name: task.get_default_result() for name, task in self.tasks.items()}
