from __future__ import annotations

import logging

from tqdm.auto import tqdm
from typing_extensions import Hashable, Mapping, Optional, TypeVar

from .base import AbsTask, task_manager
from .state import Result

logger = logging.getLogger(__name__)

T_Key = TypeVar("T_Key", bound=Hashable)

T_RootResult = TypeVar("T_RootResult", bound=Result)
T_ChildResult = TypeVar("T_ChildResult", bound=Result)


class BatchTask(AbsTask[dict[T_Key, T_ChildResult], T_RootResult]):
    def __init__(
        self,
        tasks: Mapping[T_Key, AbsTask[T_ChildResult, T_RootResult]],
    ) -> None:
        self.tasks = tasks

        self.task_pbar: Optional[tqdm] = None
        self.dynamic_pbar: bool = False

    def make_pbar(self, leave: bool) -> tqdm:
        return tqdm(total=len(self.tasks), smoothing=0, leave=leave)

    def init(self, state, dynamic_pbar: bool = False) -> None:
        self.dynamic_pbar = dynamic_pbar

        if not dynamic_pbar:
            self.task_pbar = self.make_pbar(leave=True)

        # force dynamic pbar for each task
        for name, task in self.tasks.items():
            task.init(state.child(name), dynamic_pbar=True)

    def run(self, state) -> None:
        if self.dynamic_pbar:
            self.task_pbar = self.make_pbar(leave=False)
        else:
            assert self.task_pbar is not None
            self.task_pbar.reset()

        logger.debug("BatchTask.run: %d tasks", len(self.tasks))

        for name, task in self.tasks.items():
            task_manager.check_cancelled()
            self.task_pbar.set_description(desc=f"Task [{str(name)}]")
            logger.debug("BatchTask.run: starting task '%s'", name)

            task.run(state.child(name))

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

    def get_default_result(self) -> dict[T_Key, T_ChildResult]:
        return {name: task.get_default_result() for name, task in self.tasks.items()}
