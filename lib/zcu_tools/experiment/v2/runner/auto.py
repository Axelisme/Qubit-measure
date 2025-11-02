from __future__ import annotations

from abc import abstractmethod
from typing import Dict, List

import numpy as np

from .base import AbsTask, BatchTask, ResultType, TaskContext


class AbsAutoTask(AbsTask):
    def __init__(
        self, needed_tags: List[str] = [], provided_tags: List[str] = []
    ) -> None:
        self.needed_tags = needed_tags
        self.provided_tags = provided_tags

    @abstractmethod
    def run(
        self, ctx: TaskContext, need_infos: Dict[str, complex]
    ) -> Dict[str, complex]:
        """Run the task with current context and needed information. Return provided information."""


class AutoBatchTask(BatchTask):
    def __init__(self, tasks: Dict[str, AbsAutoTask]) -> None:
        if any(name == "meta_infos" for name in tasks.keys()):
            raise ValueError("'meta_infos' is a reserved name for the meta information")

        self.tasks = tasks

        self.task_pbar = None

    def run(self, ctx: TaskContext) -> None:
        assert self.task_pbar is not None
        self.task_pbar.reset()

        meta_infos: Dict[str, complex] = {}

        for name, task in self.tasks.items():
            self.task_pbar.set_description(desc=f"Task [{str(name)}]")

            # collect needed information from previous tasks
            need_infos: Dict[str, complex] = {}
            for _name, _task in self.tasks.items():
                for tag in _task.provided_tags:
                    if tag in task.needed_tags:
                        # only last matching task's information is used
                        need_infos[tag] = meta_infos[_name][tag]
            missing_tags = [tag for tag in task.needed_tags if tag not in need_infos]
            if len(missing_tags) > 0:
                raise RuntimeError(
                    f"Task [{str(name)}] is missing needed tags: {missing_tags}"
                )

            task.init(ctx, keep=False)
            provided_infos = task.run(ctx(addr=name), need_infos)
            task.cleanup()

            meta_infos[name] = provided_infos
            self.task_pbar.update()

            # set meta information to context
            ctx(addr="meta_infos").set_current_data(meta_infos)

    def get_default_result(self) -> ResultType:
        default_result = {
            name: task.get_default_result() for name, task in self.tasks.items()
        }
        default_result["meta_infos"] = {
            name: {tag: np.array(np.nan, dtype=complex) for tag in task.provided_tags}
            for name, task in self.tasks.items()
        }
        return default_result
