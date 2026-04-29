from __future__ import annotations

import logging

from typing_extensions import Any, Callable, Optional, Sequence, TypeVar, Union

from zcu_tools.progress_bar import BaseProgressBar, make_pbar

from .base import AbsTask
from .state import T_Cfg, T_ChildResult, T_RootResult, TaskState

logger = logging.getLogger(__name__)

T_Value = TypeVar("T_Value", bound=Union[int, float, complex])


class Scan(AbsTask[list[T_ChildResult], T_RootResult, T_Cfg]):
    def __init__(
        self,
        name: str,
        values: Sequence[T_Value],
        before_each: Callable[
            [int, TaskState[list[T_ChildResult], T_RootResult, T_Cfg], T_Value], Any
        ],
        task: AbsTask[T_ChildResult, T_RootResult, T_Cfg],
    ) -> None:
        self.sweep_values = list(values)
        self.sweep_name = name
        self.before_each_fn = before_each
        self.sub_task = task

        self.sweep_pbar: Optional[BaseProgressBar] = None
        self.dynamic_pbar: bool = False

    def _build_pbar(self, leave: bool) -> BaseProgressBar:
        return make_pbar(
            total=len(self.sweep_values),
            smoothing=0,
            desc=self.sweep_name,
            leave=leave,
        )

    def init(self, dynamic_pbar: bool = False) -> None:
        self.dynamic_pbar = dynamic_pbar

        if not dynamic_pbar:
            self.sweep_pbar = self._build_pbar(leave=True)

        self.sub_task.init(dynamic_pbar=dynamic_pbar)

    def run(self, state: TaskState[list[T_ChildResult], T_RootResult, T_Cfg]) -> None:
        if self.dynamic_pbar:
            self.sweep_pbar = self._build_pbar(leave=False)
        else:
            assert self.sweep_pbar is not None
            self.sweep_pbar.reset()

        logger.debug(
            "Scan.run: name='%s', n_values=%d, path=%s",
            self.sweep_name,
            len(self.sweep_values),
            state.path,
        )

        for i, v in enumerate(self.sweep_values):
            self.before_each_fn(i, state, v)

            logger.debug(
                "Scan '%s' step %d/%d, value=%s",
                self.sweep_name,
                i + 1,
                len(self.sweep_values),
                v,
            )

            self.sub_task.run(state.child(i))

            assert self.sweep_pbar is not None
            self.sweep_pbar.update()

        if self.dynamic_pbar:
            self.sweep_pbar.close()
            self.sweep_pbar = None

    def cleanup(self) -> None:
        self.sub_task.cleanup()

        if self.sweep_pbar is not None:
            self.sweep_pbar.close()
            self.sweep_pbar = None

    def get_default_result(self) -> list[T_ChildResult]:
        return [
            self.sub_task.get_default_result() for _ in range(len(self.sweep_values))
        ]
