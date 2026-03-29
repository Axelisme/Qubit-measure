from __future__ import annotations

from tqdm.auto import tqdm
from typing_extensions import Any, Callable, Optional, Sequence, TypeVar, Union

from .base import AbsTask
from .state import Result, TaskState

T_Value = TypeVar("T_Value", bound=Union[int, float, complex])
T_RootResult = TypeVar("T_RootResult", bound=Result)
T_ChildResult = TypeVar("T_ChildResult", bound=Result)


class Scan(AbsTask[Sequence[T_ChildResult], T_RootResult]):
    def __init__(
        self,
        name: str,
        values: Sequence[T_Value],
        before_each: Callable[
            [int, TaskState[Sequence[T_ChildResult], T_RootResult], T_Value], Any
        ],
        task: AbsTask[T_ChildResult, T_RootResult],
    ) -> None:
        self.sweep_values = values
        self.sweep_name = name
        self.update_cfg_fn = before_each
        self.sub_task = task

        self.sweep_pbar: Optional[tqdm] = None
        self.dynamic_pbar: bool = False

    def make_pbar(self, leave: bool) -> tqdm:
        return tqdm(
            total=len(self.sweep_values),
            smoothing=0,
            desc=self.sweep_name,
            leave=leave,
        )

    def init(
        self,
        state: TaskState[Sequence[T_ChildResult], T_RootResult],
        dynamic_pbar: bool = False,
    ) -> None:
        self.dynamic_pbar = dynamic_pbar

        if not dynamic_pbar:
            self.sweep_pbar = self.make_pbar(leave=True)

        # Pre-update cfg for the first value
        if len(self.sweep_values) > 0:
            self.update_cfg_fn(0, state, self.sweep_values[0])

        self.sub_task.init(state.child(0), dynamic_pbar=dynamic_pbar)

    def run(self, state: TaskState[Sequence[T_ChildResult], T_RootResult]) -> None:
        if self.dynamic_pbar:
            self.sweep_pbar = self.make_pbar(leave=False)
        else:
            assert self.sweep_pbar is not None
            self.sweep_pbar.reset()

        for i, v in enumerate(self.sweep_values):
            self.update_cfg_fn(i, state, v)

            self.sub_task.run(state.child(i))

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
