from __future__ import annotations

from numbers import Number
from typing import Sequence

from tqdm.auto import tqdm
from typing_extensions import Any, Callable, Generic, List, TypeVar

from .base import AbsTask, T_ResultType, T_TaskContextType

T_ValueType = TypeVar("T_ValueType", bound=Number)


class SoftTask(
    AbsTask[Sequence[T_ResultType], T_TaskContextType],
    Generic[T_ResultType, T_TaskContextType, T_ValueType],
):
    def __init__(
        self,
        sweep_name: str,
        sweep_values: Sequence[T_ValueType],
        update_cfg_fn: Callable[[int, T_TaskContextType, T_ValueType], Any],
        sub_task: AbsTask[T_ResultType, T_TaskContextType],
    ) -> None:
        self.sweep_values = sweep_values
        self.sweep_name = sweep_name
        self.update_cfg_fn = update_cfg_fn
        self.sub_task = sub_task

    def make_pbar(self, leave: bool) -> tqdm:
        return tqdm(
            total=len(self.sweep_values),
            smoothing=0,
            desc=self.sweep_name,
            leave=leave,
        )

    def init(self, ctx, dynamic_pbar=False) -> None:
        self.dynamic_pbar = dynamic_pbar
        if dynamic_pbar:
            self.sweep_pbar = None  # initialize in run()
        else:
            self.sweep_pbar = self.make_pbar(leave=True)

        self.update_cfg_fn(0, ctx, self.sweep_values[0])  # initialize the first value
        self.sub_task.init(ctx, dynamic_pbar=dynamic_pbar)

    def run(self, ctx) -> None:
        if self.dynamic_pbar:
            self.sweep_pbar = self.make_pbar(leave=False)
        else:
            assert self.sweep_pbar is not None
            self.sweep_pbar.reset()

        for i, v in enumerate(self.sweep_values):
            self.update_cfg_fn(i, ctx, v)

            self.sub_task.run(ctx(addr=i))

            self.sweep_pbar.update()

        if self.dynamic_pbar:
            self.sweep_pbar.close()
            self.sweep_pbar = None

    def cleanup(self) -> None:
        self.sub_task.cleanup()

        if not self.dynamic_pbar:
            assert self.sweep_pbar is not None
            self.sweep_pbar.close()
            self.sweep_pbar = None

    def get_default_result(self) -> List[T_ResultType]:
        return [
            self.sub_task.get_default_result() for _ in range(len(self.sweep_values))
        ]
