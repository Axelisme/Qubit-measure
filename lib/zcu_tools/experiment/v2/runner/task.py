from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from tqdm.auto import tqdm
from typing_extensions import Any, Callable, Generic, Optional, Sequence, Type, TypeVar

from .base import AbsTask
from .state import Result, TaskState

logger = logging.getLogger(__name__)


def default_raw2signal_fn(raw: Sequence[NDArray[np.float64]]) -> NDArray[np.complex128]:
    return raw[0][0].dot([1, 1j])


T_Raw = TypeVar("T_Raw")
T_DType = TypeVar("T_DType", bound=np.number, default=np.complex128)
T_RootResult = TypeVar("T_RootResult", bound=Result)


class Task(
    AbsTask[NDArray[T_DType], T_RootResult],
    Generic[T_RootResult, T_Raw, T_DType],
):
    def __init__(
        self,
        measure_fn: Callable[
            [
                TaskState[NDArray[T_DType], T_RootResult],
                Callable[[int, T_Raw], Any],
            ],
            T_Raw,
        ],
        raw2signal_fn: Callable[[T_Raw], NDArray[T_DType]] = default_raw2signal_fn,
        result_shape: tuple[int, ...] = (),
        dtype: Type[T_DType] = np.complex128,
        pbar_n: Optional[int] = None,
    ) -> None:
        self.measure_fn = measure_fn
        self.raw2signal_fn = raw2signal_fn
        self.result_shape = result_shape
        self.dtype = dtype
        self.pbar_n = pbar_n

        self.avg_pbar: Optional[tqdm] = None
        self.dynamic_pbar: bool = False

    def set_pbar_n(self, pbar_n: Optional[int]) -> None:
        self.pbar_n = pbar_n
        if self.avg_pbar is not None:
            self.avg_pbar.total = pbar_n
            self.avg_pbar.refresh()

    def make_pbar(self, leave: bool) -> tqdm:
        total = self.pbar_n
        return tqdm(
            total=total,
            smoothing=0,
            desc="rounds",
            leave=leave,
            disable=total == 1,
        )

    def init(
        self,
        state: TaskState[NDArray[T_DType], T_RootResult],
        dynamic_pbar: bool = False,
    ) -> None:
        del state  # kept for AbsTask.init contract
        self.dynamic_pbar = dynamic_pbar

        if not dynamic_pbar:
            self.avg_pbar = self.make_pbar(leave=True)

    def run(self, state: TaskState[NDArray[T_DType], T_RootResult]) -> None:
        if self.dynamic_pbar:
            self.avg_pbar = self.make_pbar(leave=False)
        else:
            assert self.avg_pbar is not None
            self.avg_pbar.reset()

        logger.debug(
            "Task.run: path=%s, pbar_n=%s, cfg_keys=%s",
            state.path,
            self.pbar_n,
            list(state.cfg.keys()),
        )

        def update_hook(ir: int, raw: T_Raw) -> None:
            assert self.avg_pbar is not None
            self.avg_pbar.update(ir - self.avg_pbar.n)

            state.set_value(self.raw2signal_fn(raw))

        signal = self.raw2signal_fn(self.measure_fn(state, update_hook))

        if self.pbar_n is not None:
            self.avg_pbar.update(self.pbar_n - self.avg_pbar.n)

        logger.debug("Task.run: done, signal shape=%s", getattr(signal, "shape", "?"))

        state.set_value(signal)

        if self.dynamic_pbar:
            self.avg_pbar.close()
            self.avg_pbar = None

    def cleanup(self) -> None:
        # if raise error in run(), avg_pbar may not be closed
        if self.avg_pbar is not None:
            self.avg_pbar.close()
            self.avg_pbar = None

    def get_default_result(self) -> NDArray[T_DType]:
        return np.full(self.result_shape, np.nan, dtype=self.dtype)
