from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Any, Callable, Generic, Optional, Sequence, Type, TypeVar

from zcu_tools.progress_bar import ProgressSink, make_progress_sink

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
        progress_sink: Optional[ProgressSink] = None,
    ) -> None:
        self.measure_fn = measure_fn
        self.raw2signal_fn = raw2signal_fn
        self.result_shape = result_shape
        self.dtype = dtype
        self.pbar_n = pbar_n
        self.progress_sink = progress_sink or make_progress_sink("auto")

        self._progress_started = False
        self.dynamic_pbar: bool = False

    def set_pbar_n(self, pbar_n: Optional[int]) -> None:
        self.pbar_n = pbar_n

    def init(
        self,
        state: TaskState[NDArray[T_DType], T_RootResult],
        dynamic_pbar: bool = False,
    ) -> None:
        del state  # kept for AbsTask.init contract
        self.dynamic_pbar = dynamic_pbar

        if not dynamic_pbar:
            self.progress_sink.start(total=self.pbar_n, desc="rounds", leave=True)
            self._progress_started = True

    def run(self, state: TaskState[NDArray[T_DType], T_RootResult]) -> None:
        if self.dynamic_pbar:
            self.progress_sink.start(total=self.pbar_n, desc="rounds", leave=False)
            self._progress_started = True
        else:
            if self._progress_started:
                self.progress_sink.close()
            self.progress_sink.start(total=self.pbar_n, desc="rounds", leave=True)
            self._progress_started = True

        logger.debug(
            "Task.run: path=%s, pbar_n=%s, cfg_keys=%s",
            state.path,
            self.pbar_n,
            list(state.cfg.keys()),
        )

        def update_hook(ir: int, raw: T_Raw) -> None:
            self.progress_sink.update_to(ir)
            state.set_value(self.raw2signal_fn(raw))

        signal = self.raw2signal_fn(self.measure_fn(state, update_hook))

        if self.pbar_n is not None:
            self.progress_sink.update_to(self.pbar_n)

        logger.debug("Task.run: done, signal shape=%s", getattr(signal, "shape", "?"))

        state.set_value(signal)

        if self.dynamic_pbar:
            self.progress_sink.close()
            self._progress_started = False

    def cleanup(self) -> None:
        # if raise error in run(), avg_pbar may not be closed
        if self._progress_started:
            self.progress_sink.close()
            self._progress_started = False

    def get_default_result(self) -> NDArray[T_DType]:
        return np.full(self.result_shape, np.nan, dtype=self.dtype)
