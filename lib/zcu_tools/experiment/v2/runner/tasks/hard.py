from __future__ import annotations

from typing import Generic, Sequence

import numpy as np
from numpy.typing import DTypeLike, NDArray
from tqdm.auto import tqdm
from typing_extensions import Any, Callable, Tuple, TypeVar

from zcu_tools.device import GlobalDeviceManager

from .base import AbsTask, Result, TaskConfig, TaskContextView


def default_raw2signal_fn(raw: Sequence[NDArray[np.float64]]) -> NDArray[np.complex128]:
    return raw[0][0].dot([1, 1j])


T_Raw = TypeVar("T_Raw")
T_DType = TypeVar("T_DType", bound=np.number)
T_RootResult = TypeVar("T_RootResult", bound=Result)
T_TaskConfig = TypeVar("T_TaskConfig", bound=TaskConfig)


class HardTask(
    AbsTask[NDArray[T_DType], T_RootResult, T_TaskConfig],
    Generic[T_DType, T_RootResult, T_TaskConfig, T_Raw],
):
    def __init__(
        self,
        measure_fn: Callable[
            [
                TaskContextView[NDArray[T_DType], T_RootResult, T_TaskConfig],
                Callable[[int, T_Raw], Any],
            ],
            T_Raw,
        ],
        raw2signal_fn: Callable[[T_Raw], NDArray[T_DType]] = default_raw2signal_fn,
        result_shape: Tuple[int, ...] = (),
        dtype: DTypeLike = complex,
    ) -> None:
        self.measure_fn = measure_fn
        self.raw2signal_fn = raw2signal_fn
        self.result_shape = result_shape
        self.dtype = dtype

    def make_pbar(self, ctx, leave: bool) -> tqdm:
        total = ctx.cfg.get("rounds")
        return tqdm(
            total=total,
            smoothing=0,
            desc="rounds",
            leave=leave,
            disable=total == 1,
        )

    def init(self, ctx, dynamic_pbar=False) -> None:
        self.dynamic_pbar = dynamic_pbar
        if dynamic_pbar:
            self.avg_pbar = None  # initialize in run()
        else:
            self.avg_pbar = self.make_pbar(ctx, leave=True)

    def run(self, ctx) -> None:
        assert "rounds" in ctx.cfg

        if self.dynamic_pbar:
            self.avg_pbar = self.make_pbar(ctx, leave=False)
        else:
            assert self.avg_pbar is not None
            self.avg_pbar.reset()

        GlobalDeviceManager.setup_devices(ctx.cfg["dev"], progress=False)

        def update_hook(ir: int, raw) -> None:
            assert self.avg_pbar is not None
            self.avg_pbar.update(ir - self.avg_pbar.n)

            ctx.set_data(self.raw2signal_fn(raw))

        signal = self.raw2signal_fn(self.measure_fn(ctx, update_hook))

        self.avg_pbar.update(ctx.cfg["rounds"] - self.avg_pbar.n)

        ctx.set_data(signal)

        if self.dynamic_pbar:
            assert self.avg_pbar is not None
            self.avg_pbar.close()
            self.avg_pbar = None

    def cleanup(self) -> None:
        if not self.dynamic_pbar:
            assert self.avg_pbar is not None
            self.avg_pbar.close()
            self.avg_pbar = None

    def get_default_result(self) -> NDArray[T_DType]:
        return np.full(self.result_shape, np.nan, dtype=self.dtype)
