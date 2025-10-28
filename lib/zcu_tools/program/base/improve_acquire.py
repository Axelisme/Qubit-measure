from typing import Callable, List, Optional

import numpy as np
from qick.qick_asm import AcquireMixin

from zcu_tools.utils.async_func import AsyncFunc


class CallbackMixin(AcquireMixin):
    """
    Add callback functionality to the AcquireMixin class
    """

    def acquire(
        self, *args, callback: Optional[Callable[..., None]] = None, **kwargs
    ) -> List[np.ndarray]:
        extra_args = kwargs.pop("extra_args", dict())

        with AsyncFunc(callback) as async_callback:
            extra_args.update(callback=async_callback)

            return super().acquire(*args, extra_args=extra_args, **kwargs)

    def acquire_decimated(
        self, *args, callback: Optional[Callable[..., None]] = None, **kwargs
    ) -> List[np.ndarray]:
        extra_args = kwargs.pop("extra_args", dict())

        with AsyncFunc(callback) as async_callback:
            extra_args.update(callback=async_callback)

            return super().acquire_decimated(*args, extra_args=extra_args, **kwargs)

    def finish_round(self) -> bool:
        not_finish = super().finish_round()

        # trigger the callback function after each round
        assert self.acquire_params is not None
        callback = self.acquire_params["callback"]
        if callback is not None:
            assert callable(callback), "callback must be a callable function"
            assert self.rounds_buf is not None

            round_n = len(self.rounds_buf)
            if self.acquire_params["type"] == "accumulated":
                avg_d = self._summarize_accumulated(self.rounds_buf)
                callback(round_n, avg_d)
            elif self.acquire_params["type"] == "decimated":
                dec_d = self._summarize_decimated(self.rounds_buf)
                callback(round_n, dec_d)
            else:
                raise NotImplementedError(
                    "Callback is not implemented for type other than accumulated or decimated"
                )

        return not_finish


class EarlyStopMixin(AcquireMixin):
    """
    Add early stopping functionality to the AcquireMixin class
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.early_stop = False

    def set_early_stop(self, silent: bool = False) -> None:
        # tell program to return as soon as this round is finished
        if not silent:
            print("Program received early stop signal")
        self.early_stop = True

    def acquire(self, *args, **kwargs) -> List[np.ndarray]:
        self.early_stop = False
        return super().acquire(*args, **kwargs)  # type: ignore

    def acquire_decimated(self, *args, **kwargs) -> List[np.ndarray]:
        self.early_stop = False
        return super().acquire_decimated(*args, **kwargs)  # type: ignore

    def finish_round(self) -> bool:
        not_finish = super().finish_round()
        if not_finish and self.early_stop:
            assert self.rounds_pbar is not None
            self.rounds_pbar.close()
        return not_finish and not self.early_stop


class ImproveAcquireMixin(EarlyStopMixin, CallbackMixin):
    pass
