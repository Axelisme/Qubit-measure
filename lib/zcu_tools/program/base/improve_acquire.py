from typing import Callable, List, Optional

import numpy as np
from qick.qick_asm import AcquireMixin


class StdErrorMixin(AcquireMixin):
    """
    Add standard error information for acquired method to the AcquireMixin class
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.stderr_buf: Optional[List[List[np.ndarray]]] = None

    def get_stderr_raw(self) -> Optional[List[List[np.ndarray]]]:
        return self.stderr_buf

    def get_stderr(self) -> Optional[List[np.ndarray]]:
        rounds_buf = self.get_rounds()
        stderr_buf = self.get_stderr_raw()
        if stderr_buf is None:
            return None

        assert rounds_buf is not None
        return self._summarize_accumulated_std(rounds_buf, stderr_buf)

    def _summarize_accumulated_std(
        self, rounds_buf: List[List[np.ndarray]], std_buf: List[List[np.ndarray]]
    ) -> List[np.ndarray]:
        return [
            np.sqrt(
                np.mean(
                    [u[i] ** 2 + s[i] ** 2 for u, s in zip(rounds_buf, std_buf)], axis=0
                )
                - np.mean([u[i] for u in rounds_buf], axis=0) ** 2
            )
            for i in range(len(self.ro_chs))  # type: ignore
        ]

    def acquire(self, *args, record_stderr: bool = False, **kwargs) -> List[np.ndarray]:
        if record_stderr:
            self.stderr_buf = []
        extra_args = kwargs.pop("extra_args", dict())
        extra_args.update(record_stderr=record_stderr)
        return super().acquire(*args, extra_args=extra_args, **kwargs)  # type: ignore

    def _stderr_buf(
        self, d_reps: List[np.ndarray], length_norm: bool = True
    ) -> List[np.ndarray]:
        std_d = []
        for i_ch, (_, ro) in enumerate(self.ro_chs.items()):  # type: ignore
            # std over the avg_level
            std = d_reps[i_ch].std(axis=self.avg_level)
            if length_norm and not ro["edge_counting"]:
                std /= ro["length"]
            # the reads_per_shot axis should be the first one
            std_d.append(np.moveaxis(std, -2, 0))

        return std_d

    def _process_accumulated_for_stderr(self, acc_buf) -> List[np.ndarray]:
        assert self.acquire_params is not None

        if self.acquire_params["threshold"] is None:
            return self._stderr_buf(acc_buf, length_norm=True)
        else:
            d_reps = [np.zeros_like(d) for d in acc_buf]
            self.shots = self._apply_threshold(
                acc_buf,
                self.acquire_params["threshold"],
                self.acquire_params["angle"],
                self.acquire_params["remove_offset"],
            )
            for i, ch_shot in enumerate(self.shots):
                d_reps[i][..., 0] = ch_shot  # type: ignore
            # TODO: stderr on thresholded data is meaningless, but still return it for now
            return self._stderr_buf(d_reps, length_norm=False)

    def finish_round(self) -> bool:
        not_finish = super().finish_round()

        # save the standard error information for the accumulated data
        assert self.acquire_params is not None

        if self.acquire_params["type"] == "accumulated":
            if self.acquire_params["record_stderr"]:
                assert self.stderr_buf is not None
                self.stderr_buf.append(
                    self._process_accumulated_for_stderr(self.acc_buf)
                )
        elif self.acquire_params.get("record_stderr", False):
            # currently not supported for type other than accumulated
            raise NotImplementedError(
                "Standard error is not implemented for type other than accumulated"
            )

        return not_finish


class CallbackMixin(StdErrorMixin):
    """
    Add callback functionality to the AcquireMixin class
    """

    def acquire(
        self, *args, callback: Optional[Callable[..., None]] = None, **kwargs
    ) -> List[np.ndarray]:
        extra_args = kwargs.pop("extra_args", dict())
        extra_args.update(callback=callback)
        return super().acquire(*args, extra_args=extra_args, **kwargs)

    def acquire_decimated(
        self, *args, callback: Optional[Callable[..., None]] = None, **kwargs
    ) -> List[np.ndarray]:
        extra_args = kwargs.pop("extra_args", dict())
        extra_args.update(callback=callback)
        return super().acquire_decimated(*args, extra_args=extra_args, **kwargs)  # type: ignore

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
                std_d = self.get_stderr()
                callback(round_n, avg_d, std_d)
            elif self.acquire_params["type"] == "decimated":
                # currently not supported callback with data
                # dec_d = self._summarize_decimated(self.rounds_buf)
                callback(round_n)
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


class ImproveAcquireMixin(EarlyStopMixin, CallbackMixin, StdErrorMixin):
    pass
