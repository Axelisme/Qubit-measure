from typing import Callable, List, Optional, Tuple, Union, cast

import numpy as np
from numpy.typing import NDArray
from qick.qick_asm import AcquireMixin
from typing_extensions import TypeAlias


class TypedAcquireMixin(AcquireMixin):
    """
    Add type checking to the AcquireMixin class
    """

    def get_raw(self) -> Optional[List[NDArray[np.int64]]]:
        return super().get_raw()

    def get_time_axis(
        self, ro_index: int, length_only: bool = False
    ) -> Union[NDArray[np.float64], int]:
        return super().get_time_axis(ro_index, length_only)

    def acquire(self, *args, **kwargs) -> List[NDArray[np.float64]]:
        return super().acquire(*args, **kwargs)  # type: ignore

    def acquire_decimated(self, *args, **kwargs) -> List[NDArray[np.float64]]:
        return super().acquire_decimated(*args, **kwargs)  # type: ignore


class StdErrorMixin(TypedAcquireMixin):
    """
    Add standard error information for acquired method to the AcquireMixin class
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.stderr_buf: Optional[List[List[NDArray]]] = None

    def get_stderr_raw(self) -> Optional[List[List[NDArray[np.float64]]]]:
        return self.stderr_buf

    def get_stderr(self) -> Optional[List[NDArray[np.float64]]]:
        rounds_buf = self.get_rounds()
        stderr_buf = self.get_stderr_raw()
        if stderr_buf is None:
            return None

        assert rounds_buf is not None
        return self._summarize_accumulated_std(rounds_buf, stderr_buf)

    def _summarize_accumulated_std(
        self,
        rounds_buf: List[List[NDArray[np.float64]]],
        std_buf: List[List[NDArray[np.float64]]],
    ) -> List[NDArray[np.float64]]:
        return [
            np.sqrt(
                np.mean(
                    [u[i] ** 2 + s[i] ** 2 for u, s in zip(rounds_buf, std_buf)], axis=0
                )
                - np.mean([u[i] for u in rounds_buf], axis=0) ** 2
            )
            for i in range(len(self.ro_chs))  # type: ignore
        ]

    def _stderr_buf(
        self, d_reps: List[NDArray[np.int64]], length_norm: bool = True
    ) -> List[NDArray[np.float64]]:
        std_d = []
        for i_ch, (_, ro) in enumerate(self.ro_chs.items()):  # type: ignore
            # std over the avg_level
            std = d_reps[i_ch].std(axis=self.avg_level)
            if length_norm and not ro["edge_counting"]:
                std /= ro["length"]
            # the reads_per_shot axis should be the first one
            std_d.append(np.moveaxis(std, -2, 0))

        return std_d

    def _process_accumulated_for_stderr(
        self, acc_buf: List[NDArray[np.int64]]
    ) -> List[NDArray[np.float64]]:
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
                assert self.acc_buf is not None
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

    def acquire(self, *args, record_stderr=False, **kwargs):
        if record_stderr:
            self.stderr_buf = []
        extra_args = kwargs.pop("extra_args", dict())
        extra_args.update(record_stderr=record_stderr)
        return super().acquire(*args, extra_args=extra_args, **kwargs)  # type: ignore


class EarlyStopMixin(TypedAcquireMixin):
    """
    Add early stopping functionality to the AcquireMixin class
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.early_stop = False

    def set_early_stop(self, silent: bool = False) -> None:
        # tell program to return as soon as this round is finished
        if not silent:
            print("Program received early stop signal")
        self.early_stop = True

    def acquire(self, *args, **kwargs):
        self.early_stop = False
        return super().acquire(*args, **kwargs)  # type: ignore

    def acquire_decimated(self, *args, **kwargs):
        self.early_stop = False
        return super().acquire_decimated(*args, **kwargs)  # type: ignore

    def finish_round(self) -> bool:
        not_finish = super().finish_round()
        if not_finish and self.early_stop:
            assert self.rounds_pbar is not None
            self.rounds_pbar.close()
        return not_finish and not self.early_stop


BaseCallbackType: TypeAlias = Callable[[int, List[NDArray]], None]
StdCallbackType: TypeAlias = Callable[[int, Tuple[List[NDArray], List[NDArray]]], None]
CallbackType: TypeAlias = Union[BaseCallbackType, StdCallbackType]


class CallbackMixin(StdErrorMixin):
    """
    Add callback functionality to the AcquireMixin class
    """

    def acquire(self, *args, callback: Optional[CallbackType] = None, **kwargs):
        extra_args = kwargs.pop("extra_args", dict())
        extra_args.update(callback=callback)

        return super().acquire(*args, extra_args=extra_args, **kwargs)

    def acquire_decimated(
        self, *args, callback: Optional[CallbackType] = None, **kwargs
    ):
        extra_args = kwargs.pop("extra_args", dict())
        extra_args.update(callback=callback)

        return super().acquire_decimated(*args, extra_args=extra_args, **kwargs)

    def finish_round(self) -> bool:
        not_finish = super().finish_round()

        # trigger the callback function after each round
        assert self.acquire_params is not None
        callback: Optional[CallbackType] = self.acquire_params["callback"]
        if callback is not None:
            assert callable(callback), "callback must be a callable function"
            assert self.rounds_buf is not None

            round_n = len(self.rounds_buf)
            if self.acquire_params["type"] == "accumulated":
                avg_d = self._summarize_accumulated(self.rounds_buf)
                if self.acquire_params.get("record_stderr", False):
                    callback = cast(StdCallbackType, callback)

                    std_d = self.get_stderr()
                    assert std_d is not None

                    callback(round_n, (avg_d, std_d))
                else:
                    callback = cast(BaseCallbackType, callback)

                    callback(round_n, avg_d)
            elif self.acquire_params["type"] == "decimated":
                callback = cast(BaseCallbackType, callback)

                dec_d = self._summarize_decimated(self.rounds_buf)
                callback(round_n, dec_d)
            else:
                raise NotImplementedError(
                    "Callback is not implemented for type other than accumulated or decimated"
                )

        return not_finish


class SingleShotMixin(TypedAcquireMixin):
    def acquire(
        self,
        *args,
        g_center: Optional[complex] = None,
        e_center: Optional[complex] = None,
        population_radius: Optional[float] = None,
        **kwargs,
    ):
        extra_args = kwargs.pop("extra_args", dict())
        extra_args.update(
            g_center=g_center,
            e_center=e_center,
            population_radius=population_radius,
        )

        return super().acquire(*args, extra_args=extra_args, **kwargs)

    def acquire_decimated(
        self,
        *args,
        g_center: Optional[complex] = None,
        e_center: Optional[complex] = None,
        population_radius: Optional[float] = None,
        **kwargs,
    ):
        extra_args = kwargs.pop("extra_args", dict())
        extra_args.update(
            g_center=g_center,
            e_center=e_center,
            population_radius=population_radius,
        )

        return super().acquire_decimated(*args, extra_args=extra_args, **kwargs)

    def _process_accumulated(self, acc_buf):
        assert self.acquire_params is not None
        if self.acquire_params["threshold"] is not None:
            d_reps = [np.zeros_like(d) for d in acc_buf]
            self.shots = self._apply_threshold(
                acc_buf,
                self.acquire_params["threshold"],
                self.acquire_params["angle"],
                self.acquire_params["remove_offset"],
            )
            for i, ch_shot in enumerate(self.shots):
                d_reps[i][..., 0] = ch_shot
            return self._average_buf(d_reps, length_norm=False)  # type: ignore
        elif self.acquire_params["population_radius"] is not None:
            d_reps = [np.zeros_like(d) for d in acc_buf]
            self.shots = self._apply_classification(
                acc_buf,
                self.acquire_params["g_center"],
                self.acquire_params["e_center"],
                self.acquire_params["population_radius"],
                self.acquire_params["remove_offset"],
            )
            for i, ch_shot in enumerate(self.shots):
                d_reps[i] = ch_shot
            return self._average_buf(d_reps, length_norm=False)  # type: ignore
        else:
            d_reps = acc_buf
            return self._average_buf(
                d_reps,
                length_norm=True,
                remove_offset=self.acquire_params["remove_offset"],
            )

    def _apply_classification(
        self,
        acc_buf,
        g_center: complex,
        e_center: complex,
        population_radius: float,
        remove_offset: bool,
    ):
        shots = []
        for i_ch, (ro_ch, ro) in enumerate(self.ro_chs.items()):  # type: ignore
            avg = acc_buf[i_ch] / ro["length"]
            if remove_offset:
                offset = self.soccfg["readouts"][ro_ch]["iq_offset"]  # type: ignore
                avg -= offset
            g_dist = np.abs(avg.dot([1, 1j]) - g_center)
            e_dist = np.abs(avg.dot([1, 1j]) - e_center)
            g_shot = np.heaviside(population_radius - g_dist, 0)
            e_shot = np.heaviside(population_radius - e_dist, 0)
            shots.append(np.stack([g_shot, e_shot], axis=-1))
        return shots


class ImproveAcquireMixin(
    SingleShotMixin, CallbackMixin, EarlyStopMixin, StdErrorMixin
): ...
