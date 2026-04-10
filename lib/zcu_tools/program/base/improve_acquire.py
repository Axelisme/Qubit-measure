from __future__ import annotations

import warnings
from abc import ABC, abstractmethod

import numpy as np
from qick.qick_asm import AcquireMixin
from typing_extensions import (
    Callable,
    Generic,
    List,
    Optional,
    TypeAlias,
    TypeVar,
    Union,
    cast,
)

try:
    from numpy.typing import NDArray  # type: ignore
except ImportError:  # for python < 3.9
    T_val = TypeVar("T_val", bound=np.number)

    class NDArray(np.ndarray, Generic[T_val]): ...

CallbackType: TypeAlias = Callable[[int, List[NDArray[np.float64]]], None]


class TypedAcquireMixin(AcquireMixin):
    """
    Add type checking to the AcquireMixin class
    """

    def get_raw(self) -> Optional[List[NDArray[np.int64]]]:
        return super().get_raw()  # type: ignore

    def get_time_axis(
        self, ro_index: int, length_only: bool = False
    ) -> Union[NDArray[np.float64], int]:
        return super().get_time_axis(ro_index, length_only)

    def _summarize_accumulated(
        self, rounds_buf: List[NDArray[np.float64]]
    ) -> List[NDArray[np.float64]]:
        return super()._summarize_accumulated(rounds_buf)

    def _summarize_decimated(
        self, rounds_buf: List[NDArray[np.float64]]
    ) -> List[NDArray[np.float64]]:
        return super()._summarize_decimated(rounds_buf)

    def _average_buf(  # type: ignore
        self,
        d_reps: List[NDArray[np.float64]],
        length_norm: bool = True,
        remove_offset: bool = True,
    ) -> List[NDArray[np.float64]]:
        return super()._average_buf(
            d_reps,  # type: ignore
            length_norm=length_norm,
            remove_offset=remove_offset,
        )

    def _process_accumulated(  # type: ignore
        self, acc_buf: List[NDArray[np.float64]]
    ) -> List[NDArray[np.float64]]:
        return super()._process_accumulated(acc_buf)  # type: ignore

    def acquire(self, *args, **kwargs) -> List[NDArray[np.float64]]:
        return super().acquire(*args, **kwargs)  # type: ignore

    def acquire_decimated(self, *args, **kwargs) -> List[NDArray[np.float64]]:
        return super().acquire_decimated(*args, **kwargs)  # type: ignore


class AbsStatisticTracker(ABC):
    @abstractmethod
    def update(self, points: NDArray[np.float64]) -> None: ...


class StatisticMixin(TypedAcquireMixin):
    """
    Add statistic information for acquired method to the AcquireMixin class
    """

    def finish_round(self) -> bool:
        not_finish = super().finish_round()

        assert self.acc_buf is not None
        assert self.acquire_params is not None

        trackers = self.acquire_params.get("statistic_trackers")
        if trackers is not None:
            trackers = cast(List[AbsStatisticTracker], trackers)
            if self.acquire_params["type"] != "accumulated":
                raise NotImplementedError(
                    "Statistic is not implemented for type other than accumulated"
                )

            if self.acquire_params["threshold"] is not None:
                raise NotImplementedError(
                    "Statistic is not implemented for thresholded data"
                )

            ro_chs: dict = self.ro_chs  # type: ignore

            if len(trackers) != len(self.acc_buf):
                raise ValueError(
                    f"Number of statistic trackers ({len(trackers)}) must match number of readout channels ({len(self.acc_buf)})"
                )

            assert isinstance(ro_chs, dict)
            assert len(self.acc_buf) == len(ro_chs)
            for d_rep, tracker, ro in zip(self.acc_buf, trackers, ro_chs.values()):
                assert self.avg_level is not None
                d_rep = np.moveaxis(d_rep, [-2, self.avg_level], [0, -2])

                if not ro["edge_counting"]:
                    d_rep = d_rep / ro["length"]
                d_rep = cast(NDArray[np.float64], d_rep)

                tracker.update(d_rep)  # (..., m, 2)

        return not_finish

    def acquire(
        self,
        *args,
        statistic_trackers: Optional[List[AbsStatisticTracker]] = None,
        **kwargs,
    ):
        extra_args = kwargs.pop("extra_args", dict())
        extra_args.update(statistic_trackers=statistic_trackers)
        return super().acquire(*args, extra_args=extra_args, **kwargs)


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
            if self.rounds_pbar is not None:
                self.rounds_pbar.close()
            else:
                warnings.warn(
                    "Early stop signal received but rounds_pbar is not set, cannot close the progress bar"
                )
        return not_finish and not self.early_stop


class CallbackMixin(StatisticMixin):
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
                callback(round_n, avg_d)
            elif self.acquire_params["type"] == "decimated":
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

    def _process_accumulated(self, acc_buf) -> List[NDArray[np.float64]]:
        assert self.acquire_params is not None

        threshold: Optional[float] = self.acquire_params.get("threshold")
        angle: Optional[float] = self.acquire_params.get("angle")

        ge_radius: Optional[float] = self.acquire_params.get("population_radius")
        g_center: Optional[complex] = self.acquire_params.get("g_center")
        e_center: Optional[complex] = self.acquire_params.get("e_center")

        remove_offset: bool = self.acquire_params["remove_offset"]

        if threshold is not None:
            d_reps = [np.zeros_like(d) for d in acc_buf]
            if ge_radius is not None:
                warnings.warn(
                    "Both threshold and population_radius are set, threshold will be applied and population_radius will be ignored"
                )
            self.shots = self._apply_threshold(acc_buf, threshold, angle, remove_offset)
            for i, ch_shot in enumerate(self.shots):
                d_reps[i][..., 0] = ch_shot
            return self._average_buf(d_reps, length_norm=False)
        elif ge_radius is not None:
            d_reps = [np.zeros_like(d) for d in acc_buf]
            if g_center is None or e_center is None:
                raise ValueError(
                    "g_center and e_center must be provided when population_radius is set"
                )
            self.shots = self._apply_classification(
                acc_buf, g_center, e_center, ge_radius, remove_offset
            )
            for i, ch_shot in enumerate(self.shots):
                d_reps[i] = ch_shot
            return self._average_buf(d_reps, length_norm=False)
        else:
            d_reps = acc_buf
            return self._average_buf(
                d_reps,
                length_norm=True,
                remove_offset=self.acquire_params["remove_offset"],
            )

    def _apply_classification(
        self,
        acc_buf: List[NDArray[np.float64]],
        g_center: complex,
        e_center: complex,
        population_radius: float,
        remove_offset: bool,
    ) -> List[NDArray[np.float64]]:
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
    SingleShotMixin, CallbackMixin, EarlyStopMixin, StatisticMixin
): ...
