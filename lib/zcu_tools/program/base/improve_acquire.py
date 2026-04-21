from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray
from qick.qick_asm import AcquireMixin
from typing_extensions import Callable, List, Optional, Protocol, TypeAlias, Union, cast

RoundHookType: TypeAlias = Callable[[int, List[NDArray[np.float64]]], None]


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


class EarlyStopMixin(TypedAcquireMixin):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._early_stop = False

    def set_early_stop(self, silent: bool = False) -> None:
        # tell program to return as soon as this round is finished
        if not silent:
            print("Program received early stop signal")
        self._early_stop = True

    def acquire(self, *args, **kwargs):
        self._early_stop = False
        return super().acquire(*args, **kwargs)  # type: ignore

    def acquire_decimated(self, *args, **kwargs):
        self._early_stop = False
        return super().acquire_decimated(*args, **kwargs)  # type: ignore

    def finish_round(self) -> bool:
        not_finish = super().finish_round()
        if not_finish and self._early_stop:
            if self.rounds_pbar is not None:
                self.rounds_pbar.close()
            else:
                warnings.warn(
                    "Early stop signal received but rounds_pbar is not set, cannot close the progress bar"
                )
        return not_finish and not self._early_stop


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


class TrackerProtocol(Protocol):
    def update(self, points: NDArray[np.float64]) -> None:
        """points shape: (*sweep, reps, 2)"""


class TrackerMixin(TypedAcquireMixin):
    """
    Add statistic information for acquired method to the AcquireMixin class
    """

    def finish_round(self) -> bool:
        not_finish = super().finish_round()

        assert self.acc_buf is not None
        assert self.acquire_params is not None

        trackers = self.acquire_params.get("trackers")
        if trackers is not None:
            trackers = cast(List[TrackerProtocol], trackers)
            if self.acquire_params["type"] != "accumulated":
                raise NotImplementedError(
                    "Tracker is not implemented for type other than accumulated"
                )

            if self.acquire_params["threshold"] is not None:
                raise NotImplementedError(
                    "Tracker is not implemented for thresholded data"
                )

            ro_chs: dict = self.ro_chs  # type: ignore

            if len(trackers) != len(self.acc_buf):
                raise ValueError(
                    f"Number of tracker ({len(trackers)}) must match number of readout channels ({len(self.acc_buf)})"
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
        trackers: Optional[List[TrackerProtocol]] = None,
        **kwargs,
    ):
        extra_args = kwargs.pop("extra_args", dict())
        extra_args.update(trackers=trackers)
        return super().acquire(*args, extra_args=extra_args, **kwargs)


class RoundHookMixin(TypedAcquireMixin):
    """
    Add round hook functionality to the AcquireMixin class
    """

    def _reset_inc_summarize(self) -> None:
        self._inc_sum_count: int = 0
        self._inc_sum_state: Optional[List[NDArray[np.float64]]] = None

    def _inc_summarize_accumulated(
        self, rounds_buf: List[NDArray[np.float64]]
    ) -> List[NDArray[np.float64]]:
        self._inc_sum_count += 1
        if len(rounds_buf) != self._inc_sum_count:
            warnings.warn(
                f"Detected non-matching number of rounds ({len(rounds_buf)}) and _inc_sum_count ({self._inc_sum_count})"
                "The callback data may be corrupted, fallback to normal summarize"
            )
            return self._summarize_accumulated(rounds_buf)

        n_ro_chs = len(self.ro_chs)  # type: ignore

        assert self.rounds_buf is not None
        if self._inc_sum_state is None:  # first round
            self._inc_sum_state = cast(
                List[NDArray[np.float64]],
                [
                    np.zeros_like(self.rounds_buf[0][i], dtype=np.float64)
                    for i in range(n_ro_chs)
                ],
            )

        # update the state with the new data
        for i in range(n_ro_chs):
            self._inc_sum_state[i] += rounds_buf[-1][i]

        mean_data = cast(
            List[NDArray[np.float64]],
            [d / self._inc_sum_count for d in self._inc_sum_state],
        )

        return mean_data

    def _inc_summarize_decimated(
        self, rounds_buf: List[NDArray[np.float64]]
    ) -> List[NDArray[np.float64]]:
        # NOTE: currently, summarize decimated is identical to summarize accumulated
        return self._inc_summarize_accumulated(rounds_buf)

    def acquire(self, *args, callback: Optional[RoundHookType] = None, **kwargs):
        extra_args = kwargs.pop("extra_args", dict())
        extra_args.update(callback=callback)

        self._reset_inc_summarize()

        return super().acquire(*args, extra_args=extra_args, **kwargs)

    def acquire_decimated(
        self, *args, callback: Optional[RoundHookType] = None, **kwargs
    ):
        extra_args = kwargs.pop("extra_args", dict())
        extra_args.update(callback=callback)

        self._reset_inc_summarize()

        return super().acquire_decimated(*args, extra_args=extra_args, **kwargs)

    def finish_round(self) -> bool:
        not_finish = super().finish_round()

        acquire_params = self.acquire_params
        assert acquire_params is not None

        # trigger the hook function after each round
        round_hook: Optional[RoundHookType] = acquire_params["callback"]
        if not_finish and round_hook is not None:
            assert callable(round_hook), "round_hook must be a callable function"
            assert self.rounds_buf is not None

            # NOTE: increment the summary to reduce cpu usage
            round_n = len(self.rounds_buf)
            acquire_type = acquire_params["type"]
            if acquire_type == "accumulated":
                # avg_d = self._summarize_accumulated(self.rounds_buf)
                avg_d = self._inc_summarize_accumulated(self.rounds_buf)
                round_hook(round_n, avg_d)
            elif acquire_type == "decimated":
                # dec_d = self._summarize_decimated(self.rounds_buf)
                dec_d = self._inc_summarize_decimated(self.rounds_buf)
                round_hook(round_n, dec_d)
            else:
                raise NotImplementedError(
                    f"Round hook is not implemented for type {acquire_type}"
                )

        return not_finish


class ImproveAcquireMixin(
    SingleShotMixin, RoundHookMixin, EarlyStopMixin, TrackerMixin
): ...
