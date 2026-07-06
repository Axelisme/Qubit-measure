from __future__ import annotations

import functools
import operator
import warnings
from collections.abc import Callable
from typing import Protocol, TypeAlias, cast, final

import numpy as np
from numpy.typing import NDArray
from qick.qick_asm import AcquireMixin, logger, obtain, tqdm


class CancelFlagProtocol(Protocol):
    def is_set(self) -> bool: ...

    def set(self) -> None: ...


class _LocalCancelFlag:
    def __init__(self) -> None:
        self._is_set = False

    def is_set(self) -> bool:
        return self._is_set

    def set(self) -> None:
        self._is_set = True


RoundHookType: TypeAlias = Callable[
    [int, list[NDArray[np.float64]], CancelFlagProtocol], None
]


class StoppedPartialAcquireError(RuntimeError):
    """Raised when stop happens before the first round produces completed data."""


class TypedAcquireMixin(AcquireMixin):
    """
    Add type checking to the AcquireMixin class
    """

    def get_raw(self) -> list[NDArray[np.int64]] | None:
        return super().get_raw()  # type: ignore

    def get_time_axis(
        self, ro_index: int, length_only: bool = False
    ) -> NDArray[np.float64] | int:
        return super().get_time_axis(ro_index, length_only)

    def _summarize_accumulated(
        self, rounds_buf: list[NDArray[np.float64]]
    ) -> list[NDArray[np.float64]]:
        return super()._summarize_accumulated(rounds_buf)

    def _summarize_decimated(
        self, rounds_buf: list[NDArray[np.float64]]
    ) -> list[NDArray[np.float64]]:
        return super()._summarize_decimated(rounds_buf)

    def _average_buf(  # type: ignore
        self,
        d_reps: list[NDArray[np.float64]],
        length_norm: bool = True,
        remove_offset: bool = True,
    ) -> list[NDArray[np.float64]]:
        return super()._average_buf(
            d_reps,  # type: ignore
            length_norm=length_norm,
            remove_offset=remove_offset,
        )

    def _process_accumulated(  # type: ignore
        self, acc_buf: list[NDArray[np.float64]]
    ) -> list[NDArray[np.float64]]:
        return super()._process_accumulated(acc_buf)  # type: ignore

    def acquire(self, *args, **kwargs) -> list[NDArray[np.float64]]:
        return super().acquire(*args, **kwargs)  # type: ignore

    def acquire_decimated(self, *args, **kwargs) -> list[NDArray[np.float64]]:
        return super().acquire_decimated(*args, **kwargs)  # type: ignore

    def _completed_round_count(self) -> int:
        rounds_buf = self.rounds_buf
        if rounds_buf is None:
            return 0
        return len(rounds_buf)


class EarlyStopMixin(TypedAcquireMixin):
    def acquire(self, *args, cancel_flag: CancelFlagProtocol | None = None, **kwargs):
        extra_args = kwargs.pop("extra_args", dict())
        extra_args.update(
            cancel_flag=cancel_flag if cancel_flag is not None else _LocalCancelFlag()
        )
        return super().acquire(*args, extra_args=extra_args, **kwargs)  # type: ignore

    def acquire_decimated(
        self, *args, cancel_flag: CancelFlagProtocol | None = None, **kwargs
    ):
        extra_args = kwargs.pop("extra_args", dict())
        extra_args.update(
            cancel_flag=cancel_flag if cancel_flag is not None else _LocalCancelFlag()
        )
        return super().acquire_decimated(*args, extra_args=extra_args, **kwargs)  # type: ignore

    def finish_round(self) -> bool:
        assert self.acquire_params is not None
        cancel_flag = cast(
            CancelFlagProtocol | None, self.acquire_params.get("cancel_flag")
        )
        if cancel_flag is not None and cancel_flag.is_set():
            self._cleanup_prepared_round()
            self._close_rounds_pbar()
            return False

        if self.acquire_params["type"] == "accumulated" and cancel_flag is not None:
            return self._finish_accumulated_round(cancel_flag)

        not_finish = super().finish_round()
        if not_finish and cancel_flag is not None and cancel_flag.is_set():
            self._close_rounds_pbar()
            return False
        return not_finish

    def finish_acquire(self):
        if self._is_stopped_before_first_completed_round():
            self._close_rounds_pbar()
            raise StoppedPartialAcquireError(
                "acquire stopped before the first round completed"
            )
        return super().finish_acquire()

    def _finish_accumulated_round(self, cancel_flag: CancelFlagProtocol) -> bool:
        assert self.acquire_params is not None
        soc = self.acquire_params["soc"]
        total_count = functools.reduce(operator.mul, self.loop_dims)  # type: ignore[arg-type]
        reads_per_shot = [ro["trigs"] for ro in self.ro_chs.values()]  # type: ignore[union-attr]

        count = 0
        stats_start = len(self.stats) if self.stats is not None else 0
        with tqdm(
            total=total_count, disable=self.acquire_params["hidereps"]
        ) as reps_pbar:
            soc.start_readout(
                total_count,
                counter_addr=self.counter_addr,
                ch_list=list(self.ro_chs),  # type: ignore[arg-type]
                reads_per_shot=reads_per_shot,
            )
            while count < total_count:
                if cancel_flag.is_set():
                    return self._finish_stopped_partial_accumulated_round(
                        soc, stats_start=stats_start
                    )

                new_data = obtain(soc.poll_data())
                for new_points, (data, stats) in new_data:
                    if cancel_flag.is_set():
                        return self._finish_stopped_partial_accumulated_round(
                            soc, stats_start=stats_start
                        )
                    for ii, nreads in enumerate(reads_per_shot):
                        if new_points * nreads != data[ii].shape[0]:
                            logger.error(
                                "data size mismatch: new_points=%d, nreads=%d, data shape %s"
                                % (new_points, nreads, data[ii].shape)
                            )
                        if count + new_points > total_count:
                            logger.error(
                                "got too much data: count=%d, new_points=%d, total_count=%d"
                                % (count, new_points, total_count)
                            )
                        self.acc_buf[ii].reshape((-1, 2))[  # type: ignore[index]
                            count * nreads : (count + new_points) * nreads
                        ] = data[ii]
                    count += new_points
                    self.stats.append(stats)  # type: ignore[attr-defined]
                    reps_pbar.update(new_points)

                if count < total_count and cancel_flag.is_set():
                    return self._finish_stopped_partial_accumulated_round(
                        soc, stats_start=stats_start
                    )

        if cancel_flag.is_set():
            return self._finish_stopped_partial_accumulated_round(
                soc, stats_start=stats_start
            )

        assert self.rounds_buf is not None
        assert self.acc_buf is not None
        self.rounds_buf.append(self._process_accumulated(self.acc_buf))

        soc.cleanup_round()
        if self.rounds_pbar is not None:
            self.rounds_pbar.update()
        self.acquire_params["rounds_remaining"] -= 1
        done = self.acquire_params["rounds_remaining"] <= 0
        if done:
            self._close_rounds_pbar()
        elif cancel_flag.is_set():
            self._close_rounds_pbar()
            return False
        return not done

    def _finish_stopped_partial_accumulated_round(
        self, soc, *, stats_start: int
    ) -> bool:
        self._stop_started_accumulated_round(soc)
        self._discard_partial_accumulated_round()
        if self.stats is not None:
            del self.stats[stats_start:]
        soc.cleanup_round()
        self._close_rounds_pbar()
        return False

    def _stop_started_accumulated_round(self, soc) -> None:
        stop_tproc = getattr(soc, "stop_tproc", None)
        if callable(stop_tproc):
            stop_tproc()

        streamer = getattr(soc, "streamer", None)
        stop_readout = getattr(streamer, "stop_readout", None)
        if callable(stop_readout):
            stop_readout()

    def _cleanup_prepared_round(self) -> None:
        assert self.acquire_params is not None
        soc = self.acquire_params["soc"]
        cleanup = getattr(soc, "cleanup_round", None)
        if callable(cleanup):
            cleanup()

    def _discard_partial_accumulated_round(self) -> None:
        if self.acc_buf is None:
            return
        for buffer in self.acc_buf:
            np.copyto(buffer, 0)

    def _close_rounds_pbar(self) -> None:
        if self.rounds_pbar is not None:
            self.rounds_pbar.close()

    def _is_stopped_before_first_completed_round(self) -> bool:
        if self.acquire_params is None:
            return False
        cancel_flag = cast(
            CancelFlagProtocol | None, self.acquire_params.get("cancel_flag")
        )
        return (
            cancel_flag is not None
            and cancel_flag.is_set()
            and self._completed_round_count() == 0
        )


class SingleShotMixin(TypedAcquireMixin):
    def acquire(
        self,
        *args,
        g_center: complex | None = None,
        e_center: complex | None = None,
        ge_radius: float | None = None,
        **kwargs,
    ):
        extra_args = kwargs.pop("extra_args", dict())
        extra_args.update(g_center=g_center, e_center=e_center, ge_radius=ge_radius)

        return super().acquire(*args, extra_args=extra_args, **kwargs)

    def acquire_decimated(
        self,
        *args,
        g_center: complex | None = None,
        e_center: complex | None = None,
        ge_radius: float | None = None,
        **kwargs,
    ):
        extra_args = kwargs.pop("extra_args", dict())
        extra_args.update(g_center=g_center, e_center=e_center, ge_radius=ge_radius)

        return super().acquire_decimated(*args, extra_args=extra_args, **kwargs)

    @final
    def _process_accumulated(self, acc_buf) -> list[NDArray[np.float64]]:
        assert self.acquire_params is not None

        threshold: float | None = self.acquire_params.get("threshold")
        angle: float | None = self.acquire_params.get("angle")

        g_center: complex | None = self.acquire_params.get("g_center")
        e_center: complex | None = self.acquire_params.get("e_center")
        ge_radius: float | None = self.acquire_params.get("ge_radius")

        remove_offset: bool = self.acquire_params["remove_offset"]

        if threshold is not None:
            d_reps = [np.zeros_like(d) for d in acc_buf]
            if ge_radius is not None:
                warnings.warn(
                    "Both threshold and ge_radius are set, threshold will be applied and ge_radius will be ignored"
                )
            self.shots = self._apply_threshold(acc_buf, threshold, angle, remove_offset)
            for i, ch_shot in enumerate(self.shots):
                d_reps[i][..., 0] = ch_shot
            return self._average_buf(d_reps, length_norm=False)
        elif ge_radius is not None:
            d_reps = [np.zeros_like(d) for d in acc_buf]
            if g_center is None or e_center is None:
                raise ValueError(
                    "g_center and e_center must be provided when ge_radius is set"
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
        acc_buf: list[NDArray[np.float64]],
        g_center: complex,
        e_center: complex,
        ge_radius: float,
        remove_offset: bool,
    ) -> list[NDArray[np.float64]]:
        shots = []
        for i_ch, (ro_ch, ro) in enumerate(self.ro_chs.items()):  # type: ignore
            avg = acc_buf[i_ch] / ro["length"]
            if remove_offset:
                offset = self.soccfg["readouts"][ro_ch]["iq_offset"]  # type: ignore
                avg -= offset
            g_dist = np.abs(avg.dot([1, 1j]) - g_center)
            e_dist = np.abs(avg.dot([1, 1j]) - e_center)
            g_shot = np.heaviside(ge_radius - g_dist, 0)
            e_shot = np.heaviside(ge_radius - e_dist, 0)
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
        completed_rounds = self._completed_round_count()
        not_finish = super().finish_round()
        if self._completed_round_count() == completed_rounds:
            return not_finish

        assert self.acc_buf is not None
        assert self.acquire_params is not None

        trackers = cast(
            list[TrackerProtocol] | None, self.acquire_params.get("trackers")
        )
        if trackers is not None:
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
        trackers: list[TrackerProtocol] | None = None,
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
        self._inc_sum_state: list[NDArray[np.float64]] | None = None

    def _inc_summarize_accumulated(
        self, rounds_buf: list[NDArray[np.float64]]
    ) -> list[NDArray[np.float64]]:
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
                list[NDArray[np.float64]],
                [
                    np.zeros_like(self.rounds_buf[0][i], dtype=np.float64)
                    for i in range(n_ro_chs)
                ],
            )

        # update the state with the new data
        for i in range(n_ro_chs):
            self._inc_sum_state[i] += rounds_buf[-1][i]

        mean_data = cast(
            list[NDArray[np.float64]],
            [d / self._inc_sum_count for d in self._inc_sum_state],
        )

        return mean_data

    def _inc_summarize_decimated(
        self, rounds_buf: list[NDArray[np.float64]]
    ) -> list[NDArray[np.float64]]:
        # NOTE: currently, summarize decimated is identical to summarize accumulated
        return self._inc_summarize_accumulated(rounds_buf)

    def acquire(self, *args, round_hook: RoundHookType | None = None, **kwargs):
        extra_args = kwargs.pop("extra_args", dict())
        extra_args.update(round_hook=round_hook)

        self._reset_inc_summarize()

        return super().acquire(*args, extra_args=extra_args, **kwargs)

    def acquire_decimated(
        self, *args, round_hook: RoundHookType | None = None, **kwargs
    ):
        extra_args = kwargs.pop("extra_args", dict())
        extra_args.update(round_hook=round_hook)

        self._reset_inc_summarize()

        return super().acquire_decimated(*args, extra_args=extra_args, **kwargs)

    def finish_round(self) -> bool:
        completed_rounds = self._completed_round_count()
        not_finish = super().finish_round()
        if self._completed_round_count() == completed_rounds:
            return not_finish

        acquire_params = self.acquire_params
        assert acquire_params is not None

        # trigger the hook function after each round
        round_hook = cast(RoundHookType | None, acquire_params.get("round_hook"))
        if round_hook is not None:
            assert callable(round_hook), "round_hook must be a callable function"
            assert self.rounds_buf is not None

            # NOTE: increment the summary to reduce cpu usage
            round_count = len(self.rounds_buf)
            acquire_type = acquire_params["type"]
            if acquire_type == "accumulated":
                # avg_d = self._summarize_accumulated(self.rounds_buf)
                avg_d = self._inc_summarize_accumulated(self.rounds_buf)
                round_hook(round_count, avg_d, self._round_hook_cancel_flag())
            elif acquire_type == "decimated":
                # dec_d = self._summarize_decimated(self.rounds_buf)
                dec_d = self._inc_summarize_decimated(self.rounds_buf)
                round_hook(round_count, dec_d, self._round_hook_cancel_flag())
            else:
                raise NotImplementedError(
                    f"Round hook is not implemented for type {acquire_type}"
                )

        cancel_flag = self._round_hook_cancel_flag()
        if cancel_flag.is_set():
            if self.rounds_pbar is not None:
                self.rounds_pbar.close()
            return False
        return not_finish

    def _round_hook_cancel_flag(self) -> CancelFlagProtocol:
        acquire_params = self.acquire_params
        assert acquire_params is not None
        cancel_flag = cast(CancelFlagProtocol | None, acquire_params.get("cancel_flag"))
        if cancel_flag is None:
            raise RuntimeError("round_hook requires acquire cancel_flag")
        return cancel_flag


class ImproveAcquireMixin(
    RoundHookMixin, TrackerMixin, SingleShotMixin, EarlyStopMixin
): ...
