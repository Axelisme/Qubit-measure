from __future__ import annotations

import functools
import operator
import os
import warnings
from collections.abc import Callable, Sequence
from typing import Literal, Protocol, TypeAlias, cast, final, overload

import numpy as np
from numpy.typing import NDArray
from qick import qick_asm
from qick.qick_asm import (
    AcquireMixin,
    logger,
    # tolerated: QICK imports obtain dynamically from optional RPyC or defines its local fallback.
    obtain,  # pyright: ignore[reportAttributeAccessIssue]
)

from zcu_tools.progress_bar import make_pbar
from zcu_tools.utils.shot_classification import classify_shots

__all__ = [
    "ImproveAcquireMixin",
    "CancelFlagProtocol",
    "StoppedPartialAcquireError",
    "AcquisitionContractError",
    "TrackerProtocol",
]


# Skip both program wrappers and QICK's loop: acquire and stepped finish_round
# enter through different depths, as do accumulated and decimated round hooks.
_ACQUIRE_WARNING_INTERNAL_PATHS = (
    os.path.dirname(__file__) + os.sep,
    os.path.dirname(qick_asm.__file__) + os.sep,
)


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


class AcquisitionContractError(RuntimeError):
    """The transport returned more shots than the acquisition requested.

    The offending chunk is not written or truncated. Earlier raw chunks remain
    available through get_raw(); the incomplete round is not published.
    """


class StoppedPartialAcquireError(RuntimeError):
    """Raised when stop happens before the first round produces completed data."""


class TypedAcquireMixin(AcquireMixin):
    """
    Add type checking to the AcquireMixin class
    """

    def get_raw(self) -> list[NDArray[np.int64]] | None:
        return super().get_raw()

    def get_time_axis(
        self,
        ro_index: int,
        # tolerated: QICK's get_time_axis accepts the length-only selector positionally.
        length_only: bool = False,  # noqa: FBT001, FBT002
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

    # tolerated: QICK annotates acquisition buffers as ndarray although its implementations consume and return per-channel lists.
    def _average_buf(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        d_reps: Sequence[NDArray[np.float64] | NDArray[np.int64]],
        # tolerated: QICK's averaging override accepts positional readout-length normalization.
        length_norm: bool = True,  # noqa: FBT001, FBT002
        # tolerated: QICK's averaging override accepts positional IQ-offset subtraction.
        remove_offset: bool = True,  # noqa: FBT001, FBT002
    ) -> list[NDArray[np.float64]]:
        return super()._average_buf(
            # tolerated: QICK annotates acquisition buffers as ndarray although its implementations consume and return per-channel lists.
            d_reps,  # pyright: ignore[reportArgumentType, reportReturnType]
            length_norm=length_norm,
            remove_offset=remove_offset,
        )

    # tolerated: QICK annotates acquisition buffers as ndarray although its implementations consume and return per-channel lists.
    def _process_accumulated(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, acc_buf: Sequence[NDArray[np.float64] | NDArray[np.int64]]
    ) -> list[NDArray[np.float64]]:
        # tolerated: QICK annotates acquisition buffers as ndarray although its implementations consume and return per-channel lists.
        return super()._process_accumulated(acc_buf)  # pyright: ignore[reportReturnType]

    @overload
    def acquire(
        self, soc: object, *, step_rounds: Literal[False] = False, **kwargs: object
    ) -> list[NDArray[np.float64]]: ...

    @overload
    def acquire(
        self, soc: object, *, step_rounds: Literal[True], **kwargs: object
    ) -> None: ...

    @overload
    def acquire(
        self, soc: object, *args, **kwargs
    ) -> list[NDArray[np.float64]] | None: ...

    def acquire(self, *args, **kwargs) -> list[NDArray[np.float64]] | None:
        """Return averaged IQ for omitted/False step_rounds; True prepares only.

        Preparation returns None. Mode typing is precise with only soc
        positional; extra positional options or a dynamic bool retain the union.
        QICK argument/transport errors propagate. Cancellation retains completed
        rounds, or raises StoppedPartialAcquireError if none completed.
        """
        return self._acquire_configured(*args, **kwargs)

    def _acquire_configured(self, *args, **kwargs) -> list[NDArray[np.float64]] | None:
        # Environment adapters can replace buffer allocation after modifiers
        # have supplied their options through the normal acquire chain.
        return super().acquire(*args, **kwargs)

    @overload
    def acquire_decimated(
        self, soc: object, *, step_rounds: Literal[False] = False, **kwargs: object
    ) -> list[NDArray[np.float64]]: ...

    @overload
    def acquire_decimated(
        self, soc: object, *, step_rounds: Literal[True], **kwargs: object
    ) -> None: ...

    @overload
    def acquire_decimated(
        self, soc: object, *args, **kwargs
    ) -> list[NDArray[np.float64]] | None: ...

    def acquire_decimated(self, *args, **kwargs) -> list[NDArray[np.float64]] | None:
        """Return decimated IQ for omitted/False step_rounds; True prepares only.

        Preparation returns None. Mode typing is precise with only soc
        positional; extra positional options or a dynamic bool retain the union.
        QICK argument/transport errors propagate. Cancellation retains completed
        rounds, or raises StoppedPartialAcquireError if none completed.
        """
        return super().acquire_decimated(*args, **kwargs)

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
        return super().acquire(*args, extra_args=extra_args, **kwargs)

    def acquire_decimated(
        self, *args, cancel_flag: CancelFlagProtocol | None = None, **kwargs
    ):
        extra_args = kwargs.pop("extra_args", dict())
        extra_args.update(
            cancel_flag=cancel_flag if cancel_flag is not None else _LocalCancelFlag()
        )
        return super().acquire_decimated(*args, extra_args=extra_args, **kwargs)

    def finish_round(self) -> bool:
        assert self.acquire_params is not None
        # tolerated(cast): QICK merges the typed acquire cancel_flag into its untyped, heterogeneous acquire_params mapping.
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
        # Mirrors the "accumulated" branch of qick's AcquireMixin.finish_round
        # (qick 0.2.436, qick/qick_asm.py) with cancel_flag checks injected.
        # qick's loop has no early-exit seam (its `while` only checks
        # `count < total_count`), so the copy is deliberate. On QICK upgrades,
        # manually compare AcquireMixin.finish_round and port relevant changes.
        # Validate acquisition and early-stop behavior with existing public-interface tests.
        assert self.acquire_params is not None
        soc = self.acquire_params["soc"]
        assert isinstance(self, qick_asm.AbsQickProgram)
        assert self.loop_dims is not None
        assert self.acc_buf is not None
        total_count = functools.reduce(operator.mul, self.loop_dims)
        reads_per_shot = [ro["trigs"] for ro in self.ro_chs.values()]

        count = 0
        # tolerated: QICK infers stats from its [] initialization; preserve optional-stats handling in this mirrored round loop.
        stats_start = len(self.stats) if self.stats is not None else 0  # pyright: ignore[reportUnnecessaryComparison]
        reps_pbar = make_pbar(
            total=total_count,
            disable=self.acquire_params["hidereps"],
            leave=False,
        )
        try:
            # tolerated: QICK mixes the original soc transport with scalar options in acquire_params without a per-key type.
            soc.start_readout(  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]
                total_count,
                counter_addr=self.counter_addr,
                ch_list=list(self.ro_chs),
                reads_per_shot=reads_per_shot,
            )
            while count < total_count:
                if cancel_flag.is_set():
                    return self._finish_stopped_partial_accumulated_round(
                        soc, stats_start=stats_start
                    )

                # tolerated: QICK's heterogeneous acquire_params loses the original soc transport's poll_data method.
                new_data = obtain(soc.poll_data())  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]
                for new_points, (data, stats) in new_data:
                    if cancel_flag.is_set():
                        return self._finish_stopped_partial_accumulated_round(
                            soc, stats_start=stats_start
                        )
                    self._store_accumulated_chunk(
                        reads_per_shot, data, count, new_points, total_count
                    )
                    count += new_points
                    self.stats.append(stats)
                    reps_pbar.update(new_points)

                if count < total_count and cancel_flag.is_set():
                    return self._finish_stopped_partial_accumulated_round(
                        soc, stats_start=stats_start
                    )
        finally:
            reps_pbar.close()

        if cancel_flag.is_set():
            return self._finish_stopped_partial_accumulated_round(
                soc, stats_start=stats_start
            )

        assert self.rounds_buf is not None
        assert self.acc_buf is not None
        self.rounds_buf.append(self._process_accumulated(self.acc_buf))

        # tolerated: QICK's heterogeneous acquire_params loses the original soc transport's cleanup_round method.
        soc.cleanup_round()  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]
        if self.rounds_pbar is not None:
            self.rounds_pbar.update()
        # tolerated: QICK stores the integer round counter alongside transport and string options without a per-key type.
        self.acquire_params["rounds_remaining"] -= 1  # pyright: ignore[reportOperatorIssue]
        done = self.acquire_params["rounds_remaining"] <= 0
        if done or cancel_flag.is_set():
            self._close_rounds_pbar()
            return False
        return True

    def _store_accumulated_chunk(
        self,
        reads_per_shot: Sequence[int],
        data: Sequence[NDArray[np.int64]],
        count: int,
        new_points: int,
        total_count: int,
    ) -> None:
        """Reject excess shots before writing any channel at its shot offset.

        Within the requested shot count, preserve QICK's channel order and
        NumPy broadcasting; row-count mismatches still produce diagnostics.
        """
        assert self.acc_buf is not None
        if count + new_points > total_count:
            raise AcquisitionContractError(
                "Accumulated acquisition shot overrun: "
                f"expected_shots={total_count}, received_shots={count}, "
                f"chunk_shots={new_points}, chunk_rows={[len(rows) for rows in data]}, "
                f"round_index={self._completed_round_count()}"
            )
        for ii, nreads in enumerate(reads_per_shot):
            if new_points * nreads != data[ii].shape[0]:
                logger.error(
                    "data size mismatch: new_points=%d, nreads=%d, data shape %s"
                    % (new_points, nreads, data[ii].shape)
                )
            self.acc_buf[ii].reshape((-1, 2))[
                count * nreads : (count + new_points) * nreads
            ] = data[ii]

    def _finish_stopped_partial_accumulated_round(
        self, soc, *, stats_start: int
    ) -> bool:
        self._stop_started_accumulated_round(soc)
        self._discard_partial_accumulated_round()
        # tolerated: QICK infers stats from its [] initialization; preserve optional-stats handling when discarding a stopped round.
        if self.stats is not None:  # pyright: ignore[reportUnnecessaryComparison]
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
        # tolerated(cast): finish_acquire reads the same QICK-owned flag entry to distinguish no completed data from a valid partial result.
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

        # tolerated: QICK documents scalar or per-channel thresholds, but acquire_params also contains unrelated string options.
        threshold: float | list[float] | None = self.acquire_params.get("threshold")  # pyright: ignore[reportAssignmentType]
        # tolerated: QICK documents scalar or per-channel angles, but acquire_params has no per-key value types.
        angle: float | list[float] | None = self.acquire_params.get("angle")  # pyright: ignore[reportAssignmentType]

        # tolerated: QICK merges the typed acquire g_center option into its heterogeneous acquire_params mapping.
        g_center: complex | None = self.acquire_params.get("g_center")  # pyright: ignore[reportAssignmentType]
        # tolerated: QICK merges the typed acquire e_center option into its heterogeneous acquire_params mapping.
        e_center: complex | None = self.acquire_params.get("e_center")  # pyright: ignore[reportAssignmentType]
        # tolerated: QICK merges the typed acquire ge_radius option into its heterogeneous acquire_params mapping.
        ge_radius: float | None = self.acquire_params.get("ge_radius")  # pyright: ignore[reportAssignmentType]

        # tolerated: QICK stores its boolean remove_offset option alongside transport and string options without a per-key type.
        remove_offset: bool = self.acquire_params["remove_offset"]  # pyright: ignore[reportAssignmentType]

        if threshold is not None:
            d_reps = [np.zeros_like(d) for d in acc_buf]
            if ge_radius is not None:
                warnings.warn(
                    "Both threshold and ge_radius are set, threshold will be applied and ge_radius will be ignored",
                    stacklevel=2,
                    skip_file_prefixes=_ACQUIRE_WARNING_INTERNAL_PATHS,
                )
            self.shots = self._apply_threshold(acc_buf, threshold, angle, remove_offset)
            for i, ch_shot in enumerate(self.shots):
                d_reps[i][..., 0] = ch_shot
            return self._average_buf(d_reps, length_norm=False)
        if ge_radius is not None:
            d_reps = [np.zeros_like(d) for d in acc_buf]
            if g_center is None or e_center is None:
                raise ValueError(
                    "g_center and e_center must be provided when ge_radius is set"
                )
            self.shots = self._apply_classification(
                acc_buf, g_center, e_center, ge_radius, remove_offset=remove_offset
            )
            for i, ch_shot in enumerate(self.shots):
                d_reps[i] = ch_shot
            return self._average_buf(d_reps, length_norm=False)
        d_reps = acc_buf
        return self._average_buf(
            d_reps,
            length_norm=True,
            remove_offset=remove_offset,
        )

    def _apply_classification(
        self,
        acc_buf: Sequence[NDArray[np.float64] | NDArray[np.int64]],
        g_center: complex,
        e_center: complex,
        ge_radius: float,
        *,
        remove_offset: bool,
    ) -> list[NDArray[np.float64]]:
        assert isinstance(self, qick_asm.AbsQickProgram)
        shots = []
        for i_ch, (ro_ch, ro) in enumerate(self.ro_chs.items()):
            avg = acc_buf[i_ch] / ro["length"]
            if remove_offset:
                offset = self.soccfg["readouts"][ro_ch]["iq_offset"]
                avg -= offset
            g_shot, e_shot, _ = classify_shots(
                avg.dot([1, 1j]), g_center, e_center, ge_radius
            )
            shots.append(np.stack([g_shot, e_shot], axis=-1).astype(np.float64))
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

        assert self.acquire_params is not None

        # tolerated(cast): QICK's extra_args update erases the typed acquire trackers list; retain the original objects for completed-round updates.
        trackers = cast(
            list[TrackerProtocol] | None, self.acquire_params.get("trackers")
        )
        if trackers is not None:
            assert self.acc_buf is not None
            if self.acquire_params["type"] != "accumulated":
                raise NotImplementedError(
                    "Tracker is not implemented for type other than accumulated"
                )

            if self.acquire_params["threshold"] is not None:
                raise NotImplementedError(
                    "Tracker is not implemented for thresholded data"
                )

            assert isinstance(self, qick_asm.AbsQickProgram)
            ro_chs = self.ro_chs

            if len(trackers) != len(self.acc_buf):
                raise ValueError(
                    f"Number of tracker ({len(trackers)}) must match number of readout channels ({len(self.acc_buf)})"
                )

            assert isinstance(ro_chs, dict)
            assert len(self.acc_buf) == len(ro_chs)
            for d_rep, tracker, ro in zip(
                self.acc_buf, trackers, ro_chs.values(), strict=False
            ):
                assert self.avg_level is not None
                d_rep = np.moveaxis(d_rep, [-2, self.avg_level], [0, -2])

                if not ro["edge_counting"]:
                    d_rep = d_rep / ro["length"]
                # tolerated(cast): The tracker protocol requires float64, but QICK edge-counting buffers stay int64; preserve this no-conversion handoff.
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
                "The callback data may be corrupted, fallback to normal summarize",
                stacklevel=2,
                skip_file_prefixes=_ACQUIRE_WARNING_INTERNAL_PATHS,
            )
            return self._summarize_accumulated(rounds_buf)

        assert isinstance(self, qick_asm.AbsQickProgram)
        n_ro_chs = len(self.ro_chs)

        assert self.rounds_buf is not None
        if self._inc_sum_state is None:  # first round
            self._inc_sum_state = [
                np.zeros_like(self.rounds_buf[0][i], dtype=np.float64)
                for i in range(n_ro_chs)
            ]

        # update the state with the new data
        for i in range(n_ro_chs):
            self._inc_sum_state[i] += rounds_buf[-1][i]

        return [d / self._inc_sum_count for d in self._inc_sum_state]

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
        round_hook = acquire_params.get("round_hook")
        if round_hook is not None:
            assert callable(round_hook), "round_hook must be a callable function"
            assert self.rounds_buf is not None

            # NOTE: increment the summary to reduce cpu usage
            round_count = len(self.rounds_buf)
            acquire_type = acquire_params["type"]
            if acquire_type == "accumulated":
                avg_d = self._inc_summarize_accumulated(self.rounds_buf)
                round_hook(round_count, avg_d, self._round_hook_cancel_flag())
            elif acquire_type == "decimated":
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
        # tolerated(cast): QICK's dynamic mapping loses the flag protocol; the callback must receive the original acquire flag, not a replacement.
        cancel_flag = cast(CancelFlagProtocol | None, acquire_params.get("cancel_flag"))
        if cancel_flag is None:
            raise RuntimeError("round_hook requires acquire cancel_flag")
        return cancel_flag


class ImproveAcquireMixin(
    RoundHookMixin, TrackerMixin, SingleShotMixin, EarlyStopMixin
): ...
