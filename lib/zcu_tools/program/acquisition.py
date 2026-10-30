from __future__ import annotations

import functools
import operator
import os
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Literal, Protocol, TypeAlias, cast, overload

import numpy as np
from numpy.typing import NDArray
from qick import qick_asm
from qick.qick_asm import (
    AcquireMixin,
    logger,
    # tolerated: QICK imports obtain dynamically from optional RPyC or defines its local fallback.
    obtain,  # pyright: ignore[reportAttributeAccessIssue]
)

from zcu_tools.program._shot_processing import validated_sums
from zcu_tools.progress_bar import make_pbar

__all__ = [
    "ImproveAcquireMixin",
    "CancelFlagProtocol",
    "StoppedPartialAcquireError",
    "AcquisitionContractError",
    "TrackerProtocol",
    "ShotSumProcessor",
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


# All declared ADCs, normalized (points, reps, reads, 2), to (points, *axes) sums.
ShotSumProcessor: TypeAlias = Callable[
    [Mapping[int, NDArray[np.float64]]], NDArray[np.float64]
]


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

        Nonempty herald raises ValueError because post-selection needs integrated
        shots. Omission or an empty sequence leaves the trace path unchanged.
        Preparation returns None. Mode typing is precise with only soc
        positional; extra positional options or a dynamic bool retain the union.
        QICK argument/transport errors propagate. Cancellation retains completed
        rounds, or raises StoppedPartialAcquireError if none completed.
        """
        if kwargs.pop("herald", None):
            raise ValueError("acquire_decimated does not support herald")
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


class ImproveAcquireMixin(RoundHookMixin, TrackerMixin, EarlyStopMixin):
    _sum_processor: ShotSumProcessor | None = None
    _summed_stepped: bool = False

    def acquire(
        self,
        *args,
        _sum_processor: ShotSumProcessor | None = None,
        **kwargs,
    ):
        """Acquire native IQ or dispatch one joint-ADC shot-sum processor.

        acquire_summed supplies the reserved _sum_processor option. QICK options,
        round hooks, cancellation and stepped ownership follow the native
        pipeline; each new acquire resets processor output dimensions.
        """
        self._sum_processor = _sum_processor
        self._sum_shape: tuple[int, ...] | None = None
        self._summed_stepped = _sum_processor is not None and kwargs.get(
            "step_rounds", False
        )
        return super().acquire(*args, **kwargs)

    def acquire_decimated(self, *args, **kwargs):
        """Acquire native ADC traces, clearing previous summed-acquire dispatch.

        Positional args and kwargs follow TypedAcquireMixin.acquire_decimated:
        soc is required, rounds controls software averaging, and step_rounds=True
        prepares only (None). Synchronous/finish_acquire results are ADC-ordered
        float64 trace arrays. Native QICK timing, lookback, remove_offset,
        round_hook and cancel_flag options pass through unchanged. Nonempty
        herald raises ValueError. Cancellation retains completed rounds or raises
        StoppedPartialAcquireError if none completed; transport errors propagate.
        """
        self._sum_processor = None
        self._summed_stepped = False
        return super().acquire_decimated(*args, **kwargs)

    def acquire_summed(
        self,
        soc: object,
        *,
        processor: ShotSumProcessor,
        rounds: int = 1,
        step_rounds: bool = False,
        **qick_options: object,
    ) -> NDArray[np.float64] | None:
        """Sum one joint-ADC processor's per-point output over completed rounds.

        processor receives all declared ADCs as normalized float64 arrays of
        shape (points, reps, reads, 2), in C-order non-reps loop order. Divide
        integer integrals by ADC length without removing IQ offsets. Return
        float64 (points, *axes), with positive axes fixed during this acquire.
        Results and round_hook data have shape (*point_dims, *axes), without
        division by reps or rounds. Invalid outputs raise ValueError with the
        expected and actual dtype/shape. Processor exceptions propagate.
        threshold, angle and herald options raise ValueError; other QICK options
        are forwarded. Cancellation discards incomplete rounds, retains completed
        sums, and raises StoppedPartialAcquireError if none completed.
        step_rounds=True prepares hardware only and returns None; finish_round /
        prepare_round drive rounds, and finish_acquire returns the final ndarray.
        Simulation retains its existing rejection of stepped acquisition.
        """
        for option in ("threshold", "angle", "herald"):
            if option in qick_options:
                raise ValueError(
                    f"acquire_summed expected {option} omitted; actual {qick_options[option]!r}"
                )
        round_hook = qick_options.pop("round_hook", None)
        if round_hook is not None:
            if not callable(round_hook):
                raise ValueError(f"round_hook expected callable; actual {round_hook!r}")

            def hook(count, data, cancel_flag):
                round_hook(count, data[0], cancel_flag)

            qick_options["round_hook"] = hook
        result = self.acquire(
            soc,
            _sum_processor=processor,
            rounds=rounds,
            step_rounds=step_rounds,
            **qick_options,
        )
        return None if result is None else result[0]

    def finish_acquire(self):
        """Return completed acquisition data after manually driving hardware rounds.

        Stepped acquire_summed returns float64 (*point_dims, *axes) sums; native
        acquisition retains ADC-ordered lists. Cancelled incomplete
        rounds are discarded. No completed round on cancellation raises
        StoppedPartialAcquireError. Call after finish_round has returned False.
        """
        result = super().finish_acquire()
        return result[0] if self._summed_stepped and result is not None else result

    def _summarize_accumulated(self, rounds_buf) -> list[NDArray[np.float64]]:
        if self._sum_processor is not None:
            return [
                np.asarray(
                    np.sum([data[0] for data in rounds_buf], axis=0, dtype=np.float64)
                )
            ]
        return super()._summarize_accumulated(rounds_buf)

    def _inc_summarize_accumulated(self, rounds_buf):
        if self._sum_processor is not None:
            if self._inc_sum_state is None:
                self._inc_sum_state = [np.zeros_like(rounds_buf[-1][0])]
            self._inc_sum_state[0] += rounds_buf[-1][0]
            return [self._inc_sum_state[0].copy()]
        return super()._inc_summarize_accumulated(rounds_buf)

    def _process_summed(self, acc_buf) -> list[NDArray[np.float64]]:
        assert isinstance(self, qick_asm.AbsQickProgram)
        assert self.loop_dims is not None and self.avg_level is not None
        assert self._sum_processor is not None
        point_dims = tuple(
            n for i, n in enumerate(self.loop_dims) if i != self.avg_level
        )
        points = int(np.prod(point_dims))
        reps = self.loop_dims[self.avg_level]
        normalized = {
            ch: np.moveaxis(raw / ro["length"], self.avg_level, -3).reshape(
                points, reps, ro["trigs"], 2
            )
            for raw, (ch, ro) in zip(acc_buf, self.ro_chs.items(), strict=True)
        }
        payload = validated_sums(self._sum_processor(normalized))
        expected = self._sum_shape
        if expected is None:
            self._sum_shape = payload.shape
        if (
            payload.ndim < 1
            or payload.shape[0] != points
            or any(size < 1 for size in payload.shape[1:])
            or (expected is not None and payload.shape != expected)
        ):
            raise ValueError(
                f"processor expected shape {expected or f'({points}, *positive_axes)'}; "
                f"actual shape {payload.shape}"
            )
        return [payload.reshape(point_dims + payload.shape[1:]).copy()]

    def _process_accumulated(self, acc_buf) -> list[NDArray[np.float64]]:
        if self._sum_processor is not None:
            return self._process_summed(acc_buf)
        return super()._process_accumulated(acc_buf)
