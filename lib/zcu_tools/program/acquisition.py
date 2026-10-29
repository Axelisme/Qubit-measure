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

from zcu_tools.cfg_model import Centroid3Classifier, CircleClassifier, ReadoutClassifier
from zcu_tools.program.herald import HeraldSpec, herald_ground_mask
from zcu_tools.progress_bar import make_pbar
from zcu_tools.utils.shot_classification import classify_shots, nearest_centroid3

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


def _classified_populations(
    iq: NDArray[np.complex128], classifier: ReadoutClassifier
) -> NDArray[np.float64]:
    if isinstance(classifier, CircleClassifier):
        if classifier.radius is None:
            raise ValueError("circle radius missing; run GE Post analysis")
        ground, excited, _ = classify_shots(
            iq, classifier.g_center, classifier.e_center, classifier.radius
        )
    elif isinstance(classifier, Centroid3Classifier):
        labels = nearest_centroid3(
            iq, classifier.g_center, classifier.e_center, classifier.f_center
        )
        ground, excited = labels == 0, labels == 1
    else:
        # GE theta rotates by exp(+i theta), opposite QICK's projection angle.
        excited = (iq * np.exp(1j * classifier.angle)).real > classifier.threshold
        ground = ~excited
    return np.stack([ground, excited], axis=-1).astype(np.float64)


class SingleShotMixin(TypedAcquireMixin):
    _classify: Mapping[int, ReadoutClassifier] | None = None

    @overload
    def acquire(
        self,
        soc: object,
        *,
        step_rounds: Literal[False] = False,
        classify: Mapping[int, ReadoutClassifier] | None = None,
        **kwargs: object,
    ) -> list[NDArray[np.float64]]: ...

    @overload
    def acquire(
        self,
        soc: object,
        *,
        step_rounds: Literal[True],
        classify: Mapping[int, ReadoutClassifier] | None = None,
        **kwargs: object,
    ) -> None: ...

    @overload
    def acquire(
        self,
        soc: object,
        *args,
        classify: Mapping[int, ReadoutClassifier] | None = None,
        **kwargs,
    ) -> list[NDArray[np.float64]] | None: ...

    def acquire(
        self,
        *args,
        classify: Mapping[int, ReadoutClassifier] | None = None,
        **kwargs,
    ) -> list[NDArray[np.float64]] | None:
        """Acquire IQ, or g/e populations with explicit per-ADC calibration.

        classify maps ADC channel numbers to circle/linear/centroid3 calibration in raw
        sums divided by ADC length, without IQ-offset subtraction. None retains
        QICK IQ/threshold behavior. Circle requires a Post radius. Population
        output ends in (g, e); 1-g-e is circle's other or centroid3's f.
        Centroid3 chooses the nearest g/e/f center, ties in that order.
        Linear excited
        means real(IQ * exp(1j*angle)) > threshold; equality belongs to ground.
        Keys must exactly match declared ADC channels or ValueError is raised
        before transport acquisition. No program registry supplies calibration.
        """
        if classify is not None:
            assert isinstance(self, qick_asm.AbsQickProgram)
            missing = sorted(self.ro_chs.keys() - classify.keys())
            unexpected = sorted(classify.keys() - self.ro_chs.keys())
            if missing or unexpected:
                raise ValueError(
                    f"classify ADC keys: missing={missing}, unexpected={unexpected}"
                )
            for ch, classifier in classify.items():
                if (
                    isinstance(classifier, CircleClassifier)
                    and classifier.radius is None
                ):
                    raise ValueError(
                        f"ADC {ch}: circle radius missing; run GE Post analysis"
                    )
        self._classify = classify
        return super().acquire(*args, **kwargs)

    def _process_accumulated(self, acc_buf) -> list[NDArray[np.float64]]:
        if self._classify is None:
            return super()._process_accumulated(acc_buf)
        assert isinstance(self, qick_asm.AbsQickProgram)
        shots: list[NDArray[np.float64]] = []
        for raw, (ch, ro) in zip(acc_buf, self.ro_chs.items(), strict=True):
            classifier = self._classify[ch]
            iq = (raw / ro["length"]).dot([1, 1j])
            shots.append(_classified_populations(iq, classifier))
        self.shots = shots
        return self._average_buf(shots, length_norm=False)


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


class HeraldAcquireMixin(RoundHookMixin, SingleShotMixin):
    """Own per-point accepted sums/counts, shared by transport and simulation.

    This modifier precedes RoundHookMixin so hooks and finish_acquire use the
    same count-weighted summary instead of averaging round means equally.
    """

    _herald_specs: dict[int, HeraldSpec] | None = None
    _herald_sums: list[NDArray[np.float64]] | None = None
    _herald_counts: list[NDArray[np.int64]] | None = None
    _herald_attempts: int = 0

    def acquire(
        self,
        *args,
        herald: Sequence[HeraldSpec] | None = None,
        trackers: list[TrackerProtocol] | None = None,
        **kwargs,
    ):
        """Post-select reads using explicit herald specs, at most one per ADC.

        herald identifies ADC numbers and trigger-order reads including herald.
        None or an empty sequence retains ordinary acquisition. Results retain
        ADC declaration order and have shape (non-herald reads, *point_dims, 2).
        IQ is normalized by ADC samples with the ordinary remove_offset policy.
        Herald calibration always uses raw sums/length without offset subtraction.
        classify applies its per-ADC calibration only to accepted signal reads,
        returning (g,e) instead of IQ and taking precedence over threshold/angle.
        Otherwise QICK threshold/angle projects accepted shots in its original
        radians/sign/offset convention and returns (excited_fraction,0).
        Each ADC uses only its own herald mask. Completed rounds are weighted by
        accepted shot count; no accepted shots produces NaN in both components.
        trackers together with herald, duplicate/undeclared ADCs, count mismatches
        and invalid/terminal read indices raise ValueError before transport.
        Cancellation keeps completed rounds only and raises
        StoppedPartialAcquireError if none completed. Ordinary QICK options,
        including step_rounds, are forwarded. No new per-shot history is retained.
        """
        self._reset_herald()
        if herald:
            if trackers is not None:
                raise ValueError("herald cannot be combined with trackers")
            self._herald_specs = self._validate_herald(herald)
        return super().acquire(*args, trackers=trackers, **kwargs)

    def acquire_decimated(
        self, *args, herald: Sequence[HeraldSpec] | None = None, **kwargs
    ):
        """Acquire traces without herald; nonempty herald raises ValueError.

        An ordinary decimated call clears prior herald pass fractions. Other
        parameters, return shapes, errors and cancellation follow acquire_decimated
        on the normal QICK pipeline. Post-selection requires integrated shots.
        """
        self._reset_herald()
        if herald:
            raise ValueError("acquire_decimated does not support herald")
        return super().acquire_decimated(*args, **kwargs)

    def _reset_herald(self) -> None:
        self._herald_specs = None
        self._herald_sums = None
        self._herald_counts = None
        self._herald_attempts = 0

    def _validate_herald(self, herald: Sequence[HeraldSpec]) -> dict[int, HeraldSpec]:
        assert isinstance(self, qick_asm.AbsQickProgram)
        specs: dict[int, HeraldSpec] = {}
        for spec in herald:
            ch = spec.ro_ch
            if ch not in self.ro_chs:
                raise ValueError(f"Herald ADC {ch} is not declared")
            if ch in specs:
                raise ValueError(f"ADC {ch} supports at most one herald")
            actual_reads = self.ro_chs[ch]["trigs"]
            if spec.reads_per_shot != actual_reads:
                raise ValueError(
                    f"ADC {ch}: herald reads_per_shot={spec.reads_per_shot}, "
                    f"actual reads_per_shot={actual_reads}"
                )
            if (
                type(spec.read_index) is not int
                or not 0 <= spec.read_index < actual_reads - 1
            ):
                raise ValueError(
                    f"ADC {ch}: herald read_index={spec.read_index} must be a "
                    "nonnegative integer with at least one later read"
                )
            specs[ch] = spec
        return specs

    def get_herald_pass_fraction(self) -> dict[int, NDArray[np.float64]] | None:
        """Return detached per-ADC accepted/attempted fractions in [0,1].

        Keys are herald ADC numbers. Each float64 array has shape point_dims,
        with repetitions and reads removed (a scalar array without sweeps).
        Only completed rounds contribute, including after cancellation. Returns
        None before a herald acquisition completes any round or when herald is
        omitted on the latest acquire. A new acquire clears the previous state.
        This getter neither samples shots nor runs a transport/backend.
        """
        if self._herald_specs is None or self._herald_counts is None:
            return None
        assert isinstance(self, qick_asm.AbsQickProgram)
        return {
            ch: np.asarray(
                self._herald_counts[index] / self._herald_attempts, dtype=np.float64
            )
            for index, ch in enumerate(self.ro_chs)
            if ch in self._herald_specs
        }

    def _herald_buffers(self, acc_buf):
        assert isinstance(self, qick_asm.AbsQickProgram)
        assert self._herald_specs is not None
        buffers = []
        masks = []
        for raw, (ch, ro) in zip(acc_buf, self.ro_chs.items(), strict=True):
            spec = self._herald_specs.get(ch)
            mask = np.ones(raw.shape[:-2], dtype=np.bool_)
            if spec is not None:
                iq = (raw[..., spec.read_index, :] / ro["length"]).dot([1, 1j])
                mask = herald_ground_mask(iq, spec.classifier)
                raw = np.delete(raw, spec.read_index, axis=-2)
            buffers.append(raw)
            masks.append(mask)
        return buffers, masks

    def _herald_selected_values(self, selected) -> list[NDArray[np.float64]]:
        assert isinstance(self, qick_asm.AbsQickProgram)
        assert self.acquire_params is not None
        if self._classify is not None:
            return [
                _classified_populations(
                    (raw / ro["length"]).dot([1, 1j]), self._classify[ch]
                )
                for raw, (ch, ro) in zip(selected, self.ro_chs.items(), strict=True)
            ]
        params = self.acquire_params
        if params["threshold"] is not None:
            # Reuse QICK's scalar/list, angle sign, offset and equality policy.
            # Every input contains accepted signal reads only, before projection.
            shots = self._apply_threshold(
                selected, params["threshold"], params["angle"], params["remove_offset"]
            )
            return [np.stack([shot, np.zeros_like(shot)], axis=-1) for shot in shots]
        values = []
        for raw, (ch, ro) in zip(selected, self.ro_chs.items(), strict=True):
            value = raw.astype(np.float64)
            if not ro["edge_counting"]:
                value /= ro["length"]
                if params["remove_offset"]:
                    value -= self._ro_offset(ch, ro.get("ro_config"))
            values.append(value)
        return values

    def _process_accumulated(self, acc_buf) -> list[NDArray[np.float64]]:
        if self._herald_specs is None:
            return super()._process_accumulated(acc_buf)
        assert self.avg_level is not None
        buffers, masks = self._herald_buffers(acc_buf)
        # Flatten only point/repetition coordinates, preserving each read's IQ.
        # Rejected shots never enter either classification or QICK projection.
        selected = self._herald_selected_values(
            [raw[mask] for raw, mask in zip(buffers, masks, strict=True)]
        )
        sums: list[NDArray[np.float64]] = []
        counts: list[NDArray[np.int64]] = []
        for raw, mask, values in zip(buffers, masks, selected, strict=True):
            dense = np.zeros_like(raw, dtype=np.float64)
            dense[mask] = values
            sums.append(np.moveaxis(dense.sum(axis=self.avg_level), -2, 0))
            counts.append(np.asarray(mask.sum(axis=self.avg_level), dtype=np.int64))
        if self._herald_sums is None:
            self._herald_sums = [value.copy() for value in sums]
            self._herald_counts = [value.copy() for value in counts]
        else:
            assert self._herald_counts is not None
            for total, value in zip(self._herald_sums, sums, strict=True):
                total += value
            for total, value in zip(self._herald_counts, counts, strict=True):
                total += value
        self._herald_attempts += acc_buf[0].shape[self.avg_level]
        return self._herald_means(sums, counts)

    def _herald_means(
        self, sums: list[NDArray[np.float64]], counts: list[NDArray[np.int64]]
    ) -> list[NDArray[np.float64]]:
        return [
            np.divide(
                total,
                count[..., None],
                out=np.full_like(total, np.nan),
                where=count[..., None] != 0,
            )
            for total, count in zip(sums, counts, strict=True)
        ]

    def _summarize_accumulated(
        self, rounds_buf: list[NDArray[np.float64]]
    ) -> list[NDArray[np.float64]]:
        if self._herald_specs is None:
            return super()._summarize_accumulated(rounds_buf)
        assert self._herald_sums is not None
        assert self._herald_counts is not None
        return self._herald_means(self._herald_sums, self._herald_counts)

    def _inc_summarize_accumulated(
        self, rounds_buf: list[NDArray[np.float64]]
    ) -> list[NDArray[np.float64]]:
        if self._herald_specs is None:
            return super()._inc_summarize_accumulated(rounds_buf)
        return self._summarize_accumulated(rounds_buf)


class ImproveAcquireMixin(
    HeraldAcquireMixin, TrackerMixin, SingleShotMixin, EarlyStopMixin
): ...
