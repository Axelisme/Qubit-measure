from __future__ import annotations

import functools
import operator
import os
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import prod
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

from zcu_tools.program._shot_masks import ShotMasks, ShotMaskSource
from zcu_tools.program.shot_processing import (
    RoundReduction,
    ShotReducer,
    start_reduction,
)
from zcu_tools.progress_bar import make_pbar

__all__ = [
    "ImproveAcquireMixin",
    "CancelFlagProtocol",
    "StoppedPartialAcquireError",
    "AcquisitionContractError",
    "ShotReducer",
    "ShotMaskSource",
    "ShotResult",
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


@dataclass(frozen=True)
class ShotResult:
    """Visible raw integrated IQ and its unapplied post-selection mask.

    shots maps ADC channel numbers to int64 arrays (points, reps, reads, 2)
    without length normalization or offset subtraction. Points flatten the
    non-repetition loops in C order. Reads are in trigger order, excluding
    consumed reads and ADCs with no visible reads.
    mask is bool (points, reps), True for shots kept by all registered sources.
    With no sources every entry is True. Rejected shots are still in shots.
    """

    shots: Mapping[int, NDArray[np.int64]]
    mask: NDArray[np.bool_]


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
            return self._stop_interrupted_round()

        if self.acquire_params["type"] == "accumulated" and cancel_flag is not None:
            return self._finish_accumulated_round(cancel_flag)

        not_finish = (
            self._finish_decimated_round()
            if self.acquire_params["type"] == "decimated"
            else super().finish_round()
        )
        if not_finish and cancel_flag is not None and cancel_flag.is_set():
            self._close_rounds_pbar()
            return False
        return not_finish

    def finish_acquire(self):
        params = self.acquire_params
        if params is not None:
            stopped_early = params.get("_stopped_early") is True
            # tolerated(cast): QICK's heterogeneous params lose the cancel flag protocol.
            cancel_flag = cast(CancelFlagProtocol | None, params.get("cancel_flag"))
            if (
                stopped_early or (cancel_flag is not None and cancel_flag.is_set())
            ) and self._completed_round_count() == 0:
                if not stopped_early:
                    self._close_rounds_pbar()
                raise StoppedPartialAcquireError(
                    "acquire stopped before the first round completed"
                )
        return super().finish_acquire()

    def _stop_interrupted_round(self) -> bool:
        """Clean up the interrupted round and close progress; return False to stop."""
        assert self.acquire_params is not None
        self._cleanup_prepared_round()
        self._close_rounds_pbar()
        # ADR-0083: the params are new for each acquire; leave the caller's flag alone.
        self.acquire_params["_stopped_early"] = True
        return False

    def _finish_decimated_round(self) -> bool:
        return super().finish_round()

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

        n_ro_chs = len(rounds_buf[0])

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


class ImproveAcquireMixin(RoundHookMixin, EarlyStopMixin):
    _shot_reduction: RoundReduction | None = None
    _summed_stepped: bool = False
    _shot_masks: ShotMasks | None = None
    _last_round_mask: NDArray[np.bool_] | None = None

    def register_shot_mask_source(
        self, consumed_reads: Mapping[int, Sequence[int]], source: ShotMaskSource
    ) -> None:
        """Register a callback before acquisition.

        consumed_reads maps ADC channels to zero-based physical read indices in
        callback order. Each completed round passes only those reads as float64
        (points, reps, reads, 2) arrays of length-normalized IQ without offset
        subtraction. The callback returns bool (points, reps), True to keep a
        shot. Sources run in registration order and their masks are ANDed.
        Results omit consumed reads and ADCs with no remaining reads. Each IQ
        average is Σ m·x / N, where N counts every shot in the completed rounds.
        Non-integer or negative indices raise ValueError at registration.
        Indices outside compiled read counts and invalid masks raise ValueError
        on acquire. Callback exceptions propagate.
        """
        masks = self._shot_masks if self._shot_masks is not None else ShotMasks()
        masks.register(consumed_reads, source)
        self._shot_masks = masks

    @property
    def has_shot_mask_sources(self) -> bool:
        """Whether integrated acquisition requires host shot selection."""
        return self._shot_masks is not None

    def clear_shot_mask_sources(self) -> None:
        """Discard registered sources when rebuilding the program's declarations."""
        self._shot_masks = None

    def check_shot_mask_options(
        self, *, threshold: object = None, decimated: bool = False
    ) -> None:
        """Raise ValueError before acquisition for options incompatible with masks.

        With registered sources, reject decimated=True and any non-None native
        QICK threshold. Otherwise these options are unchanged.
        """
        if not self.has_shot_mask_sources:
            return
        if decimated:
            raise ValueError(
                "acquire_decimated does not support registered shot mask sources"
            )
        if threshold is not None:
            raise ValueError(
                "acquire with native QICK threshold does not support registered shot mask sources"
            )

    def logical_read_count(self, channel: int, physical_count: int) -> int:
        """Count unconsumed reads among the first physical_count triggers of an ADC."""
        if self._shot_masks is None:
            return physical_count
        return len(self._shot_masks.visible_reads(channel, physical_count))

    def logical_read_index(self, channel: int, physical_index: int) -> int | None:
        """Translate an ADC's zero-based physical read index, or None if consumed."""
        if self._shot_masks is None:
            return physical_index
        return self._shot_masks.logical_index(channel, physical_index)

    def get_shot_result(self) -> ShotResult | None:
        """Return the last integrated acquire round's visible raw shots and mask.

        Call after a successful synchronous integrated acquire. Return None
        if the program has no retained raw buffer, and raise RuntimeError if
        registered sources did not produce a mask for the retained round.
        get_raw() remains the physical transport buffer including consumed reads.
        """
        raw = self.get_raw()
        if raw is None:
            return None

        assert isinstance(self, qick_asm.AbsQickProgram)
        assert self.loop_dims is not None and self.avg_level is not None
        reps = self.loop_dims[self.avg_level]
        points = int(np.prod(self.loop_dims)) // reps

        masks = self._shot_masks
        mask = self._last_round_mask
        if masks is None:
            mask = np.ones((points, reps), dtype=np.bool_)
        if mask is None:
            raise RuntimeError("program did not produce a mask for the retained shots")

        shots = {}
        for ch, data in zip(self.ro_chs, raw, strict=True):
            count = data.shape[-2]
            logical = np.moveaxis(data, self.avg_level, -3).reshape(
                points, reps, count, 2
            )
            if masks is not None:
                visible = masks.visible_reads(ch, count)
                if not visible:
                    continue
                if len(visible) != count:
                    logical = logical[:, :, visible, :]
            shots[ch] = logical
        return ShotResult(shots=shots, mask=mask)

    def acquire(
        self,
        *args,
        _shot_reduction: RoundReduction | None = None,
        **kwargs,
    ):
        """Acquire averaged IQ or run the reduction provided by acquire_summed.

        QICK options, round hooks, cancellation and stepped ownership follow the
        native pipeline. With registered sources and no reducer, compute
        Σ m·x / N on the host from offset-corrected IQ, where N counts every
        shot in the completed rounds. Native threshold projection rejects sources.
        """
        self.check_shot_mask_options(threshold=kwargs.get("threshold"))
        self._last_round_mask = None
        self._shot_reduction = _shot_reduction
        self._summed_stepped = _shot_reduction is not None and kwargs.get(
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
        herald or registered shot mask sources raise ValueError. Cancellation
        retains completed rounds or raises StoppedPartialAcquireError if none
        completed; transport errors propagate.
        """
        self.check_shot_mask_options(decimated=True)
        self._shot_reduction = None
        self._summed_stepped = False
        return super().acquire_decimated(*args, **kwargs)

    def acquire_summed[State](
        self,
        soc: object,
        *,
        reducer: ShotReducer[State],
        rounds: int = 1,
        step_rounds: bool = False,
        **qick_options: object,
    ) -> NDArray[np.float64] | None:
        """Reduce completed rounds using one state owned by this acquisition.

        reducer.add receives normalized float64 ADC arrays (points, reps, reads, 2)
        without IQ-offset subtraction, in C-order non-reps loop order. Source
        reads and mask-only ADCs are omitted. The bool (points, reps) mask is
        all True if no sources are registered. Each complete round updates state
        once. Final results and round_hook data are detached reducer.output
        arrays reshaped to (*point_dims, *axes); the framework does not average.
        Invalid outputs raise ValueError with the expected and actual dtype/shape.
        Reducer and source exceptions propagate. Explicit threshold, angle and
        herald options raise ValueError; other QICK options are forwarded.
        Cancellation discards incomplete rounds, retains completed state, and
        raises StoppedPartialAcquireError if none completed. step_rounds=True
        prepares hardware only and returns None; finish_round / prepare_round drive
        rounds, and finish_acquire returns the final ndarray. Simulation rejects
        stepped acquisition.
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
        point_dims = self._point_dims()
        result = self.acquire(
            soc,
            _shot_reduction=start_reduction(reducer, point_dims),
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
        if self._shot_reduction is not None:
            return [self._shot_reduction.output()]
        if self.has_shot_mask_sources:
            return [
                np.asarray(np.mean(channel_rounds, axis=0))
                for channel_rounds in zip(*rounds_buf, strict=True)
            ]
        return super()._summarize_accumulated(rounds_buf)

    def _inc_summarize_accumulated(self, rounds_buf):
        if self._shot_reduction is not None:
            return self._summarize_accumulated(rounds_buf)
        return super()._inc_summarize_accumulated(rounds_buf)

    def _point_dims(self) -> tuple[int, ...]:
        if self.loop_dims is None or self.avg_level is None:
            raise ValueError("shot processing requires compiled acquisition loops")
        return tuple(
            size for axis, size in enumerate(self.loop_dims) if axis != self.avg_level
        )

    def _prepare_shot_round(
        self,
        acc_buf: Sequence[NDArray[np.float64] | NDArray[np.int64]],
        point_dims: tuple[int, ...],
    ) -> tuple[dict[int, NDArray[np.float64]], NDArray[np.bool_]]:
        """Normalize physical IQ, evaluate the mask and omit consumed reads."""
        assert isinstance(self, qick_asm.AbsQickProgram)
        assert self.loop_dims is not None and self.avg_level is not None
        points = prod(point_dims)
        reps = self.loop_dims[self.avg_level]
        normalized = {
            ch: np.moveaxis(raw / ro["length"], self.avg_level, -3).reshape(
                points, reps, ro["trigs"], 2
            )
            for raw, (ch, ro) in zip(acc_buf, self.ro_chs.items(), strict=True)
        }
        masks = self._shot_masks
        if masks is None:
            mask = np.ones((points, reps), dtype=np.bool_)
        else:
            reads = {ch: ro["trigs"] for ch, ro in self.ro_chs.items()}
            masks.validate_read_counts(reads)
            mask = masks.evaluate(normalized, (points, reps))
            normalized = {
                ch: normalized[ch][:, :, visible, :]
                for ch, count in reads.items()
                if (visible := masks.visible_reads(ch, count))
            }
        return normalized, mask

    def _process_summed(
        self, acc_buf: Sequence[NDArray[np.float64] | NDArray[np.int64]]
    ) -> list[NDArray[np.float64]]:
        assert self._shot_reduction is not None
        shots, mask = self._prepare_shot_round(acc_buf, self._point_dims())
        self._shot_reduction.add(shots, mask)
        self._last_round_mask = mask
        # QICK tracks completed rounds by list length. Reduction state replaces
        # per-round payloads, so only the completion marker goes in rounds_buf.
        return []

    def _process_masked_average(
        self, acc_buf: Sequence[NDArray[np.float64] | NDArray[np.int64]]
    ) -> list[NDArray[np.float64]]:
        assert isinstance(self, qick_asm.AbsQickProgram)
        assert self.acquire_params is not None
        point_dims = self._point_dims()
        shots, mask = self._prepare_shot_round(acc_buf, point_dims)
        self._last_round_mask = mask
        averaged = []
        for ch, iq in shots.items():
            if self.acquire_params["remove_offset"]:
                iq -= self._ro_offset(ch, self.ro_chs[ch].get("ro_config"))
            means = np.sum(iq, axis=1, where=mask[..., None, None]) / mask.shape[1]
            averaged.append(
                np.moveaxis(means.reshape(*point_dims, iq.shape[-2], 2), -2, 0)
            )
        return averaged

    def _process_accumulated(self, acc_buf) -> list[NDArray[np.float64]]:
        if self._shot_reduction is not None:
            return self._process_summed(acc_buf)
        if self.has_shot_mask_sources:
            return self._process_masked_average(acc_buf)
        return super()._process_accumulated(acc_buf)
