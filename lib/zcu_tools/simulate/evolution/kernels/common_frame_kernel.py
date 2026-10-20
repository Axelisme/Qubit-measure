"""Kernel handoffs for common-frame products; no separate tensor cache."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..kernel import (
    ApplyContext,
    GuardPreparation,
    GuardResult,
    KernelContext,
    Prepared,
    SegmentTable,
    prepared_key_indices,
)
from ..workplan import FactKey
from .common_frame import (
    CommonFrame,
    CommonFrameAdmission,
    admit_common_frame,
    advance_admitted_common_frame,
    build_common_frame,
    coefficient_keys,
)


@dataclass(frozen=True)
class CommonFramePrepared:
    """Immutable coefficient batch charged by owned arrays, excluding models."""

    keys: NDArray[np.void]
    frames: tuple[CommonFrame, ...]

    @property
    def nbytes(self) -> int:
        """Owned key and compressed monomial array bytes."""
        return self.keys.nbytes + sum(frame.nbytes for frame in self.frames)


def _admission_key(
    table: SegmentTable, row: int, frame: CommonFrame
) -> FactKey[CommonFrameAdmission]:
    # Full immutable row content avoids aliasing recycled prefix-table IDs.
    count = len(frame.ports)
    physics = (
        frame.model.identity,
        frame.model.level,
        frame.ports,
        frame.frequencies_ghz.tobytes(),
        int(table.timeline[row]),
        row,
        float(table.start_ns[row]),
        float(table.duration_ns[row]),
        table.tone_amplitude[row, :count].tobytes(),
        table.tone_theta_rad[row, :count].tobytes(),
        table.tone_envelope_offset_ns[row, :count].tobytes(),
        tuple(table.envelopes[e].identity for e in table.tone_envelope[row, :count]),
    )
    return FactKey(("common-frame-admission", physics), CommonFrameAdmission)


class CommonFrameKernel:
    """Production SW2 variant with row-local admissions and charged handoffs.

    WorkPlan retains only small immutable admissions. Guard preparations own all
    coefficients and follow the ordinary bounded resident/pending account.
    """

    name = "common_frame"
    guard_builds = True

    def prepare_keys(
        self, table: SegmentTable, rows: NDArray[np.int64], context: KernelContext
    ) -> NDArray[np.void]:
        """Complete coefficient identity; apply-only data does not partition it."""
        return coefficient_keys(table, rows, context.source)

    def prepare(
        self,
        keys: NDArray[np.void],
        table: SegmentTable,
        rows: NDArray[np.int64],
        context: KernelContext,
    ) -> CommonFramePrepared:
        """Build one immutable product per unique key, including cost bypasses."""
        frames = []
        for row in rows:
            count = int(table.tones[row])
            model = context.source.model(tuple(int(v) for v in table.level[row]))
            frames.append(
                build_common_frame(
                    model,
                    tuple(int(v) for v in table.tone_port[row, :count]),
                    table.tone_frequency_ghz[row, :count],
                )
            )
        context.work_plan.routing.built(self.name, len(frames))
        owned_keys = keys.copy()
        owned_keys.setflags(write=False)
        return CommonFramePrepared(owned_keys, tuple(frames))

    def admit_prepared(
        self,
        prepared: Prepared,
        table: SegmentTable,
        rows: NDArray[np.int64],
        context: KernelContext,
    ) -> GuardResult | None:
        """Recheck actual gain, duration and envelope once per current row."""
        if not isinstance(prepared, CommonFramePrepared):
            raise TypeError("common-frame requires its own preparation")
        keys = self.prepare_keys(table, rows, context)
        indices = prepared_key_indices(prepared.keys, keys)
        if indices is None:
            return None
        accepted = []
        for row, key in zip(rows, keys, strict=True):
            frame = prepared.frames[indices[key.tobytes()]]

            def build(row=int(row), frame=frame) -> CommonFrameAdmission:
                context.work_plan.routing.admitted(self.name, 1)
                return admit_common_frame(frame, table, row)

            admission = context.work_plan.fact(
                _admission_key(table, int(row), frame), build
            )
            accepted.append(admission.accepted)
        return GuardResult(
            np.array(accepted, dtype=bool), (GuardPreparation(prepared.keys, prepared),)
        )

    def guard_prepared(
        self, table: SegmentTable, rows: NDArray[np.int64], context: KernelContext
    ) -> GuardResult:
        """Build unique products and hand them to apply without a second build."""
        keys, first = np.unique(
            self.prepare_keys(table, rows, context), return_index=True
        )
        prepared = self.prepare(keys, table, rows[first], context)
        result = self.admit_prepared(prepared, table, rows, context)
        if result is None:
            raise RuntimeError("new common-frame product did not cover its rows")
        return result

    def guard(
        self, table: SegmentTable, rows: NDArray[np.int64], context: KernelContext
    ) -> NDArray[np.bool_]:
        """Standalone validity check; production uses guard_prepared."""
        return self.guard_prepared(table, rows, context).accepted

    def cost(self, table: SegmentTable, rows: NDArray[np.int64]) -> NDArray[np.float64]:
        """Relative eight-nanosecond CF4 scheduling work, not CPU seconds."""
        return np.maximum(1.0, table.duration_ns[rows] / 8)

    def bind_input(
        self,
        prepared: Prepared,
        key_index: int,
        table: SegmentTable,
        rows: NDArray[np.int64],
        source_initial: NDArray[np.complex128],
        context: KernelContext,
    ) -> None:
        """No density-dependent coefficient contraction."""
        del prepared, key_index, table, rows, source_initial, context

    def apply(
        self,
        prepared: Prepared,
        key_index: NDArray[np.int64],
        table: SegmentTable,
        rows: NDArray[np.int64],
        states: NDArray[np.complex128],
        context: ApplyContext,
    ) -> NDArray[np.bool_]:
        """Consume the exact dispatch admission, never reassess or fall back."""
        context.require_no_plan()
        if not isinstance(prepared, CommonFramePrepared):
            raise TypeError("common-frame requires its own preparation")
        for index, (row, key) in enumerate(zip(rows, key_index, strict=True)):

            def missing() -> CommonFrameAdmission:
                raise RuntimeError("common-frame apply requires dispatch admission")

            admission = context.work_plan.fact(
                _admission_key(table, int(row), prepared.frames[key]), missing
            )
            states[index] = advance_admitted_common_frame(
                prepared.frames[key], table, int(row), states[index], admission
            )
        return np.zeros(len(rows), dtype=bool)
