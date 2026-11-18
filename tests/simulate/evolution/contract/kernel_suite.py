"""Reusable black-box checks for any C5 kernel.

Kernel tickets supply a KernelCase fixture to subclasses of KernelContract.
Each row uses its segment eigenbasis, with carriers and constant or sampled
envelopes. When table.flux_waveform is supplied the reference evaluates the
exact continuous Hamiltonian and can switch quantized baths inside a segment.
It constructs individual Lindblad jumps, never production RHS helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
import pytest
import zcu_tools.simulate.evolution as evolution
from numpy.typing import NDArray
from zcu_tools.simulate.evolution._dop853_dispatch import direct_plans
from zcu_tools.simulate.evolution.kernel import ApplyContext, KernelContext
from zcu_tools.simulate.evolution.workplan import WorkPlan

from .reference import reference
from .synthetic_models import synthetic_model as synthetic_model

if TYPE_CHECKING:
    from zcu_tools.simulate.evolution import kernel


def kernel_context(
    family: str, table: kernel.SegmentTable, budget: float
) -> ApplyContext:
    """Use the production timeline planner when invoking a direct solver seam."""
    names = np.full(len(table.timeline), family)
    plans, _ = direct_plans(table, names, budget=budget)
    return ApplyContext(budget, plans.get(family))


@dataclass(frozen=True)
class KernelCase:
    """One accepted set of independent rows and the states to evolve.

    kernel: algorithm under test; all rows must pass its guard.
    source, table: model source and segment parameters passed unchanged to C5.
    states: complex (rows,M,M) input densities, including coherence.
    budget: population and low-four absolute error allowance.
    """

    kernel: kernel.Kernel
    source: evolution.ModelSource
    table: kernel.SegmentTable
    states: NDArray[np.complex128]
    budget: float = 1e-3

    def context(self, rows: NDArray[np.int64]) -> ApplyContext:
        """Guard rows once and retain any family plan for repeated apply calls."""
        result = self.kernel.guard_prepared(
            self.table, rows, KernelContext(self.source, self.budget, WorkPlan())
        )
        if result is not None:
            assert result.accepted.all()
        if result is not None and result.plan is not None:
            return ApplyContext(self.budget, result.plan)
        return kernel_context(self.kernel.name, self.table, self.budget)


def assert_budget(
    actual: NDArray[np.complex128], expected: NDArray[np.complex128], budget: float
) -> None:
    """Check populations and the low-four density block independently."""
    assert np.isfinite(actual).all()
    assert (
        np.max(
            np.abs(
                np.diagonal(actual, axis1=-2, axis2=-1)
                - np.diagonal(expected, axis1=-2, axis2=-1)
            )
        )
        <= budget
    )
    assert np.max(np.abs(actual[..., :4, :4] - expected[..., :4, :4])) <= budget


class KernelContract:
    """Inherit and supply the kernel_case fixture; no kernel internals used."""

    def test_error_within_budget(self, kernel_case: KernelCase) -> None:
        case = kernel_case
        rows = np.arange(len(case.states), dtype=np.int64)
        assert case.kernel.guard(
            case.table, rows, KernelContext(case.source, case.budget, WorkPlan())
        ).all()
        keys, first, inverse = np.unique(
            case.kernel.prepare_keys(
                case.table, rows, KernelContext(case.source, case.budget, WorkPlan())
            ),
            return_index=True,
            return_inverse=True,
        )
        prepared = case.kernel.prepare(
            keys,
            case.table,
            rows[first],
            KernelContext(case.source, case.budget, WorkPlan()),
        )
        assert prepared.nbytes >= 0
        actual = case.states.copy()
        case.kernel.apply(
            prepared, inverse, case.table, rows, actual, case.context(rows)
        )
        expected = np.stack(
            [reference(case, row, state) for row, state in enumerate(case.states)]
        )
        assert_budget(actual, expected, case.budget)

    def test_guard_is_row_local(self, kernel_case: KernelCase) -> None:
        case = kernel_case
        rows = np.arange(len(case.table.timeline), dtype=np.int64)
        whole = case.kernel.guard(
            case.table, rows, KernelContext(case.source, case.budget, WorkPlan())
        )
        partitions = (
            tuple(rows[i : i + 1] for i in range(len(rows))),
            (rows[:1], rows[1:]),
            # Call the later segment first and reverse its rows, then restore
            # each result to its original row position.
            (rows[1:][::-1], rows[:1]),
        )
        for subsets in partitions:
            split = np.empty_like(whole)
            for subset in subsets:
                split[subset] = case.kernel.guard(
                    case.table,
                    subset,
                    KernelContext(case.source, case.budget, WorkPlan()),
                )
            np.testing.assert_array_equal(split, whole)

    def test_guard_prepared_matches_guard_and_fresh_apply(
        self, kernel_case: KernelCase
    ) -> None:
        case = kernel_case
        hook = getattr(case.kernel, "guard_prepared", None)
        assert callable(hook), "every C5 kernel must offer guard_prepared"
        rows = np.arange(len(case.states), dtype=np.int64)
        expected_mask = case.kernel.guard(
            case.table, rows, KernelContext(case.source, case.budget, WorkPlan())
        )
        for subset in (rows, rows[::-1], rows[::2], rows[:0]):
            result = case.kernel.guard_prepared(
                case.table, subset, KernelContext(case.source, case.budget, WorkPlan())
            )
            if result is None:
                continue
            np.testing.assert_array_equal(result.accepted, expected_mask[subset])
            assert result.accepted.dtype == np.bool_
            for item in result.preparations:
                keys = case.kernel.prepare_keys(
                    case.table,
                    subset,
                    KernelContext(case.source, case.budget, WorkPlan()),
                )
                positions = {key.tobytes(): i for i, key in enumerate(item.keys)}
                selected = np.array(
                    [i for i, key in enumerate(keys) if key.tobytes() in positions],
                    dtype=np.int64,
                )
                selected = selected[result.accepted[selected]]
                indices = np.array(
                    [positions[key.tobytes()] for key in keys[selected]], dtype=np.int64
                )
                chosen = subset[selected]
                actual = case.states[chosen].copy()
                case.kernel.apply(
                    item.prepared,
                    indices,
                    case.table,
                    chosen,
                    actual,
                    ApplyContext(case.budget, result.plan),
                )
                fresh_keys, first, inverse = np.unique(
                    keys[selected], return_index=True, return_inverse=True
                )
                fresh = case.kernel.prepare(
                    fresh_keys,
                    case.table,
                    chosen[first],
                    KernelContext(case.source, case.budget, WorkPlan()),
                )
                expected = case.states[chosen].copy()
                case.kernel.apply(
                    fresh,
                    inverse,
                    case.table,
                    chosen,
                    expected,
                    ApplyContext(case.budget, result.plan),
                )
                np.testing.assert_array_equal(actual, expected)

    def test_rows_can_be_split(self, kernel_case: KernelCase) -> None:
        case = kernel_case
        rows = np.arange(len(case.states), dtype=np.int64)
        keys, first, inverse = np.unique(
            case.kernel.prepare_keys(
                case.table, rows, KernelContext(case.source, case.budget, WorkPlan())
            ),
            return_index=True,
            return_inverse=True,
        )
        prepared = case.kernel.prepare(
            keys,
            case.table,
            rows[first],
            KernelContext(case.source, case.budget, WorkPlan()),
        )
        context = case.context(rows)
        whole = case.states.copy()
        whole_mask = case.kernel.apply(
            prepared, inverse, case.table, rows, whole, context
        )
        assert isinstance(whole_mask, np.ndarray), "apply must return an exceeded mask"
        assert whole_mask.shape == (len(rows),)
        assert whole_mask.dtype == np.bool_
        if case.kernel.name != "dyson":
            assert not whole_mask.any()
        split = case.states.copy()
        split_mask = np.empty(len(rows), dtype=np.bool_)
        # Non-contiguous, reversed subsets detect accidental row-position indexing.
        for subset in (rows[::2][::-1], rows[1::2][::-1]):
            chunk = split[subset].copy()
            mask = case.kernel.apply(
                prepared,
                inverse[subset],
                case.table,
                subset,
                chunk,
                context,
            )
            assert isinstance(mask, np.ndarray), "apply must return an exceeded mask"
            assert mask.shape == (len(subset),)
            assert mask.dtype == np.bool_
            split_mask[subset] = mask
            split[subset] = chunk
        np.testing.assert_allclose(split, whole, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(split_mask, whole_mask)

    def test_prepare_is_density_independent(self, kernel_case: KernelCase) -> None:
        case = kernel_case
        rows = np.arange(len(case.states), dtype=np.int64)
        keys, first, inverse = np.unique(
            case.kernel.prepare_keys(
                case.table,
                rows,
                KernelContext(case.source, case.budget, WorkPlan()),
            ),
            return_index=True,
            return_inverse=True,
        )

        def no_density_source() -> evolution.ModelSource:
            class NoDensity(evolution.ModelSource):
                @property
                def initial_density(self) -> NDArray[np.complex128]:
                    raise AssertionError("prepare must not read density")

            expansion = (
                case.source.flux_expansion()
                if case.table.flux_waveform is not None
                else None
            )
            return NoDensity(case.states[0], case.source.model, expansion)

        prepared = case.kernel.prepare(
            keys,
            case.table,
            rows[first],
            KernelContext(no_density_source(), case.budget, WorkPlan()),
        )
        context = case.context(rows)
        for states in (
            case.states,
            np.repeat(
                (np.eye(case.states.shape[-1]) / case.states.shape[-1])[None].astype(
                    complex
                ),
                len(rows),
                axis=0,
            ),
        ):
            actual = states.copy()
            case.kernel.apply(prepared, inverse, case.table, rows, actual, context)
            expected = np.stack(
                [reference(case, row, state) for row, state in enumerate(states)]
            )
            assert_budget(actual, expected, case.budget)

    @pytest.mark.parametrize("prefix_input", [False, True], ids=["source", "prefix"])
    def test_bind_input_is_read_only_and_apply_equivalent(
        self, kernel_case: KernelCase, prefix_input: bool
    ) -> None:
        case = kernel_case
        bind = getattr(case.kernel, "bind_input", None)
        assert callable(bind), "every C5 kernel must offer bind_input"
        table = replace(
            case.table,
            start_ns=np.full(len(case.states), 7.0 if prefix_input else 0.0),
            prefix_group=(tuple(0 for _ in case.states) if prefix_input else None),
            prefix_no_drive=(True,) if prefix_input else (),
        )
        rows = np.arange(len(case.states), dtype=np.int64)
        keys, first, inverse = np.unique(
            case.kernel.prepare_keys(
                table, rows, KernelContext(case.source, case.budget, WorkPlan())
            ),
            return_index=True,
            return_inverse=True,
        )
        prepared = case.kernel.prepare(
            keys,
            table,
            rows[first],
            KernelContext(case.source, case.budget, WorkPlan()),
        )
        context = replace(case, table=table).context(rows)
        initial = case.states[0].copy()
        if prefix_input:
            # Independently evolve a no-drive prefix. The hook accepts this
            # actual input at a later start, not just source.initial_density.
            idle = replace(
                table,
                start_ns=np.zeros(len(rows)),
                duration_ns=np.full(len(rows), 7.0),
                tones=np.zeros(len(rows), dtype=np.int64),
                tone_port=np.full_like(table.tone_port, -1),
                tone_envelope=np.full_like(table.tone_envelope, -1),
                prefix_group=None,
            )
            initial = reference(replace(case, table=idle), 0, initial)
        initial.setflags(write=False)
        before = initial.copy()
        for key in range(len(keys)):
            candidates = rows[inverse == key]
            ports = (
                table.tone_port[candidates, 0]
                if table.tone_port.shape[1]
                else np.zeros(len(candidates))
            )
            for port in np.unique(ports):
                selected = candidates[ports == port][::-1]
                saved_rows = selected.copy()
                bound = case.kernel.bind_input(
                    prepared,
                    key,
                    table,
                    selected,
                    initial,
                    KernelContext(case.source, case.budget, WorkPlan()),
                )
                np.testing.assert_array_equal(initial, before)
                np.testing.assert_array_equal(selected, saved_rows)
                if case.kernel.name == "dyson":
                    assert bound is not None, "Dyson must bind common-input paths"
                if bound is None:
                    continue
                states = np.repeat(initial[None], len(selected), axis=0)
                expected = states.copy()
                expected_mask = case.kernel.apply(
                    prepared,
                    np.full(len(selected), key, dtype=np.int64),
                    table,
                    selected,
                    expected,
                    context,
                )
                # Repeated reversed/non-contiguous applies may not consume or mutate
                # the payload. Empty batches must also be harmless.
                for subset in (
                    np.arange(len(selected)),
                    np.arange(len(selected))[::-2],
                    np.empty(0, dtype=np.int64),
                ):
                    actual = states[subset].copy()
                    actual_mask = case.kernel.apply(
                        bound,
                        np.zeros(len(subset), dtype=np.int64),
                        table,
                        selected[subset],
                        actual,
                        context,
                    )
                    if len(subset):
                        allowance = (
                            case.budget / 8
                            if prefix_input and case.kernel.name == "dyson"
                            else case.budget
                        )
                        assert_budget(actual, expected[subset], allowance)
                    np.testing.assert_array_equal(actual_mask, expected_mask[subset])
