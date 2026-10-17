"""Engine-owned legacy covers preserve scheduler behavior without cost assignment.

These migration comparisons retire with the legacy adapter. Independent kernel
and engine contracts remain the authority after legacy dispatch is removed.
"""

from dataclasses import replace
from threading import Event
from typing import Literal
import warnings

import numpy as np
import pytest
from zcu_tools.simulate.evolution import Cancelled, Engine, EnvelopeSpec, FluxSchedule
from zcu_tools.simulate.evolution.kernel import GuardPreparation, GuardResult

from ._engine import Payload, RecordingKernel, batch, source, track


@pytest.fixture(params=["legacy", "legacy-plan"])
def execution(request) -> Literal["legacy", "legacy-plan"]:
    return request.param


def test_cover_preserves_all_legacy_domains(execution):
    shaped = EnvelopeSpec(
        "sampled", samples=np.array([0, 1, 0]), sample_ns=np.array([0, 1, 3])
    )
    cases = [
        ((), "t4"),
        ((EnvelopeSpec("constant"),), "dyson"),
        ((shaped,), "shaped_sambe"),
        ((EnvelopeSpec("constant"),) * 2, "floquet"),
        ((shaped, EnvelopeSpec("constant")), "dop853"),
        ((EnvelopeSpec("constant"),) * 3, "dop853"),
    ]
    for envelopes, expected in cases:
        kernels = tuple(
            RecordingKernel(name)
            for name in ("t4", "dyson", "shaped_sambe", "floquet", "dop853")
        )
        b = replace(batch(2), tracks=tuple(track(2, env) for env in envelopes))
        engine = Engine(1, 0, kernels=kernels)
        try:
            result = engine.evolve(source(), (b,), cancel=Event(), execution=execution)
        finally:
            engine.close()
        np.testing.assert_array_equal(result.points, [0, 1])
        np.testing.assert_array_equal(result.densities, [source().initial_density] * 2)
        assert result.paths.segments_by_kernel == {expected: 2}
        assert result.paths.points_by_class == {expected: 2}
        assert result.paths.max_trace_error == 0


@pytest.mark.parametrize("driven", [False, True])
def test_flux_cover_keeps_basis_boundaries(execution, driven):
    b = replace(
        batch(1),
        flux=FluxSchedule(
            np.array([0, 2]), np.array([0.0, 1.0]), np.array([[0], [1]])
        ),
        tracks=(track(1, EnvelopeSpec("constant")),) if driven else (),
    )
    family = "flux_dop853" if driven else "t4"
    kernel = RecordingKernel(family)
    engine = Engine(1, 0, kernels=(kernel,))
    try:
        result = engine.evolve(source(), (b,), cancel=Event(), execution=execution)
    finally:
        engine.close()
    assert result.paths.segments_by_kernel == {family: 2}
    np.testing.assert_array_equal(result.densities[0], source().initial_density)


class GuardTable(RecordingKernel):
    """A numerical builder accepting only even points, with observable handoff."""

    guard_builds = True

    def __init__(self, shared):
        super().__init__("dyson")
        self.shared = shared
        self.guarded: list[int] = []
        self.products: list[Payload] = []
        self.applied: list[int] = []

    def prepare_keys(self, table, rows):
        keys = np.empty(len(rows), dtype=[("group", "i8")])
        keys["group"] = 0 if self.shared else table.timeline[rows]
        return keys

    def guard_prepared(self, table, rows, source, budget):
        self.guarded.extend(table.timeline[rows].tolist())
        keys = np.unique(self.prepare_keys(table, rows))
        payload = Payload(len(keys))
        self.products.append(payload)
        return GuardResult(
            table.timeline[rows] % 2 == 0, (GuardPreparation(keys, payload),)
        )

    def prepare(self, keys, table, rows, source, budget):
        raise AssertionError("accepted guard products must not be rebuilt")

    def apply(self, prepared, key_index, table, rows, states, budget):
        assert any(prepared is product for product in self.products)
        self.applied.extend(table.timeline[rows].tolist())
        return super().apply(prepared, key_index, table, rows, states, budget)


@pytest.mark.parametrize("shared", [False, True])
def test_guard_product_partial_acceptance_and_deferred_fallback(execution, shared):
    builder = GuardTable(shared)
    fallback = RecordingKernel("floquet_period")
    b = replace(batch(4), tracks=(track(4, EnvelopeSpec("constant")),))
    engine = Engine(1, 0, kernels=(builder, fallback))
    try:
        result = engine.evolve(source(), (b,), cancel=Event(), execution=execution)
    finally:
        engine.close()
    assert sorted(builder.guarded) == [0, 1, 2, 3]
    assert sorted(builder.applied) == [0, 2]
    assert result.paths.segments_by_kernel == {"dyson": 2, "floquet_period": 2}
    np.testing.assert_array_equal(result.densities, [source().initial_density] * 4)


class PrefixStep(RecordingKernel):
    def apply(self, prepared, key_index, table, rows, states, budget):
        super().apply(prepared, key_index, table, rows, states, budget)
        states[:, 0, 0] += 0.125
        return np.ones(len(rows), dtype=bool)


def test_prefix_adoption_keeps_diagnostics_exceedance_and_uncorrected_trace(execution):
    prefix = PrefixStep("shaped_sambe")
    shaped = EnvelopeSpec(
        "sampled", samples=np.array([0, 1, 0]), sample_ns=np.array([0, 0.5, 1])
    )
    b = replace(
        batch(3),
        tracks=(
            track(3, shaped, duration=1),
            track(3, EnvelopeSpec("constant"), start=1, duration=1),
        ),
    )
    engine = Engine(
        1, 0, kernels=(prefix, RecordingKernel("dyson"), RecordingKernel("t4"))
    )
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = engine.evolve(source(), (b,), cancel=Event(), execution=execution)
    finally:
        engine.close()
    assert sum(len(rows) for _, rows, _ in prefix.calls) == 1
    assert result.paths.segments_by_kernel == {"shaped_sambe": 3, "dyson": 3, "t4": 3}
    assert result.paths.points_by_class == {"shaped_sambe": 3, "dyson": 3, "t4": 3}
    assert result.paths.exceeded_by_kernel == {"shaped_sambe": 3}
    assert result.paths.max_trace_error == pytest.approx(0.125)
    expected = source().initial_density.copy()
    expected[0, 0] += 0.125
    np.testing.assert_array_equal(result.densities, [expected] * 3)
    assert len(caught) == 1
    assert "3 segments (shaped_sambe: 3)" in str(caught[0].message)


@pytest.mark.parametrize("family,driven", [("t4", False), ("t6", True)])
def test_terminal_and_missing_registry_errors(execution, family, driven):
    b = replace(
        batch(1), tracks=(track(1, EnvelopeSpec("constant")),) if driven else ()
    )
    kernels = (
        RecordingKernel("dyson", accepts=False),
        RecordingKernel("floquet_period", accepts=False),
        RecordingKernel(family, accepts=False),
        RecordingKernel("dop853"),
    )
    engine = Engine(1, 0, kernels=kernels)
    try:
        with pytest.raises(RuntimeError, match="no valid kernel"):
            engine.evolve(source(), (b,), cancel=Event(), execution=execution)
    finally:
        engine.close()
    engine = Engine(1, 0, kernels=())
    try:
        with pytest.raises(RuntimeError, match="requires kernel (dyson|t4)"):
            engine.evolve(source(), (b,), cancel=Event(), execution=execution)
    finally:
        engine.close()


@pytest.mark.parametrize("stage", ["guard", "prepare", "apply"])
def test_numerical_exceptions_propagate_without_retry(execution, stage):
    failure = ArithmeticError("numeric collaborator failure")

    class Failing(RecordingKernel):
        def guard(self, table, rows, source, budget):
            if stage == "guard":
                raise failure
            return super().guard(table, rows, source, budget)

        def prepare(self, keys, table, rows, source, budget):
            if stage == "prepare":
                raise failure
            return super().prepare(keys, table, rows, source, budget)

        def apply(self, prepared, key_index, table, rows, states, budget):
            raise failure

    fallback = RecordingKernel("floquet_period")
    engine = Engine(1, 0, kernels=(Failing("dyson"), fallback))
    b = replace(batch(1), tracks=(track(1, EnvelopeSpec("constant")),))
    try:
        with pytest.raises(ArithmeticError) as caught:
            engine.evolve(source(), (b,), cancel=Event(), execution=execution)
        assert caught.value is failure
        assert not fallback.calls
        assert fallback.preparations == 0
    finally:
        engine.close()


def test_cancel_returns_no_partial_result_and_engine_remains_usable(execution):
    cancel = Event()
    kernel = RecordingKernel("t4", cancel=cancel)
    engine = Engine(1, 0, kernels=(kernel,), chunks_per_worker=4)
    try:
        with pytest.raises(Cancelled):
            engine.evolve(source(), (batch(4),), cancel=cancel, execution=execution)
        assert len(kernel.calls) == 1
        kernel.cancel = None
        cancel.clear()
        result = engine.evolve(source(), (batch(1),), cancel=cancel, execution=execution)
        np.testing.assert_array_equal(result.densities[0], source().initial_density)
    finally:
        engine.close()


@pytest.mark.parametrize("invalid", ["arbitrary", "shadow"])
def test_invalid_execution_fails_even_for_empty_work(invalid):
    engine = Engine(1, 0)
    try:
        with pytest.raises(ValueError, match="execution"):
            engine.evolve(source(), (), cancel=Event(), execution=invalid)
    finally:
        engine.close()
