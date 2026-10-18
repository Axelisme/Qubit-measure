"""Behavior at the documented numeric declaration and legacy provider seams."""

from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.simulate.evolution import Engine, EnvelopeSpec, ModelSource, SecularModel
from zcu_tools.simulate.evolution.declarations import (
    AlgorithmIdentity,
    Eligibility,
    GuardDeclaration,
    KernelDeclaration,
    MemoryDeclaration,
    ReuseDeclaration,
    StageCosts,
)
from zcu_tools.simulate.evolution.kernel import GuardPreparation, GuardResult
from zcu_tools.simulate.evolution.provider import (
    LegacyKernelAdapter,
    LegacyProvider,
    Selection,
    segment,
)

from ._engine import Payload, RecordingKernel, batch, run, source, track


def declaration(name: str, *, builds: bool = False) -> KernelDeclaration:
    """Declare a synthetic family, without inferring metadata from its object."""
    return KernelDeclaration(
        AlgorithmIdentity(name, "test-v1"),
        Eligibility(max_tones=0),
        GuardDeclaration(
            ("model-build",) if builds else ("no-build",), builds_model=builds
        ),
        "exact-secular",
        ReuseDeclaration(f"test/{name}/v1"),
    )


def test_candidate_discovery_excludes_impossible_rows_without_guards_or_builds():
    class Unguardable(RecordingKernel):
        def guard(self, table, rows, source, budget):
            pytest.fail("cheap eligibility must not run a numerical guard")

    idle = Unguardable("t4")
    driven = Unguardable("dyson")
    provider = LegacyProvider(
        (
            LegacyKernelAdapter(idle, declaration("t4")),
            LegacyKernelAdapter(
                driven,
                replace(
                    declaration("dyson"), eligibility=Eligibility(1, 1, "constant")
                ),
            ),
        )
    )
    geometry = segment(
        source(),
        (replace(batch(1), tracks=(track(1, EnvelopeSpec("constant")),)),),
    )
    rows = np.arange(len(geometry.table.timeline), dtype=np.int64)
    candidates = provider.eligible(geometry.table, rows)
    assert candidates["t4"].size == 0
    np.testing.assert_array_equal(candidates["dyson"], rows)
    assert idle.preparations == driven.preparations == 0


def test_default_provider_entry_prepares_and_applies_exact_idle_decay():
    initial = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
    model = SecularModel(
        (0,),
        "two-level-decay",
        np.array([0.0, 0.25]),
        np.eye(2, dtype=complex),
        (),
        np.array([[0.0, 0.2], [0.0, 0.0]]),
        np.zeros((2, 2)),
    )
    src = ModelSource(initial, lambda level: replace(model, level=level))
    provider = LegacyProvider()
    geometry = segment(src, (batch(1, duration=2.0),))
    dispatch = provider.classify(geometry, src, 1e-3)
    np.testing.assert_array_equal(dispatch.names, ["t4"])
    kernel = provider.entry(str(dispatch.names[0])).kernel
    rows = np.array([0], dtype=np.int64)
    prepared = kernel.prepare(
        kernel.prepare_keys(geometry.table, rows), geometry.table, rows, src, 1e-3
    )
    states = initial[None].copy()
    exceeded = kernel.apply(
        prepared, np.array([0], dtype=np.int64), geometry.table, rows, states, 1e-3
    )
    excited = 0.5 * np.exp(-0.4)
    coherence = -0.5 * np.exp(-0.2)
    np.testing.assert_allclose(
        states[0], [[1 - excited, coherence], [coherence, excited]], atol=1e-14
    )
    np.testing.assert_array_equal(exceeded, [False])
    with pytest.raises(KeyError, match="unknown-family"):
        provider.entry("unknown-family")


def test_active_envelopes_control_eligibility_and_legacy_dispatch_in_any_row_order():
    constant = EnvelopeSpec("constant")
    shaped = EnvelopeSpec(
        "sampled", np.array([0.2, 1.0], dtype=complex), np.array([0.0, 3.0])
    )
    # No drive, one constant, one shaped, two constant, shaped in either slot.
    first = replace(
        track(6, constant),
        envelopes=(constant, shaped),
        envelope_index=np.array([0, 0, 1, 0, 1, 0]),
        active=np.array([False, True, True, True, True, True]),
    )
    second = replace(
        track(6, constant),
        envelopes=(constant, shaped),
        envelope_index=np.array([0, 0, 0, 0, 0, 1]),
        active=np.array([False, False, False, True, True, True]),
    )
    batches = (replace(batch(6), tracks=(first, second)),)
    entries = tuple(
        LegacyKernelAdapter(
            RecordingKernel(name), replace(declaration(name), eligibility=eligibility)
        )
        for name, eligibility in (
            ("t4", Eligibility(max_tones=0)),
            ("dyson", Eligibility(1, 2, "constant")),
            ("floquet", Eligibility(2, 2, "constant")),
            ("shaped_sambe", Eligibility(1, 2, "shaped")),
            ("dop853", Eligibility()),
        )
    )
    provider = LegacyProvider(entries)
    src = source()
    geometry = segment(src, batches)
    # The sampled envelope also occupies inactive track slots in this fixture.
    for rows, expected_constant, expected_shaped in (
        (np.arange(6), [1, 3], [2, 4, 5]),
        (np.arange(6)[::-1], [3, 1], [5, 4, 2]),
        (np.array([5, 0, 3, 2]), [3], [5, 2]),
        (np.array([], dtype=np.int64), [], []),
    ):
        candidates = provider.eligible(geometry.table, rows)
        np.testing.assert_array_equal(candidates["dyson"], expected_constant)
        np.testing.assert_array_equal(candidates["shaped_sambe"], expected_shaped)
    expected = ["t4", "dyson", "shaped_sambe", "floquet", "dop853", "dop853"]
    dispatch = provider.classify(geometry, src, 1e-3)
    np.testing.assert_array_equal(dispatch.names, expected)
    for rows in (np.arange(6)[::-1], np.array([5, 0, 3, 2])):
        subset = provider.classify(
            geometry,
            src,
            1e-3,
            selection=Selection(rows, dispatch.shaped, dispatch.rejected),
        )
        np.testing.assert_array_equal(subset.names[rows], np.array(expected)[rows])
    result = run(Engine(1, 0, kernels=tuple(e.kernel for e in entries)), src, batches)
    assert result.paths.segments_by_kernel == {
        "t4": 1,
        "dyson": 1,
        "shaped_sambe": 1,
        "floquet": 1,
        "dop853": 2,
    }
    np.testing.assert_array_equal(
        result.densities, np.repeat(src.initial_density[None], 6, axis=0)
    )


@pytest.mark.parametrize("value", [-1.0, np.inf, np.nan])
def test_invalid_stage_cost_fails_before_candidate_competition(value):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        StageCosts(apply=value)


@pytest.mark.parametrize("value", [-1, 1.5])
def test_invalid_memory_charge_fails_at_declaration_boundary(value):
    with pytest.raises(ValueError, match="nonnegative integer"):
        MemoryDeclaration(resident=value)


def test_legacy_adapter_requires_matching_identity_and_handoff():
    with pytest.raises(ValueError, match="identity"):
        LegacyKernelAdapter(RecordingKernel("t4"), declaration("other"))
    with pytest.raises(ValueError, match="handoff"):
        LegacyKernelAdapter(RecordingKernel("t4"), declaration("t4", builds=True))


def test_legacy_provider_preserves_terminal_and_missing_family_errors():
    geometry = segment(source(), (batch(1),))
    provider = LegacyProvider(
        (LegacyKernelAdapter(RecordingKernel("t4", accepts=False), declaration("t4")),)
    )
    with pytest.raises(RuntimeError, match="no valid kernel for segment 0"):
        provider.classify(geometry, source(), 1e-3)
    with pytest.raises(RuntimeError, match="t4"):
        LegacyProvider(()).classify(geometry, source(), 1e-3)


@pytest.mark.parametrize("count", [1, 2])
def test_legacy_provider_retains_shared_and_deferred_guard_products(count):
    class BuildingIdle(RecordingKernel):
        guard_builds = True

        def guard_prepared(self, table, rows, source, budget):
            keys = self.prepare_keys(table, rows[:1])
            self.preparations += 1
            return GuardResult(
                np.ones(len(rows), dtype=bool),
                (GuardPreparation(keys, Payload(1)),),
            )

        def prepare(self, keys, table, rows, source, budget):
            pytest.fail("accepted guard payload must not be rebuilt")

    kernel = BuildingIdle("t4")
    provider = LegacyProvider(
        (LegacyKernelAdapter(kernel, declaration("t4", builds=True)),)
    )
    src = source()
    geometry = segment(src, (batch(count),))
    result = provider.classify(geometry, src, 1e-3)
    if count == 1:
        assert kernel.preparations == 0
        np.testing.assert_array_equal(result.deferred, [True])
        result = provider.classify(
            geometry,
            src,
            1e-3,
            selection=Selection(np.array([0]), result.shaped, result.rejected),
        )
    assert kernel.preparations == 1
    np.testing.assert_array_equal(result.names, ["t4"] * count)
    handoff = result.preparations["t4"][0]
    states = np.repeat(src.initial_density[None], count, axis=0)
    before = states.copy()
    exceeded = kernel.apply(
        handoff.prepared,
        np.zeros(count, dtype=np.int64),
        geometry.table,
        np.arange(count, dtype=np.int64),
        states,
        1e-3,
    )
    np.testing.assert_array_equal(states, before)
    np.testing.assert_array_equal(exceeded, np.zeros(count, dtype=bool))
