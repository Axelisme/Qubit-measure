"""Explicit profile admission and developer whole-acquire evaluation contracts."""

from dataclasses import replace
from threading import Event

import numpy as np
import pytest
from zcu_tools.simulate.evolution import Cancelled, Engine, EnvelopeSpec
from zcu_tools.simulate.evolution.calibration import (
    AcquireEvidence,
    CalibratedConstantProfile,
    ConstantWorkload,
    constant_kernel_revision,
)
from zcu_tools.simulate.evolution.constant import ConstantPlanner
from zcu_tools.simulate.evolution.declarations import MemoryDeclaration, StageCosts
from zcu_tools.simulate.evolution.kernels.dop853 import DOP853Kernel
from zcu_tools.simulate.evolution.profiles import (
    FamilyProfile,
    OneCoreProfile,
    ProfileDomain,
    ProfileProvenance,
)
from zcu_tools.simulate.evolution.provider import LegacyProvider
from zcu_tools.simulate.evolution.shadow import ShadowRequest

from ._engine import batch, source, track


@pytest.fixture
def scenario():
    src = source()
    b = replace(
        batch(2, duration=0.1),
        tracks=(
            replace(
                track(2, EnvelopeSpec("constant"), duration=0.1),
                frequency_ghz=np.full(2, 0.3),
                amplitude=np.full(2, 0.001),
            ),
        ),
    )
    provider = LegacyProvider()
    estimates = OneCoreProfile(
        "synthetic-contract-only",
        "v1",
        ProfileDomain("exact-fixture", (3, 3), (1e-3, 1e-3)),
        ProfileProvenance(
            "test", (), "test", "test", "contract fixture, not shipped evidence"
        ),
        tuple(
            FamilyProfile(
                provider.entry(name).declaration.identity,
                StageCosts(0, 0, 0, 0 if name == "dop853" else 100, 0, 0, 0),
                MemoryDeclaration(0, 0, 0, 0),
            )
            for name in ("t4", "dyson", "floquet_period", "t6", "dop853")
        ),
    )
    planner = ConstantPlanner()
    try:
        pinned = planner.plan(
            src, (b,), estimates, ShadowRequest(3, 1e-3, 1, 10000000, 288)
        )
    finally:
        planner.close()
    workload = ConstantWorkload(pinned.input_digest, 1, 100000)
    pending = CalibratedConstantProfile(
        estimates, constant_kernel_revision(), (workload,)
    )
    evidence = tuple(
        AcquireEvidence(workload, temperature, 0.02, 0.01, 0.001, 0.001, "test-only")
        for temperature in ("cold", "warm")
    )
    return src, b, pending, replace(pending, acquire_evidence=evidence)


def test_auto_without_profile_preserves_legacy(scenario):
    src, b, _, _ = scenario
    engine = Engine(1, 100000)
    try:
        legacy = engine.evolve(src, (b,), cancel=Event(), execution="legacy")
        automatic = engine.evolve(
            src, (b,), cancel=Event(), execution="auto", constant_profile=None
        )
        np.testing.assert_array_equal(automatic.densities, legacy.densities)
        assert automatic.paths.segments_by_kernel == legacy.paths.segments_by_kernel
    finally:
        engine.close()


def test_pending_is_evaluated_only_by_developer_entry_and_complete_profile_is_opt_in(
    scenario,
):
    src, b, pending, admitted = scenario
    engine = Engine(1, 100000)
    try:
        legacy = engine.evolve(src, (b,), cancel=Event(), constant_profile=pending)
        evaluated = engine.evaluate_constant_profile(
            src, (b,), profile=pending, cancel=Event()
        )
        production = engine.evolve(src, (b,), cancel=Event(), constant_profile=admitted)
        assert evaluated.paths.segments_by_kernel == {"dop853": 2}
        assert production.paths.segments_by_kernel == evaluated.paths.segments_by_kernel
        np.testing.assert_array_equal(production.densities, evaluated.densities)
        assert legacy.paths.segments_by_kernel != production.paths.segments_by_kernel
        for execution in ("legacy", "legacy-plan"):
            result = engine.evolve(
                src,
                (b,),
                cancel=Event(),
                execution=execution,
                constant_profile=admitted,
            )
            np.testing.assert_array_equal(result.densities, legacy.densities)
    finally:
        engine.close()


@pytest.mark.parametrize(
    "defect",
    [
        "revision",
        "source",
        "input",
        "controls",
        "workers",
        "cache",
        "memory",
        "failed-evidence",
    ],
)
def test_nonadmission_is_before_execution_and_developer_entry_rejects(scenario, defect):
    src, b, pending, admitted = scenario
    workers, cache = 1, 100000
    if defect == "revision":
        pending = replace(pending, kernel_source_digest="stale")
        admitted = replace(admitted, kernel_source_digest="stale")
    elif defect == "source":
        from zcu_tools.simulate.evolution import ModelSource

        model = src.model((0,))
        src = ModelSource(
            src.initial_density,
            lambda level: replace(
                model, level=level, energies_ghz=model.energies_ghz * 1.01
            ),
        )
    elif defect == "input":
        from zcu_tools.simulate.evolution import ModelSource

        src = ModelSource(np.eye(3, dtype=np.complex128) / 3, src.model)
    elif defect == "controls":
        b = replace(b, tracks=(replace(b.tracks[0], phase_rad=np.full(2, 0.73)),))
    elif defect == "workers":
        workers = 2
    elif defect == "cache":
        cache = 0
    elif defect == "memory":
        pending = replace(pending, large_allocation_allowance=0)
        admitted = replace(admitted, large_allocation_allowance=0)
    else:
        bad = tuple(
            replace(e, candidate_seconds=0.04) for e in admitted.acquire_evidence
        )
        pending = admitted = replace(admitted, acquire_evidence=bad)
    engine = Engine(workers, cache)
    try:
        expected = engine.evolve(src, (b,), cancel=Event(), execution="legacy")
        actual = engine.evolve(src, (b,), cancel=Event(), constant_profile=admitted)
        np.testing.assert_array_equal(actual.densities, expected.densities)
        with pytest.raises(ValueError, match="rejected"):
            engine.evaluate_constant_profile(src, (b,), profile=pending, cancel=Event())
    finally:
        engine.close()


@pytest.mark.parametrize("stage", ["guard", "prepare", "apply"])
def test_evaluation_propagates_numeric_failure_without_retry(
    scenario, monkeypatch, stage
):
    src, b, pending, _ = scenario
    calls = []

    def fail(*args, **kwargs):
        calls.append(stage)
        raise RuntimeError("numeric failure")

    monkeypatch.setattr(DOP853Kernel, stage, fail)
    engine = Engine(1, 100000)
    try:
        with pytest.raises(RuntimeError, match="numeric failure"):
            engine.evaluate_constant_profile(src, (b,), profile=pending, cancel=Event())
        assert calls == [stage]
    finally:
        engine.close()


def test_evaluation_cancellation_returns_no_partial_result(scenario):
    src, b, pending, _ = scenario
    engine = Engine(1, 100000)
    cancel = Event()
    cancel.set()
    try:
        with pytest.raises(Cancelled):
            engine.evaluate_constant_profile(src, (b,), profile=pending, cancel=cancel)
    finally:
        engine.close()


def test_unknown_memory_is_not_fabricated_and_t6_channel_allowance_is_separate(
    scenario,
):
    src, b, pending, _ = scenario
    estimates = replace(
        pending.estimates,
        families=tuple(
            replace(f, memory=MemoryDeclaration()) for f in pending.estimates.families
        ),
    )
    planner = ConstantPlanner()
    try:
        pinned = planner.plan(
            src, (b,), estimates, ShadowRequest(3, 1e-3, 1, 10000000, 288)
        )
        assert pinned.report.selected.required_bytes is None
        assert pinned.report.incumbent.large_allocation_bytes > 0
        with pytest.raises(ValueError, match="allowance"):
            planner.plan(src, (b,), estimates, ShadowRequest(3, 1e-3, 1, 0, 288))
    finally:
        planner.close()


def test_idle_candidate_preserves_complete_exact_secular_cover(scenario):
    src, _, pending, _ = scenario
    b = batch(2, duration=0.1)
    estimates = replace(
        pending.estimates,
        families=tuple(
            replace(
                f, stages=replace(f.stages, apply=0 if f.identity.name == "t4" else 100)
            )
            for f in pending.estimates.families
        ),
    )
    planner = ConstantPlanner()
    try:
        pinned = planner.plan(
            src, (b,), estimates, ShadowRequest(3, 1e-3, 1, 10000000, 288)
        )
    finally:
        planner.close()
    profile = replace(
        pending,
        estimates=estimates,
        workloads=(ConstantWorkload(pinned.input_digest, 1, 100000),),
    )
    engine = Engine(1, 100000)
    try:
        expected = engine.evolve(src, (b,), cancel=Event(), execution="legacy")
        actual = engine.evaluate_constant_profile(
            src, (b,), profile=profile, cancel=Event()
        )
        assert actual.paths.segments_by_kernel == {"t4": 2}
        np.testing.assert_array_equal(actual.points, expected.points)
        np.testing.assert_allclose(
            actual.densities, expected.densities, atol=1e-14, rtol=0
        )
    finally:
        engine.close()


@pytest.mark.parametrize("revision", ["numeric_revision", "planner_revision", "dependencies"])
def test_profile_execution_revision_is_checked_before_candidate_guards(scenario, revision):
    src, b, pending, admitted = scenario
    if revision == "dependencies":
        estimates = replace(pending.estimates, provenance=replace(
            pending.estimates.provenance, dependencies=("numpy==0",)))
        pending = replace(pending, estimates=estimates)
        admitted = replace(admitted, estimates=estimates)
    else:
        pending = replace(pending, **{revision: "unsupported"})
        admitted = replace(admitted, **{revision: "unsupported"})
    engine = Engine(1, 100000)
    try:
        expected = engine.evolve(src, (b,), cancel=Event(), execution="legacy")
        actual = engine.evolve(src, (b,), cancel=Event(), constant_profile=admitted)
        np.testing.assert_array_equal(actual.densities, expected.densities)
        with pytest.raises(ValueError, match="rejected"):
            engine.evaluate_constant_profile(src, (b,), profile=pending, cancel=Event())
    finally:
        engine.close()
