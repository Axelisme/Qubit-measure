"""Experimental constant covers: neutral scheduling, checked replay and numerics."""

from collections import Counter
from dataclasses import replace
from threading import Event

import numpy as np
import pytest
from zcu_tools.simulate.evolution import Cancelled, EnvelopeSpec, ModelSource
from zcu_tools.simulate.evolution.constant import ConstantPlanner
from zcu_tools.simulate.evolution.declarations import MemoryDeclaration, StageCosts
from zcu_tools.simulate.evolution.profiles import (
    FamilyProfile,
    OneCoreProfile,
    ProfileDomain,
    ProfileProvenance,
)
from zcu_tools.simulate.evolution.provider import LegacyKernelAdapter, LegacyProvider, segment
from zcu_tools.simulate.evolution.shadow import ShadowRequest

from ._engine import RecordingKernel, batch, source, track
from .contract.kernel_suite import KernelCase, reference


NAMES = ("dyson", "floquet_period", "t6", "dop853", "t4")


def entries(*kernels: RecordingKernel):
    provider = LegacyProvider()
    return tuple(
        LegacyKernelAdapter(kernel, provider.entry(kernel.name).declaration)
        for kernel in kernels
    )


def costs(*, preferred: str = "dyson", competition: bool = False):
    provider = LegacyProvider()
    estimates = []
    for name in NAMES:
        build, apply = (0, 100)
        if competition and name in ("dyson", "dop853"):
            build, apply = (6, 1) if name == "dyson" else (0, 5)
        elif name == preferred:
            build, apply = 0, 1
        estimates.append(
            FamilyProfile(
                provider.entry(name).declaration.identity,
                StageCosts(0, build, 0, apply, 0, 0, 0),
                MemoryDeclaration(0, 0, 0, 0),
            )
        )
    return OneCoreProfile(
        "neutral-not-production", "v1",
        ProfileDomain("tiny-constant", (2, 4), (1e-6, 1e-2)),
        ProfileProvenance("synthetic", (), "logical capacity only", "neutral", "unmeasured"),
        tuple(estimates),
    )


def request(n: int, workers: int = 1):
    return ShadowRequest(3, 1e-3, workers, 100000, n * 3 * 3 * 16)


def driven(n: int = 2):
    return replace(batch(n, duration=1), tracks=(track(n, EnvelopeSpec("constant"), duration=1),))


@pytest.mark.parametrize("workers,expected", [(1, "dyson"), (2, None), (6, "dop853")])
def test_worker_aware_cover_is_deterministic_and_replay_does_not_reassign(workers, expected):
    kernels = tuple(RecordingKernel(name) for name in NAMES)
    src, b = source(), driven(6)
    planner = ConstantPlanner(entries(*kernels), cache_bytes=10000)
    cold = ConstantPlanner(entries(*kernels), cache_bytes=0)
    try:
        pinned = planner.plan(src, (b,), costs(competition=True), request(6, workers))
        repeated = planner.plan(src, (b,), costs(competition=True), request(6, workers))
        assert pinned == repeated
        selected_names = [c.declaration.identity.name for c in pinned.report.selected.cover]
        assert sorted(c.row for c in pinned.report.selected.cover) == list(range(6))
        if expected is None:
            # All-Dyson is a feasible 6 + 6/2 = 9 upper bound, not an optimum claim.
            assert pinned.report.selected.wall_seconds <= 9
        else:
            assert selected_names == [expected] * 6
        assert len(pinned.report.incumbent.cover) == 6
        assert all(c.declaration.identity.name == "t6" for c in pinned.report.incumbent.cover)
        assert all(c.accepted and c.input_digest == pinned.input_digest for c in pinned.certificates)
        assert not pinned.report.production_safe
        first = planner.replay(pinned, src, (b,), cancel=Event())
        warm = planner.replay(pinned, src, (b,), cancel=Event())
        fresh = cold.replay(pinned, src, (b,), cancel=Event())
        for result in (first, warm, fresh):
            assert result.paths.segments_by_kernel == Counter(selected_names)
            np.testing.assert_array_equal(result.densities, [src.initial_density] * 6)
    finally:
        planner.close()
        cold.close()


@pytest.mark.parametrize("family", NAMES)
def test_pinned_builtin_kernel_matches_tiny_independent_reference(family):
    src = source()
    b = batch(2, duration=1) if family == "t4" else driven()
    if family != "t4":
        b = replace(b, tracks=(replace(b.tracks[0], amplitude=np.full(2, 0.001)),))
    planner = ConstantPlanner()
    try:
        pinned = planner.plan(src, (b,), costs(preferred=family), request(2))
        result = planner.replay(pinned, src, (b,), cancel=Event())
        assert result.paths.segments_by_kernel == {family: 2}
        geometry = segment(src, (b,))
        case = KernelCase(LegacyProvider().entry(family).kernel, src, geometry.table, np.array([src.initial_density] * 2), 1e-3)
        for row in range(2):
            np.testing.assert_allclose(result.densities[row], reference(case, row, src.initial_density), rtol=0, atol=1e-3)
    finally:
        planner.close()


@pytest.mark.parametrize("defect", ["controls", "density", "model", "numeric", "planner", "kernel", "certificate", "coverage"])
def test_replay_rejects_mismatched_inputs_or_unsupported_revisions_before_apply(defect):
    kernels = tuple(RecordingKernel(name) for name in NAMES)
    src, b = source(), driven()
    planner = ConstantPlanner(entries(*kernels))
    try:
        pinned = planner.plan(src, (b,), costs(), request(2))
        replay_planner = planner
        if defect == "controls":
            b = replace(b, tracks=(replace(b.tracks[0], phase_rad=np.ones(2)),))
        elif defect == "density":
            src = ModelSource(np.eye(3, dtype=complex) / 3, src.model)
        elif defect == "model":
            old = src
            src = ModelSource(old.initial_density, lambda level: replace(old.model(level), energies_ghz=np.array([0, 0.7, 1.4])))
        elif defect == "numeric":
            pinned = replace(pinned, numeric_revision="unsupported")
        elif defect == "planner":
            pinned = replace(pinned, planner_revision="unsupported")
        elif defect == "kernel":
            changed = entries(*kernels)
            changed = (replace(changed[0], declaration=replace(changed[0].declaration, identity=replace(changed[0].declaration.identity, revision="new-kernel"))), *changed[1:])
            replay_planner = ConstantPlanner(changed)
        elif defect == "certificate":
            pinned = replace(pinned, certificates=())
        else:
            pinned = replace(pinned, report=replace(pinned.report, selected=replace(pinned.report.selected, cover=pinned.report.selected.cover[:1])))
        with pytest.raises(ValueError, match="revision|input|certificate|cover"):
            replay_planner.replay(pinned, src, (b,), cancel=Event())
        assert not any(kernel.calls for kernel in kernels)
        if replay_planner is not planner:
            replay_planner.close()
    finally:
        planner.close()


def test_guard_rejection_preserves_safe_cover_and_terminal_rejection_is_fatal():
    rejecting = RecordingKernel("dyson", accepts=False)
    terminal = RecordingKernel("t6")
    planner = ConstantPlanner(entries(rejecting, terminal))
    try:
        pinned = planner.plan(source(), (driven(),), costs(), request(2))
        assert all(not c.accepted for c in pinned.certificates if c.family == "dyson")
        result = planner.replay(pinned, source(), (driven(),), cancel=Event())
        assert result.paths.segments_by_kernel == {"t6": 2}
        terminal.accepts = False
        with pytest.raises(RuntimeError, match="no valid kernel"):
            planner.plan(source(), (driven(),), costs(), request(2))
    finally:
        planner.close()


@pytest.mark.parametrize("stage", ["guard", "prepare", "apply"])
def test_numeric_failures_propagate_without_retry(stage):
    failure = ArithmeticError("constant numerical failure")

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

    fallback = RecordingKernel("t6")
    planner = ConstantPlanner(entries(Failing("dyson"), fallback))
    try:
        with pytest.raises(ArithmeticError) as caught:
            pinned = planner.plan(source(), (driven(),), costs(), request(2))
            planner.replay(pinned, source(), (driven(),), cancel=Event())
        assert caught.value is failure
        assert not fallback.calls
        assert fallback.preparations == 0
    finally:
        planner.close()


def test_inline_replay_warns_once_without_repair_and_honors_cancellation():
    class Exceeded(RecordingKernel):
        def apply(self, prepared, key_index, table, rows, states, budget):
            states[:, 0, 0] += 0.125
            return np.ones(len(rows), dtype=bool)

    planner = ConstantPlanner(entries(Exceeded("t4")))
    try:
        src, b = source(), batch(2)
        pinned = planner.plan(src, (b,), costs(preferred="t4"), request(2))
        with pytest.warns(RuntimeWarning, match="uncorrected") as caught:
            result = planner.replay(pinned, src, (b,), cancel=Event())
        assert len(caught) == 1
        assert result.paths.exceeded_by_kernel == {"t4": 2}
        assert result.paths.max_trace_error == pytest.approx(0.125)
        expected = src.initial_density.copy()
        expected[0, 0] += 0.125
        np.testing.assert_array_equal(result.densities, [expected] * 2)
        cancel = Event()
        cancel.set()
        with pytest.raises(Cancelled):
            planner.replay(pinned, src, (b,), cancel=cancel)
    finally:
        planner.close()


@pytest.mark.parametrize("geometry", ["multi-interval", "shaped", "two-tone", "empty"])
def test_unsupported_geometry_cannot_silently_skip_intervals(geometry):
    b = driven()
    if geometry == "multi-interval":
        b = replace(b, tracks=(replace(b.tracks[0], start_ns=np.full(2, 0.5)),))
    elif geometry == "shaped":
        b = replace(b, tracks=(track(2, EnvelopeSpec("sampled", samples=np.array([0, 1]), sample_ns=np.array([0, 1])), duration=1),))
    elif geometry == "two-tone":
        b = replace(b, tracks=b.tracks * 2)
    planner = ConstantPlanner()
    try:
        with pytest.raises(ValueError, match="constant|interval|nonempty"):
            planner.plan(source(), () if geometry == "empty" else (b,), costs(), request(2))
    finally:
        planner.close()


def test_production_admission_separates_revision_mismatch_from_uncalibrated_evidence():
    profile = costs()
    current = tuple(family.identity for family in profile.families)
    matching = profile.production_admission(current)
    assert matching.execution == "legacy"
    assert matching.revisions_match
    assert "uncalibrated" in matching.reason
    changed = (replace(current[0], revision="new-kernel"), *current[1:])
    mismatch = profile.production_admission(changed)
    assert mismatch.execution == "legacy"
    assert not mismatch.revisions_match
    assert "revision" in mismatch.reason
    assert "dyson" in mismatch.reason
