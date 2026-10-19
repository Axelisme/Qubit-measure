"""Resident payload admission through Engine's injected kernel boundary."""

from contextlib import closing
from dataclasses import dataclass, replace
from threading import Event

import numpy as np
import pytest
from zcu_tools.simulate.evolution import Engine, EnvelopeSpec, FluxSchedule, ModelSource
from zcu_tools.simulate.evolution.cache import RESIDENT_METADATA_BYTES
from zcu_tools.simulate.evolution.kernel import GuardPreparation, GuardResult
from zcu_tools.simulate.evolution.kernels.single_tone_rwa import SingleToneRWAKernel

from ._engine import RecordingKernel, batch, source, track
from .contract.kernel_suite import synthetic_model


@pytest.mark.parametrize("frequency", [0.04, -0.04])
@pytest.mark.parametrize("mixed", [False, True])
def test_low_carrier_fallback_is_independent_of_warm_rwa_residents(frequency, mixed):
    model = synthetic_model()
    model = replace(
        model,
        rates_per_ns=np.zeros((3, 3)),
        dephasing_per_ns=np.zeros((3, 3)),
    )
    src = ModelSource(np.eye(3, dtype=np.complex128) / 3, lambda level: model)
    duration = 20.0

    def timelines(frequencies, point):
        pulse = replace(
            track(len(frequencies), EnvelopeSpec("constant"), duration=duration),
            frequency_ghz=np.array(frequencies),
            amplitude=np.full(len(frequencies), 0.001),
        )
        return (
            replace(
                batch(len(frequencies), duration=duration),
                tracks=(pulse,),
                points=np.arange(len(frequencies)) + point,
            ),
        )

    class CountedRWA(SingleToneRWAKernel):
        def __init__(self):
            self.preparations = 0

        def prepare(self, keys, table, rows, context):
            self.preparations += len(keys)
            return super().prepare(keys, table, rows, context)

    rwa = CountedRWA()
    fallback = RecordingKernel("floquet_period")
    kernels = (RecordingKernel("dyson", accepts=False), rwa, fallback)
    with closing(Engine(1, 1000000, kernels=kernels)) as engine:
        cold = engine.evolve(src, timelines([frequency], 0), cancel=Event())
        assert cold.paths.segments_by_kernel == {"floquet_period": 1}
        np.testing.assert_array_equal(cold.densities[0], src.initial_density)

        first = engine.evolve(src, timelines([0.13], 10), cancel=Event())
        warm = engine.evolve(src, timelines([0.13], 20), cancel=Event())
        assert first.paths.segments_by_kernel == {"single_tone_rwa": 1}
        assert warm.paths.segments_by_kernel == first.paths.segments_by_kernel
        assert rwa.preparations == 1, "the second acquire must borrow a resident"

        frequencies = [frequency, 0.13] if mixed else [frequency]
        try:
            result = engine.evolve(src, timelines(frequencies, 30), cancel=Event())
        except ValueError as error:
            raise AssertionError(
                "warm RWA residents must preserve low-carrier fallback"
            ) from error
        expected = {"floquet_period": 1}
        if mixed:
            expected["single_tone_rwa"] = 1
        assert result.paths.segments_by_kernel == expected
        np.testing.assert_array_equal(result.densities[0], cold.densities[0])
        assert rwa.preparations == 1


class CountedGuardKernel(RecordingKernel):
    """Identity builder with an explicit payload domain and row admission."""

    guard_builds = True

    def __init__(self):
        super().__init__("t4")
        self.builds = 0
        self.admissions = 0
        self.domain_covers = True

    def guard_prepared(self, table, rows, context):
        self.builds += 1
        keys = self.prepare_keys(table, rows, context)
        prepared = self.prepare(keys, table, rows, context)
        return GuardResult(
            np.ones(len(rows), dtype=bool), (GuardPreparation(keys, prepared),)
        )

    def admit_prepared(self, prepared, table, rows, context):
        self.admissions += 1
        if not self.domain_covers:
            return None
        keys = self.prepare_keys(table, rows, context)
        return GuardResult(
            np.full(len(rows), self.accepts), (GuardPreparation(keys, prepared),)
        )


@pytest.mark.parametrize("timelines", [1, 2])
def test_resident_guard_payload_skips_build_but_rechecks_rows_and_domain(timelines):
    kernel = CountedGuardKernel()
    src = source()
    # Include metadata for the intended resident working set.
    with closing(
        Engine(1, 4096 + 8 * RESIDENT_METADATA_BYTES, kernels=(kernel,))
    ) as engine:
        engine.evolve(src, (batch(2),), cancel=Event())
        cold_builds = kernel.builds
        # Distinct point IDs bypass exact-result memo while preserving prepare keys.
        result = engine.evolve(
            src,
            (replace(batch(timelines), points=np.arange(timelines) + 10),),
            cancel=Event(),
        )
        assert result.paths.segments_by_kernel == {"t4": timelines}
        assert kernel.builds == cold_builds, "resident domain hit must not build"
        assert kernel.admissions > 0

        kernel.domain_covers = False
        engine.evolve(
            src,
            (replace(batch(timelines), points=np.arange(timelines) + 20),),
            cancel=Event(),
        )
        assert kernel.builds > cold_builds, "domain miss must retain ordinary build"
        rebuilt = kernel.builds

        kernel.domain_covers = True
        kernel.accepts = False
        with pytest.raises(RuntimeError, match="no valid kernel"):
            engine.evolve(
                src,
                (replace(batch(timelines), points=np.arange(timelines) + 30),),
                cancel=Event(),
            )
        assert kernel.builds == rebuilt, "row rejection is not a domain miss"


@dataclass(frozen=True)
class DomainPayload:
    """Identity dynamics covering the listed duration keys, in ns."""

    durations: tuple[float, ...]

    @property
    def nbytes(self) -> int:
        return 16 * len(self.durations)


class OverlappingGuardKernel(RecordingKernel):
    """Build overlapping domains and record the rows offered for admission."""

    guard_builds = True

    def __init__(self):
        super().__init__("t4")
        self.decline_first = False
        self.admitted: list[tuple[tuple[float, ...], tuple[float, ...]]] = []
        self.built: list[tuple[float, ...]] = []

    def prepare_keys(self, table, rows, context):
        keys = np.empty(len(rows), dtype=[("duration", "f8")])
        keys["duration"] = table.duration_ns[rows]
        return keys

    def guard_prepared(self, table, rows, context):
        keys = self.prepare_keys(table, rows, context)
        self.built.append(tuple(keys["duration"]))
        durations = {1.0: (1.0, 3.0, 4.0), 2.0: (2.0, 3.0, 5.0)}.get(
            float(keys["duration"][0]), tuple(np.unique(keys["duration"]))
        )
        domain = np.array([(value,) for value in durations], dtype=keys.dtype)
        return GuardResult(
            np.ones(len(rows), dtype=bool),
            (GuardPreparation(domain, DomainPayload(durations)),),
        )

    def admit_prepared(self, prepared, table, rows, context):
        self.admitted.append((prepared.durations, tuple(table.duration_ns[rows])))
        if self.decline_first and prepared.durations[0] == 1.0:
            return None
        return GuardResult(np.ones(len(rows), dtype=bool))


@pytest.fixture
def overlapping_guard():
    kernel = OverlappingGuardKernel()
    src = source()
    # Include metadata for the intended resident working set.
    with closing(
        Engine(1, 4096 + 8 * RESIDENT_METADATA_BYTES, kernels=(kernel,))
    ) as engine:
        # Two shared guard builds retain overlapping domains in this order.
        engine.evolve(
            src,
            (replace(batch(4), end_ns=np.array([1.0, 1.0, 2.0, 2.0])),),
            cancel=Event(),
        )
        kernel.admitted.clear()
        kernel.built.clear()
        yield engine, src, kernel


def test_overlapping_cached_domains_prefer_first_payload(overlapping_guard):
    engine, src, kernel = overlapping_guard
    result = engine.evolve(src, (batch(2, duration=3.0),), cancel=Event())
    assert result.paths.segments_by_kernel == {"t4": 2}
    assert kernel.admitted == [((1.0, 3.0, 4.0), (3.0, 3.0))]
    assert kernel.built == []


def multi_interval_batch():
    """One timeline with duration keys 5, 3, 4, 3, 6 in chronological order."""
    return replace(
        batch(1, duration=21.0),
        flux=FluxSchedule(
            np.array([0, 5]),
            np.array([0.0, 5.0, 8.0, 12.0, 15.0]),
            np.arange(5, dtype=np.int64)[:, None],
        ),
    )


def test_cached_admission_preserves_timeline_row_subsets_and_order(overlapping_guard):
    engine, src, kernel = overlapping_guard
    result = engine.evolve(src, (multi_interval_batch(),), cancel=Event())
    assert result.paths.segments_by_kernel == {"t4": 5}
    assert kernel.admitted == [
        ((1.0, 3.0, 4.0), (3.0, 4.0, 3.0)),
        ((2.0, 3.0, 5.0), (5.0,)),
    ]
    assert kernel.built == [(6.0,)]
    np.testing.assert_array_equal(result.densities[0], src.initial_density)


def test_cached_domain_miss_tries_later_overlapping_payload(overlapping_guard):
    engine, src, kernel = overlapping_guard
    kernel.decline_first = True
    result = engine.evolve(src, (multi_interval_batch(),), cancel=Event())
    assert result.paths.segments_by_kernel == {"t4": 5}
    assert kernel.admitted == [
        ((1.0, 3.0, 4.0), (3.0, 4.0, 3.0)),
        ((2.0, 3.0, 5.0), (5.0, 3.0, 3.0)),
    ]
    assert kernel.built == [(4.0, 6.0)]
