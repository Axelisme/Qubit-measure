"""Engine behavior through public batches and injected C5 collaborators."""

import warnings
import weakref
from contextlib import closing
from dataclasses import dataclass, replace
from threading import Event, get_ident

import numpy as np
import pytest
from numpy.typing import NDArray
from zcu_tools.simulate.evolution import (
    Cancelled,
    Engine,
    EnvelopeSpec,
    FluxSchedule,
    ModelSource,
    SecularModel,
)
from zcu_tools.simulate.evolution.cache import RESIDENT_METADATA_BYTES
from zcu_tools.simulate.evolution.kernel import ApplyContext, GuardResult, KernelContext

from ._engine import Payload, RecordingKernel, batch, run, source, track
from .contract.kernel_suite import synthetic_model


def test_geometry_preserves_track_order_mux_plateaus_and_flux_levels():
    from zcu_tools.simulate.evolution.kernels import REGISTRY

    base = synthetic_model()
    model = replace(base, drives=(base.drives[0], replace(base.drives[0], name="mux")))
    src = ModelSource(
        source().initial_density, lambda level: replace(model, level=level)
    )
    env = EnvelopeSpec(
        "sampled", np.array([0, 1j, 1j, 0], complex), np.array([0.0, 2.0, 12.0, 14.0])
    )
    pulse = replace(track(1, env, duration=20), frequency_ghz=np.array([-0.1]))
    mux = replace(
        track(1, EnvelopeSpec("constant"), start=4, duration=12),
        port="mux",
        phase_rad=np.array([0.7]),
    )
    silent = replace(track(1, env, start=1, duration=22), amplitude=np.zeros(1))
    timeline = replace(
        batch(1, duration=24),
        tracks=(pulse, silent, mux),
        flux=FluxSchedule(np.array([0, 2]), np.array([0.0, 8.0]), np.array([[0], [1]])),
    )
    kernels = tuple(RecordingKernel(k.name) for k in REGISTRY)
    result = run(Engine(1, 0, kernels=kernels), src, (timeline,))
    observations = {
        float(table.start_ns[row]): (table, int(row), kernel.name)
        for kernel in kernels
        for table, rows, _ in kernel.calls
        for row in rows
    }
    assert sorted(observations) == [0.0, 2.0, 4.0, 8.0, 12.0, 16.0, 20.0]
    for start, (table, row, family) in observations.items():
        count = (start < 20) + (4 <= start < 16)
        assert table.tones[row] == count
        assert table.level[row].tolist() == [int(start >= 8)]
        assert family == ("flux_dop853" if count else "t4")
        if count:
            assert table.tone_port[row, :count].tolist() == (
                [0, 1] if count == 2 else [0]
            )
            flat = 2 <= start < 12
            assert table.envelopes[table.tone_envelope[row, 0]].kind == (
                "constant" if flat else "sampled"
            )
            assert table.tone_envelope_offset_ns[row, 0] == (0 if flat else start)
            assert table.tone_theta_rad[row, 0] == (
                2 * np.pi * -0.1 * start + (np.pi / 2 if flat else 0)
            )
            if count == 2:
                assert table.tone_envelope_offset_ns[row, 1] == start - 4
                assert table.tone_theta_rad[row, 1] == 2 * np.pi * 0.1 * start + 0.7
    np.testing.assert_array_equal(result.densities[0], src.initial_density)


@pytest.mark.parametrize(
    ("times", "values"),
    [
        ([0.0, 10.0, 30.0, 240.0], [0.0, 1j, 1j, 0.3j]),
        ([0.0, 12.0, 32.0, 240.0], [0.0, 1j, 1j, 0.3j]),
        ([0.0, 10.0, 30.0, 240.0], [0.0, 1j, 1j, 0.4j]),
    ],
)
def test_equal_waveform_copies_keep_physical_rows_across_batches(times, values):
    from zcu_tools.simulate.evolution.kernels import REGISTRY

    base = synthetic_model()
    drives = (base.drives[0], replace(base.drives[0], name="mux"))

    def model(level):
        return replace(
            base, level=level, drives=drives if level == (0,) else drives[::-1]
        )

    src = ModelSource(source().initial_density, model)
    batches = []
    for point, duration in enumerate((254.0, 256.0, 254.0, 256.0)):
        # Equal content is deliberately reconstructed as a distinct record.
        env = EnvelopeSpec("sampled", np.array(values), np.array(times))
        batches.append(
            replace(
                batch(1, duration=duration),
                points=np.array([point]),
                flux=FluxSchedule(
                    np.array([0, 2]),
                    np.array([0.0, 100.0]),
                    np.array([[0], [1]]),
                ),
                tracks=(
                    replace(
                        track(1, env, duration=duration),
                        phase_rad=np.array([point * 0.13]),
                    ),
                    replace(
                        track(1, env, duration=duration),
                        port="absent",
                        active=np.array([False]),
                    ),
                    replace(
                        track(1, EnvelopeSpec("constant"), start=4, duration=238),
                        port="mux",
                    ),
                ),
            )
        )
    kernels = tuple(RecordingKernel(k.name) for k in REGISTRY)
    result = run(Engine(1, 0, kernels=kernels), src, tuple(batches))
    observations = {
        (int(table.timeline[row]), float(table.start_ns[row])): (table, int(row))
        for kernel in kernels
        for table, rows, _ in kernel.calls
        for row in rows
    }
    for point, duration in enumerate((254.0, 256.0, 254.0, 256.0)):
        boundaries = [0.0, 4.0, times[1], times[2], 100.0, 240.0, 242.0, duration]
        assert (
            sorted(start for p, start in observations if p == point) == boundaries[:-1]
        )
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
            table, row = observations[point, start]
            assert table.duration_ns[row] == end - start
            count = 1 + (4 <= start < 242)
            level = int(start >= 100)
            assert table.level[row].tolist() == [level]
            assert table.tones[row] == count
            assert (
                table.tone_port[row, :count].tolist()
                == ([0, 1] if not level else [1, 0])[:count]
            )
            physical = table.envelopes[table.tone_envelope[row, 0]]
            flat = times[1] <= start < times[2] or start >= 240
            assert physical.kind == ("constant" if flat else "sampled")
            if flat:
                value = values[-1] if start >= 240 else 1j
                assert table.tone_amplitude[row, 0] == abs(value)
                assert table.tone_envelope_offset_ns[row, 0] == 0
                assert table.tone_theta_rad[
                    row, 0
                ] == 2 * np.pi * 0.1 * start + point * 0.13 + np.angle(value)
            else:
                np.testing.assert_array_equal(physical.samples, values)
                np.testing.assert_array_equal(physical.sample_ns, times)
                assert table.tone_envelope_offset_ns[row, 0] == start
    np.testing.assert_array_equal(result.points, np.arange(4))
    np.testing.assert_array_equal(
        result.densities, np.repeat(src.initial_density[None], 4, axis=0)
    )


def test_waveform_time_and_complex_samples_remain_distinct_in_one_acquire():
    envelopes = (
        EnvelopeSpec("sampled", np.array([0.2, 1j]), np.array([0.0, 2.0])),
        EnvelopeSpec("sampled", np.array([0.2, 1j]), np.array([0.0, 3.0])),
        EnvelopeSpec("sampled", np.array([0.2, -1j]), np.array([0.0, 2.0])),
    )
    timelines = tuple(
        replace(
            batch(1),
            points=np.array([point]),
            tracks=(track(1, env),),
        )
        for point, env in enumerate(envelopes)
    )
    direct = RecordingKernel("dop853")
    src = source()
    result = run(Engine(1, 0, kernels=(direct,)), src, timelines)
    for table, rows, _ in direct.calls:
        for row in rows:
            expected = envelopes[table.timeline[row]]
            physical = table.envelopes[table.tone_envelope[row, 0]]
            np.testing.assert_array_equal(physical.samples, expected.samples)
            np.testing.assert_array_equal(physical.sample_ns, expected.sample_ns)
    np.testing.assert_array_equal(
        result.densities, np.repeat(src.initial_density[None], 3, axis=0)
    )


def test_active_unknown_port_raises_value_error():
    timeline = replace(
        batch(1),
        tracks=(replace(track(1, EnvelopeSpec("constant")), port="absent"),),
    )
    with (
        closing(Engine(1, 0, kernels=(RecordingKernel("t6"),))) as engine,
        pytest.raises(ValueError, match="absent"),
    ):
        engine.evolve(source(), (timeline,), cancel=Event())


class TimelineKernel(RecordingKernel):
    """Conservative timeline keys, optionally shared by equivalent copies."""

    def __init__(self, name, *, copies=1):
        super().__init__(name)
        self.copies = copies

    def prepare_keys(self, table, rows, context: KernelContext):
        return (table.timeline[rows] // self.copies).copy().view("V8")


def test_idle_reference_and_worker_chunk_batch_independence():
    from zcu_tools.simulate.evolution.kernels.t4 import T4

    from ._tables import idle_table
    from .contract.kernel_suite import KernelCase, reference

    src = source()
    b = batch()
    expected = reference(
        KernelCase(T4(), src, idle_table(np.array([3.0])), src.initial_density[None]),
        0,
        src.initial_density,
    )
    a = run(Engine(1, 100000, chunks_per_worker=1), src, (b,))
    subsets = tuple(replace(batch(4), points=np.arange(i, i + 4)) for i in (0, 4, 8))
    c = run(Engine(2, 0, chunks_per_worker=7), src, subsets)
    np.testing.assert_allclose(
        a.densities, np.repeat(expected[None], 12, axis=0), atol=1e-11
    )
    np.testing.assert_allclose(c.densities, a.densities, atol=1e-13)
    assert a.paths.segments_by_kernel == {"t4": 12}
    assert a.paths.points_by_class == {"t4": 12}


def test_flux_basis_changes_and_readout_truncation():
    from zcu_tools.simulate.evolution.kernels.t4 import T4

    zero = np.zeros((2, 2))
    swap = np.array([[0, 1], [1, 0]], dtype=complex)

    def build(level):
        return SecularModel(
            level,
            str(level),
            np.zeros(2),
            np.eye(2, dtype=complex) if level == (0,) else swap,
            (),
            zero,
            zero,
        )

    src = ModelSource(np.diag([1.0, 0.0]).astype(complex), build)
    b = replace(
        batch(1),
        flux=FluxSchedule(
            np.array([0, 3]), np.array([0.0, 1.0, 100.0]), np.array([[0], [1], [99]])
        ),
        tracks=(replace(track(1, EnvelopeSpec("constant"), start=20), port="absent"),),
    )
    result = run(Engine(1, 1000, kernels=(T4(),)), src, (b,))
    np.testing.assert_allclose(result.densities[0], src.initial_density, atol=1e-14)
    assert result.paths.segments_by_kernel == {"t4": 2}


def test_flat_top_split_phase_offset_and_small_plateau():
    envelope = EnvelopeSpec(
        "sampled", np.array([0, 1j, 1j, 0], complex), np.array([0.0, 2.0, 12.0, 14.0])
    )
    direct, t6 = RecordingKernel("dop853"), RecordingKernel("t6")
    result = run(
        Engine(
            1,
            1000,
            kernels=(
                direct,
                RecordingKernel("dyson", accepts=False),
                RecordingKernel("floquet_period", accepts=False),
                t6,
            ),
        ),
        source(),
        (replace(batch(1, duration=14), tracks=(track(1, envelope, duration=14),)),),
    )
    assert result.paths.segments_by_kernel == {"dop853": 2, "t6": 1}
    table, rows, _ = t6.calls[0]
    assert table.start_ns[rows].tolist() == [2.0]
    assert table.duration_ns[rows].tolist() == [10.0]
    np.testing.assert_allclose(
        table.tone_theta_rad[rows, 0], 2 * np.pi * 0.1 * 2 + np.pi / 2
    )
    assert table.envelopes[table.tone_envelope[rows[0], 0]].kind == "constant"
    assert direct.calls[-1][0].tone_envelope_offset_ns[
        direct.calls[-1][1], 0
    ].tolist() == [12.0]
    short = replace(envelope, sample_ns=np.array([0.0, 2.0, 11.0, 14.0]))
    k = RecordingKernel("dop853")
    result = run(
        Engine(1, 0, kernels=(k,)),
        source(),
        (replace(batch(1, duration=14), tracks=(track(1, short, duration=14),)),),
    )
    assert result.paths.segments_by_kernel == {"dop853": 1}


@pytest.mark.parametrize("kind", ["sampled", "constant"])
def test_pulse_support_ends_at_its_absolute_event(kind):
    envelope = (
        EnvelopeSpec("sampled", np.array([0.2, 1.0]), np.array([0.0, 0.8]))
        if kind == "sampled"
        else EnvelopeSpec("constant")
    )
    idle, direct, constant, flux = (
        RecordingKernel(name) for name in ("t4", "dop853", "t6", "flux_dop853")
    )
    timeline = replace(
        batch(1, duration=4.0),
        # A level switch exactly at the first pulse's end does not overlap it.
        flux=FluxSchedule(
            np.array([0, 3]),
            np.array([0.0, 2.8, 3.2]),
            np.array([[0], [1], [1]]),
        ),
        tracks=(
            track(1, envelope, start=2.0, duration=0.8),
            track(1, envelope, start=3.2, duration=0.8),
        ),
    )
    result = run(
        Engine(
            1,
            0,
            kernels=(
                idle,
                direct,
                constant,
                flux,
                RecordingKernel("dyson", accepts=False),
                RecordingKernel("floquet_period", accepts=False),
            ),
        ),
        source(),
        (timeline,),
    )
    observations = sorted(
        (float(table.start_ns[row]), int(table.tones[row]))
        for kernel in (idle, direct, constant, flux)
        for table, rows, _ in kernel.calls
        for row in rows
    )
    assert observations == [(0.0, 0), (2.0, 1), (2.8, 0), (3.2, 1)]
    driven = direct if kind == "sampled" else constant
    assert result.paths.segments_by_kernel == {"t4": 2, driven.name: 2}
    for table, rows, _ in driven.calls:
        np.testing.assert_array_equal(table.tone_envelope_offset_ns[rows, 0], 0.0)


@pytest.mark.parametrize(
    ("start", "plateau_start", "plateau_end", "duration"),
    [(2.0, 0.8, 10.8, 12.8), (0.1, 0.8, 20.8, 22.8)],
)
def test_plateau_support_uses_absolute_events(
    start, plateau_start, plateau_end, duration
):
    envelope = EnvelopeSpec(
        "sampled",
        np.array([0.0, 1j, 1j, 0.0]),
        np.array([0.0, plateau_start, plateau_end, duration]),
    )
    direct, constant = RecordingKernel("dop853"), RecordingKernel("t6")
    result = run(
        Engine(
            1,
            0,
            kernels=(
                RecordingKernel("t4"),
                direct,
                constant,
                RecordingKernel("dyson", accepts=False),
                RecordingKernel("floquet_period", accepts=False),
            ),
        ),
        source(),
        (
            replace(
                batch(1, duration=start + duration),
                tracks=(track(1, envelope, start=start, duration=duration),),
            ),
        ),
    )
    assert result.paths.segments_by_kernel == {"t4": 1, "dop853": 2, "t6": 1}
    table, rows, _ = constant.calls[0]
    np.testing.assert_allclose(table.start_ns[rows], start + plateau_start)
    np.testing.assert_allclose(table.duration_ns[rows], plateau_end - plateau_start)
    np.testing.assert_array_equal(table.tone_envelope_offset_ns[rows, 0], 0.0)
    offsets = sorted(
        float(table.tone_envelope_offset_ns[row, 0])
        for table, rows, _ in direct.calls
        for row in rows
    )
    np.testing.assert_allclose(offsets, [0.0, plateau_end])


@pytest.mark.parametrize(
    "expected", ["dyson", "single_tone_rwa", "floquet_period", "t6"]
)
def test_constant_decision_order_and_sequence_middle(expected):
    from zcu_tools.simulate.evolution.kernels import REGISTRY

    order = ["dyson", "single_tone_rwa", "floquet_period", "t6"]
    kernels = tuple(
        RecordingKernel(
            kernel.name,
            accepts=kernel.name not in order
            or order.index(kernel.name) >= order.index(expected),
        )
        for kernel in REGISTRY
    )
    b = replace(
        batch(2, duration=6),
        tracks=(
            track(2, EnvelopeSpec("constant")),
            track(2, EnvelopeSpec("constant"), start=3),
        ),
    )
    result = run(Engine(1, 0, kernels=kernels), source(), (b,))
    assert result.paths.segments_by_kernel == {expected: 4}


@pytest.mark.parametrize(
    ("frequency", "expected"),
    [(0.02, "t6"), (0.05, "floquet_period"), (0.6, "floquet_period")],
)
def test_sideband_selective_constant_tone_uses_periodic_propagation(
    frequency, expected
):
    from zcu_tools.simulate.evolution.kernels import REGISTRY

    # Isolate periodic propagation from the constant single-tone RWA route.
    kernels = tuple(k for k in REGISTRY if k.name != "single_tone_rwa")
    upper = np.diag([0.5, 0.35], 1).astype(complex)
    model = synthetic_model(3)
    model = replace(
        model,
        energies_ghz=np.array([0.0, frequency, 2.3 * frequency]),
        drives=(
            replace(
                model.drives[0],
                in_phase_ghz=upper + upper.T,
                quadrature_ghz=0.9999j * (upper - upper.T),
            ),
        ),
        rates_per_ns=model.rates_per_ns * 1e-5,
        dephasing_per_ns=model.dephasing_per_ns * 1e-5,
    )
    src = ModelSource(np.diag([1.0, 0.0, 0.0]).astype(complex), lambda level: model)
    pulse = replace(
        track(1, EnvelopeSpec("constant"), duration=30.0),
        frequency_ghz=np.array([frequency]),
        amplitude=np.array([0.006]),
        phase_rad=np.array([0.37]),
    )
    result = run(
        Engine(1, 100000, kernels=kernels),
        src,
        (replace(batch(1, duration=30.0), tracks=(pulse,)),),
    )
    assert result.paths.segments_by_kernel == {expected: 1}
    assert result.paths.exceeded_by_kernel == {}


@pytest.mark.parametrize(
    ("family", "n"),
    [("t4", 1), ("t6", 1)],
)
def test_incomplete_terminal_family_fails_instead_of_rerouting(family, n):
    kernels = tuple(
        RecordingKernel(name, accepts=name == "dop853")
        for name in (
            "t4",
            "dyson",
            "floquet_period",
            "t6",
            "gate_cache",
            "period_table",
            "dop853",
        )
    )
    b = batch(n)
    if family != "t4":
        envelope = EnvelopeSpec("constant")
        b = replace(b, tracks=(track(n, envelope),))
    with pytest.raises(RuntimeError, match="no valid kernel for segment"):
        run(Engine(1, 0, kernels=kernels), source(), (b,))


def test_missing_kernel_names_segment_and_flux_overlap():
    b = replace(batch(1), tracks=(track(1, EnvelopeSpec("constant")),))
    engine = Engine(1, 0, kernels=())
    try:
        with pytest.raises(RuntimeError, match=r"(?s)(segment.*dyson|dyson.*segment)"):
            engine.evolve(source(), (b,), cancel=Event())
        b = replace(
            b,
            flux=FluxSchedule(
                np.array([0, 2]), np.array([0.0, 1.0]), np.array([[0], [1]])
            ),
        )
        with pytest.raises(
            RuntimeError, match=r"(?s)(segment.*flux_dop853|flux_dop853.*segment)"
        ):
            engine.evolve(source(), (b,), cancel=Event())
    finally:
        engine.close()


def test_shared_cache_namespace_budget_eviction_and_singletons():
    k = RecordingKernel("t4")
    # Include metadata for the intended resident working set.
    engine = Engine(1, 16 + RESIDENT_METADATA_BYTES, kernels=(k,))
    src = source()
    try:
        for duration in [3.0, 3.0, 4.0, 3.0]:
            engine.evolve(src, (batch(2, duration=duration),), cancel=Event())
        assert k.preparations == 3
        engine.evolve(src, (batch(2),), budget=1e-6, cancel=Event())
        assert k.preparations == 4
        other = ModelSource(
            src.initial_density,
            lambda level: replace(src.model(level), identity="different-source"),
        )
        engine.evolve(other, (batch(2),), cancel=Event())
        assert k.preparations == 5
        for _ in range(2):
            engine.evolve(src, (batch(1, duration=8),), cancel=Event())
        assert k.preparations == 6, "the second singleton evolve must reuse its payload"
    finally:
        engine.close()
        engine.close()
    with pytest.raises(RuntimeError, match="closed"):
        engine.evolve(src, (batch(),), cancel=Event())


@pytest.mark.parametrize("workers", [1, 2])
def test_singleton_residents_reuse_payloads_across_evolves(workers):
    kernel = TimelineKernel("t4")
    src = source()
    # Include metadata for the intended resident working set.
    with closing(
        Engine(workers, 128 + 8 * RESIDENT_METADATA_BYTES, kernels=(kernel,))
    ) as engine:
        first = engine.evolve(src, (batch(8),), cancel=Event())
        assert kernel.preparations == 8
        second = engine.evolve(src, (batch(8),), cancel=Event())
        assert kernel.preparations == 8, "resident singletons must not rebuild"
        np.testing.assert_array_equal(second.densities, first.densities)
        assert second.paths == first.paths


def test_oversize_local_payload_does_not_evict_fitting_singleton():
    class SizedKernel(RecordingKernel):
        def prepare(self, keys, table, rows, context: KernelContext):
            super().prepare(keys, table, rows, context)
            return Payload(2 if table.duration_ns[rows[0]] == 4 else 1)

    kernel = SizedKernel("t4")
    src = source()
    # Include metadata for the intended resident working set.
    with closing(Engine(1, 16 + RESIDENT_METADATA_BYTES, kernels=(kernel,))) as engine:
        engine.evolve(src, (batch(1),), cancel=Event())
        for point in range(2):
            # Distinct points avoid result memo after including retention metadata.
            timeline = replace(batch(1, duration=4), points=np.array([point]))
            engine.evolve(src, (timeline,), cancel=Event())
        engine.evolve(src, (batch(1),), cancel=Event())
        assert kernel.preparations == 3, (
            "oversize products must stream without eviction"
        )


def test_singleton_stream_preserves_current_shared_hit_and_fills_only_spare_bytes():
    kernel = RecordingKernel("t4")
    src = source()
    shared = batch(2)
    stream = replace(batch(3), points=np.arange(2, 5), end_ns=np.array([4.0, 5.0, 6.0]))
    # Include metadata for the intended resident working set.
    with closing(
        Engine(
            1, 32 + 2 * RESIDENT_METADATA_BYTES, kernels=(kernel,), chunks_per_worker=1
        )
    ) as engine:
        engine.evolve(src, (shared,), cancel=Event())
        engine.evolve(src, (shared, stream), cancel=Event())
        assert kernel.preparations == 4
        engine.evolve(src, (shared,), cancel=Event())
        assert kernel.preparations == 4, "streaming misses must not evict shared hits"
        engine.evolve(src, (batch(1, duration=4),), cancel=Event())
        assert kernel.preparations == 4, "the fitting local product must persist"


def test_new_source_singleton_replaces_previous_source_resident():
    kernel = RecordingKernel("t4")
    src = source()
    other = ModelSource(
        src.initial_density,
        lambda level: replace(src.model(level), identity="next-flux"),
    )
    # Include metadata for the intended resident working set.
    with closing(Engine(1, 16 + RESIDENT_METADATA_BYTES, kernels=(kernel,))) as engine:
        for current in (src, src, other, other, src):
            engine.evolve(current, (batch(1),), cancel=Event())
        assert kernel.preparations == 3, "old flux residents must yield to new ones"


@pytest.mark.parametrize(
    ("repeated", "name"),
    [(False, "t4"), (False, "dop853"), (True, "t4"), (True, "flux_dop853")],
)
def test_timeline_local_payloads_are_released_before_the_next_timeline(repeated, name):
    from zcu_tools.simulate.evolution.kernel import GuardPreparation

    class LifetimeKernel(TimelineKernel):
        guard_builds = True

        def __init__(self):
            super().__init__(name)
            self.live = 0
            self.peak_live = 0

        def release(self, count):
            self.live -= count

        def prepare(self, keys, table, rows, context: KernelContext):
            payload = super().prepare(keys, table, rows, context)
            self.live += len(keys)
            self.peak_live = max(self.peak_live, self.live)
            weakref.finalize(payload, self.release, len(keys))
            return payload

        def guard_prepared(self, table, rows, context: KernelContext):
            keys, first = np.unique(
                self.prepare_keys(table, rows, context), return_index=True
            )
            payload = self.prepare(keys, table, rows[first], context)
            return GuardResult(
                self.guard(table, rows, context),
                (GuardPreparation(keys, payload),),
            )

    n = 8
    b = batch(n)
    if repeated:
        b = replace(
            b,
            flux=FluxSchedule(
                2 * np.arange(n + 1),
                np.tile([0.0, 1.0], n),
                np.tile([[0], [1]], (n, 1)),
            ),
        )
    if name != "t4":
        b = replace(
            b, tracks=tuple(track(n, EnvelopeSpec("constant")) for _ in range(3))
        )
    kernel = LifetimeKernel()
    src = source()
    result = run(Engine(1, 0, kernels=(kernel,), chunks_per_worker=1), src, (b,))
    np.testing.assert_array_equal(
        result.densities, np.repeat(src.initial_density[None], n, axis=0)
    )
    assert kernel.preparations == n, "same-timeline repeats must prepare only once"
    assert kernel.peak_live == 1, (
        "local payloads must not accumulate in A-stage or chunks"
    )
    assert kernel.live == 0, "the evolve call must release all local payloads"


def test_guard_pool_preserves_dispatch_and_density_and_skips_small_key_sets():
    class ObservingDyson(TimelineKernel):
        guard_builds = True

        def __init__(self):
            super().__init__("dyson", copies=2)
            self.guard_threads: set[int] = set()

        def guard_prepared(self, table, rows, context: KernelContext):
            from zcu_tools.simulate.evolution.kernel import GuardPreparation

            keys, first = np.unique(
                self.prepare_keys(table, rows, context), return_index=True
            )
            payload = self.prepare(keys, table, rows[first], context)
            return GuardResult(
                self.guard(table, rows, context),
                (GuardPreparation(keys, payload),),
            )

        def guard(self, table, rows, context: KernelContext):
            self.guard_threads.add(get_ident())
            return table.timeline[rows] % 3 != 0

    caller = get_ident()
    for count in (7, 16):
        outputs = []
        for workers in (1, 2):
            kernel = ObservingDyson()
            b = replace(
                batch(2 * count), tracks=(track(2 * count, EnvelopeSpec("constant")),)
            )
            result = run(
                Engine(workers, 0, kernels=(kernel, RecordingKernel("floquet_period"))),
                source(),
                (b,),
            )
            outputs.append(result)
            if workers == 1 or count < 8:
                assert kernel.guard_threads == {caller}, "small guards must stay serial"
            else:
                assert caller not in kernel.guard_threads, (
                    "guards must reuse Engine's pool"
                )
        assert (
            outputs[0].paths.segments_by_kernel == outputs[1].paths.segments_by_kernel
        )
        np.testing.assert_array_equal(outputs[0].densities, outputs[1].densities)


@pytest.mark.parametrize(
    ("duration", "fallback"), [(3, "dop853"), (4000, "period_table")]
)
def test_local_guard_tables_build_once_and_match_shared_guard_first_fallback(
    duration, fallback
):
    from zcu_tools.simulate.evolution.kernel import GuardPreparation

    count = 8

    class TableKernel(TimelineKernel):
        guard_builds = True

        def __init__(self, copies):
            super().__init__("shaped_sambe", copies=copies)
            self.builds = np.zeros(count, dtype=np.int64)
            self.tokens = {}
            self.used = []

        def prepare(self, keys, table, rows, context: KernelContext):
            points = keys.view(np.int64)
            self.builds[points] += 1
            payload = Payload(len(keys))
            self.tokens[id(payload)] = self.builds[points].copy()
            return payload

        def guard_prepared(self, table, rows, context: KernelContext):
            keys, first = np.unique(
                self.prepare_keys(table, rows, context), return_index=True
            )
            payload = self.prepare(keys, table, rows[first], context)
            return GuardResult(
                (table.timeline[rows] // self.copies) % 2 == 0,
                (GuardPreparation(keys, payload),),
            )

        def guard(self, table, rows, context: KernelContext):
            return self.guard_prepared(table, rows, context).accepted

        def apply(
            self, prepared, key_index, table, rows, states, context: ApplyContext
        ):
            self.used.extend(
                (
                    int(table.timeline[row] // self.copies),
                    int(self.tokens[id(prepared)][index]),
                )
                for row, index in zip(rows, key_index, strict=True)
            )
            return super().apply(prepared, key_index, table, rows, states, context)

    outputs = []
    for copies, workers, chunks in ((2, 1, 1), (1, 1, 1), (1, 2, 1), (1, 2, 4)):
        kernel = TableKernel(copies)
        total = copies * count
        b = replace(
            batch(total),
            end_ns=np.full(total, duration),
            tracks=(
                track(
                    total,
                    EnvelopeSpec(
                        "sampled",
                        np.array([0, 1, 0], dtype=complex),
                        np.array([0, duration / 2, duration]),
                    ),
                    duration=duration,
                ),
            ),
        )
        result = run(
            Engine(
                workers,
                0,
                kernels=(
                    kernel,
                    TimelineKernel(
                        "period_table", copies=2 if duration == 4000 else copies
                    ),
                    RecordingKernel("dop853"),
                ),
                chunks_per_worker=chunks,
            ),
            source(),
            (b,),
        )
        assert result.paths.segments_by_kernel == {
            "shaped_sambe": total // 2,
            fallback: total // 2,
        }
        np.testing.assert_array_equal(kernel.builds, np.ones(count, dtype=np.int64))
        assert sorted(kernel.used) == sorted(
            (point, 1) for point in range(0, count, 2) for _ in range(copies)
        )
        outputs.append(result.densities[::copies])
    for densities in outputs[1:]:
        np.testing.assert_array_equal(densities, outputs[0])


def test_local_singletons_keep_shared_apply_waves_and_release_each_payload():
    from threading import get_ident

    caller = get_ident()
    for workers in (1, 2):

        class Idle(TimelineKernel):
            def __init__(self):
                super().__init__("t4")
                self.live = 0
                self.peak = 0
                self.guarded = set()

            def guard(self, table, rows, context: KernelContext):
                self.guarded.add(get_ident())
                return super().guard(table, rows, context)

            def release(self):
                self.live -= 1

            def prepare(self, keys, table, rows, context: KernelContext):
                payload = super().prepare(keys, table, rows, context)
                self.live += 1
                self.peak = max(self.peak, self.live)
                weakref.finalize(payload, self.release)
                return payload

        idle, driven = Idle(), RecordingKernel("dop853")
        b = replace(
            batch(8),
            tracks=tuple(
                track(8, EnvelopeSpec("constant"), duration=1) for _ in range(3)
            ),
        )
        result = run(
            Engine(workers, 0, kernels=(idle, driven), chunks_per_worker=1),
            source(),
            (b,),
        )
        assert result.paths.segments_by_kernel == {"t4": 8, "dop853": 8}
        assert idle.guarded == {caller}
        assert any(len(rows) > 1 for _, rows, _ in driven.calls)
        assert idle.preparations == 8
        assert idle.peak <= workers
        assert idle.live == 0


def test_deferred_guards_do_not_retry_caller_rejections():
    from collections import Counter

    from zcu_tools.simulate.evolution.kernel import GuardPreparation

    for workers in (1, 2):

        class Rejecting(RecordingKernel):
            def __init__(self):
                super().__init__("dyson")
                self.guarded = []

            def guard(self, table, rows, context: KernelContext):
                self.guarded.extend(table.timeline[rows])
                return np.zeros(len(rows), dtype=bool)

        class BuildingPeriod(TimelineKernel):
            guard_builds = True

            def guard_prepared(self, table, rows, context: KernelContext):
                keys, first = np.unique(
                    self.prepare_keys(table, rows, context), return_index=True
                )
                payload = self.prepare(keys, table, rows[first], context)
                return GuardResult(
                    np.ones(len(rows), dtype=bool), (GuardPreparation(keys, payload),)
                )

        period = BuildingPeriod("floquet_period")
        rejecting = Rejecting()
        b = replace(batch(8), tracks=(track(8, EnvelopeSpec("constant")),))
        result = run(Engine(workers, 0, kernels=(rejecting, period)), source(), (b,))
        assert result.paths.segments_by_kernel == {"floquet_period": 8}
        assert Counter(rejecting.guarded) == Counter(range(8))
        assert period.preparations == 8


def test_cancel_between_chunks_and_nonfinite_failure():
    cancel = Event()
    k = RecordingKernel("t4", cancel=cancel)
    engine = Engine(1, 1000, kernels=(k,), chunks_per_worker=4)
    try:
        with pytest.raises(Cancelled):
            engine.evolve(source(), (batch(),), cancel=cancel)
        assert len(k.calls) == 1
    finally:
        engine.close()

    class Broken(RecordingKernel):
        def apply(
            self, prepared, key_index, table, rows, states, context: ApplyContext
        ):
            states[:] = np.nan
            return np.zeros(len(rows), dtype=np.bool_)

    engine = Engine(1, 0, kernels=(Broken("t4"),))
    try:
        with pytest.raises(RuntimeError, match=r"(?s)(point.*segment|segment.*point)"):
            engine.evolve(source(), (batch(),), cancel=Event())
    finally:
        engine.close()


def test_dop853_adjacent_rows_are_one_chain():
    k = RecordingKernel("dop853")
    # Three overlapping tones, with one replaced at t=1.
    tracks = tuple(track(1, EnvelopeSpec("constant"), duration=3) for _ in range(2)) + (
        track(1, EnvelopeSpec("constant"), duration=1),
        track(1, EnvelopeSpec("constant"), start=1, duration=2),
    )
    run(Engine(1, 0, kernels=(k,)), source(), (replace(batch(1), tracks=tracks),))
    assert len(k.calls) == 1
    table, rows, _ = k.calls[0]
    assert table.start_ns[rows].tolist() == [0.0, 1.0]
    assert table.timeline[rows].tolist() == [0, 0]


def test_gibbs_constant_pulse_at_zero_uses_dyson_for_every_driven_row():
    model = synthetic_model(3)
    model = replace(
        model,
        rates_per_ns=np.zeros((3, 3)),
        dephasing_per_ns=np.zeros((3, 3)),
    )
    weights = np.exp(-model.energies_ghz / 0.2)
    weights /= weights.sum()
    src = ModelSource(
        np.diag(weights).astype(complex), lambda level: replace(model, level=level)
    )
    n = 3
    pulse = replace(
        track(n, EnvelopeSpec("constant"), duration=8),
        frequency_ghz=np.linspace(0.36, 0.44, n),
        amplitude=np.full(n, 0.002),
    )
    b = replace(batch(n, duration=9), tracks=(pulse,))
    result = run(Engine(1, 100000), src, (b,))
    assert result.paths.segments_by_kernel == {"dyson": n, "t4": n}
    assert result.paths.exceeded_by_kernel == {}


def test_preceding_idle_flux_uses_arbitrary_input_dyson_bound():
    model = synthetic_model(3)
    x = np.zeros((3, 3), dtype=complex)
    x[1, 2] = x[2, 1] = 1
    model = replace(
        model,
        energies_ghz=np.zeros(3),
        rates_per_ns=np.zeros((3, 3)),
        dephasing_per_ns=np.zeros((3, 3)),
        drives=(replace(model.drives[0], in_phase_ghz=x, quadrature_ghz=x * 0),),
    )
    basis = np.eye(3, dtype=complex)[[1, 0, 2]]

    def build(level):
        return replace(
            model,
            level=level,
            identity=str(level),
            eigenvectors=np.eye(3, dtype=complex) if level == (0,) else basis,
        )

    src = ModelSource(np.diag([0.999, 0.001, 0.0]).astype(complex), build)
    pulse = replace(
        track(1, EnvelopeSpec("constant"), start=1, duration=1),
        frequency_ghz=np.zeros(1),
        amplitude=np.full(1, 0.003),
    )
    b = replace(
        batch(1, duration=2),
        tracks=(pulse,),
        flux=FluxSchedule(np.array([0, 2]), np.array([0.0, 1.0]), np.array([[1], [0]])),
    )
    result = run(Engine(1, 100000), src, (b,))
    assert result.paths.segments_by_kernel.get("dyson", 0) == 0
    assert result.paths.segments_by_kernel["t4"] == 1
    assert sum(result.paths.segments_by_kernel.values()) == 2
    # The identical drive without a preceding segment can use DC ground input.
    direct = replace(
        batch(1, duration=1), tracks=(replace(pulse, start_ns=np.zeros(1)),)
    )
    direct_result = run(Engine(1, 100000), src, (direct,))
    assert direct_result.paths.segments_by_kernel == {"dyson": 1}


@pytest.mark.parametrize("chunks", [1, 4])
def test_exceeded_counts_warn_once_per_evolve_and_keep_results(chunks: int) -> None:
    class ReportingKernel(RecordingKernel):
        report = True

        def apply(
            self, prepared, key_index, table, rows, states, context: ApplyContext
        ):
            super().apply(prepared, key_index, table, rows, states, context)
            return (table.timeline[rows] % 2 == 0) & self.report

    kernels = (ReportingKernel("dyson"), ReportingKernel("t4"))
    engine = Engine(1, 1000, kernels=kernels, chunks_per_worker=chunks)
    src = source()
    b = replace(batch(6, duration=6), tracks=(track(6, EnvelopeSpec("constant")),))
    try:
        for _ in range(2):
            with pytest.warns(RuntimeWarning) as caught:
                result = engine.evolve(src, (b,), cancel=Event())
            assert len(caught) == 1
            message = str(caught[0].message)
            assert "6 segments" in message
            assert "dyson: 3" in message
            assert "t4: 3" in message
            assert "results use the uncorrected approximation" in message
            assert result.paths.exceeded_by_kernel == {"dyson": 3, "t4": 3}
            np.testing.assert_array_equal(
                result.densities, np.repeat(src.initial_density[None], 6, axis=0)
            )
        for kernel in kernels:
            kernel.report = False
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = engine.evolve(src, (b,), cancel=Event())
        assert not caught
        assert result.paths.exceeded_by_kernel == {}
    finally:
        engine.close()


@dataclass(frozen=True)
class BoundPayload:
    initial: NDArray[np.complex128]

    @property
    def nbytes(self) -> int:
        return self.initial.nbytes


class InputBindingKernel(RecordingKernel):
    """C5 collaborator observing lifetime and grouping without engine internals."""

    def __init__(self, *, decline: bool = False):
        super().__init__("dyson")
        self.decline = decline
        self.groups: list[tuple[NDArray[np.float64], NDArray[np.float64]]] = []
        self.bound_refs: list[weakref.ReferenceType[BoundPayload]] = []
        self.bound_rows: list[int] = []
        self.unbound_rows: list[int] = []

    def prepare_keys(self, table, rows, context: KernelContext):
        return np.zeros(len(rows), dtype="V8")

    def bind_input(
        self, prepared, key_index, table, rows, source_initial, context: KernelContext
    ):
        assert np.all(table.start_ns[rows] == 0)
        assert len(np.unique(table.tone_port[rows, 0])) == 1
        assert not source_initial.flags.writeable
        self.groups.append((table.tone_amplitude[rows, 0], table.duration_ns[rows]))
        if self.decline:
            return None
        bound = BoundPayload(source_initial)
        self.bound_refs.append(weakref.ref(bound))
        return bound

    def apply(self, prepared, key_index, table, rows, states, context: ApplyContext):
        if isinstance(prepared, BoundPayload):
            assert np.all(key_index == 0)
            self.bound_rows.extend(table.timeline[rows].tolist())
            states[:] = prepared.initial
        else:
            self.unbound_rows.extend(table.timeline[rows].tolist())
        return np.zeros(len(rows), dtype=np.bool_)


@pytest.mark.parametrize("decline", [False, True])
def test_input_bound_payload_grouping_fallback_and_evolve_lifetime(decline):
    n = 8
    k = InputBindingKernel(decline=decline)
    drive = replace(
        track(n, EnvelopeSpec("constant")),
        start_ns=np.array([0.0] * 6 + [1.0, 2.0]),
        duration_ns=(np.arange(n) + 1) / 8,
        amplitude=np.linspace(-0.001, 0.002, n),
    )
    b = replace(batch(n), tracks=(drive,))
    src = source()
    model = src.model((0,))
    other = ModelSource(
        np.diag([0.6, 0.1, 0.3]).astype(complex),
        lambda level: replace(model, level=level),
    )
    from zcu_tools.simulate.evolution.kernels.t4 import T4

    with closing(Engine(2, 100000, kernels=(k, T4()), chunks_per_worker=3)) as engine:
        first = engine.evolve(src, (b,), cancel=Event())
        assert len(k.groups) == 1, (
            "A-stage must bind all same-key source-input rows once"
        )
        np.testing.assert_array_equal(
            np.sort(k.groups[0][0]), np.sort(drive.amplitude[:6])
        )
        np.testing.assert_array_equal(
            np.sort(k.groups[0][1]), np.sort(drive.duration_ns[:6])
        )
        assert all(ref() is None for ref in k.bound_refs), (
            "bound payload must not enter PrepareCache"
        )
        second = engine.evolve(other, (b,), cancel=Event())
        assert len(k.groups) == 2
        assert all(ref() is None for ref in k.bound_refs)
        assert set(k.bound_rows) == (set() if decline else set(range(6)))
        assert set(k.unbound_rows) == (set(range(n)) if decline else {6, 7})
        assert not np.array_equal(first.densities, second.densities)


def test_guard_handoff_only_binds_repeated_keys():
    from zcu_tools.simulate.evolution.kernel import GuardPreparation

    class GuardBindingKernel(InputBindingKernel):
        guard_builds = True

        def prepare_keys(self, table, rows, context: KernelContext):
            return table.duration_ns[rows].copy().view("V8")

        def guard_prepared(self, table, rows, context: KernelContext):
            keys, first = np.unique(
                self.prepare_keys(table, rows, context), return_index=True
            )
            prepared = self.prepare(keys, table, rows[first], context)
            return GuardResult(
                self.guard(table, rows, context),
                (GuardPreparation(keys, prepared),),
            )

    k = GuardBindingKernel()
    durations = np.array([1.0, 1.0, 2.0])
    drive = replace(track(3, EnvelopeSpec("constant")), duration_ns=durations)
    b = replace(batch(3), tracks=(drive,), end_ns=durations)
    src = source()
    with closing(Engine(1, 0, kernels=(k,))) as engine:
        result = engine.evolve(src, (b,), cancel=Event())
    assert len(k.groups) == 1, "guard-built singletons must bypass bind_input"
    np.testing.assert_array_equal(k.groups[0][1], [1.0, 1.0])
    assert k.bound_rows == [0, 1]
    assert k.unbound_rows == [2]
    assert k.preparations == 2
    np.testing.assert_array_equal(
        result.densities, np.repeat(src.initial_density[None], 3, axis=0)
    )


def test_shared_dyson_is_bitwise_chunk_independent_and_source_local():
    from zcu_tools.simulate.evolution.kernels.dyson import DysonKernel
    from zcu_tools.simulate.evolution.kernels.t4 import T4

    n = 10
    drive = replace(
        track(n, EnvelopeSpec("constant")),
        duration_ns=(np.arange(n) + 1) / 16,
        frequency_ghz=np.linspace(0.11, 0.2, n),
        amplitude=np.linspace(-0.0002, 0.0002, n),
    )
    b = replace(batch(n), tracks=(drive,))
    src = source()
    model = src.model((0,))
    vector = np.exp(1j * np.arange(3)) / np.sqrt(3)
    other = ModelSource(
        np.outer(vector, vector.conj()), lambda level: replace(model, level=level)
    )
    assert callable(getattr(DysonKernel(), "bind_input", None)), (
        "Dyson must share source-input paths"
    )
    outputs = []
    for workers, chunks in ((1, 1), (1, 3), (2, 1), (2, 4)):
        with closing(
            Engine(
                workers, 100000, kernels=(DysonKernel(), T4()), chunks_per_worker=chunks
            )
        ) as engine:
            result = engine.evolve(src, (b,), cancel=Event())
            changed = engine.evolve(other, (b,), cancel=Event())
        assert result.paths.segments_by_kernel["dyson"] == n
        outputs.append((result.densities, changed.densities))
    for original, changed in outputs[1:]:
        np.testing.assert_array_equal(original, outputs[0][0])
        np.testing.assert_array_equal(changed, outputs[0][1])
    assert np.max(abs(outputs[0][0] - outputs[0][1])) > 0.01


@pytest.mark.parametrize("residual", [2.62e-8, 2e-3])
def test_trace_diagnostics_report_maximum_without_correcting_kernel_output(residual):
    class TraceResidualKernel(RecordingKernel):
        def apply(
            self, prepared, key_index, table, rows, states, context: ApplyContext
        ):
            states[:, 0, 0] += residual * (table.timeline[rows] + 1)
            return np.zeros(len(rows), dtype=np.bool_)

    engine = Engine(1, 0, kernels=(TraceResidualKernel("t4"),))
    result = run(engine, source(), (batch(3),))
    expected = np.repeat(source().initial_density[None], 3, axis=0)
    expected[:, 0, 0] += residual * np.arange(1, 4)
    np.testing.assert_allclose(result.densities, expected, rtol=0, atol=1e-15)
    assert getattr(result.paths, "max_trace_error", None) == pytest.approx(3 * residual)


def test_empty_evolve_has_zero_trace_diagnostic():
    result = run(Engine(1, 0), source(), ())
    assert getattr(result.paths, "max_trace_error", None) == 0.0


def test_unknown_input_probe_uses_dyson_without_binding_and_matches_dop853():
    from zcu_tools.simulate.evolution.kernels.dop853 import DOP853Kernel
    from zcu_tools.simulate.evolution.kernels.dyson import DysonKernel
    from zcu_tools.simulate.evolution.kernels.floquet_period import FloquetPeriodKernel
    from zcu_tools.simulate.evolution.kernels.t4 import T4
    from zcu_tools.simulate.evolution.kernels.t6 import T6Kernel

    class ObservedDyson(DysonKernel):
        """Delegate C5 input binding, recording only its public row inputs."""

        def __init__(self):
            self.bound_starts = []

        def bind_input(
            self,
            prepared,
            key_index,
            table,
            rows,
            source_initial,
            context: KernelContext,
        ):
            self.bound_starts.extend(table.start_ns[rows].tolist())
            return super().bind_input(
                prepared, key_index, table, rows, source_initial, context
            )

    class ReferenceDirect(DOP853Kernel):
        def apply(
            self, prepared, key_index, table, rows, states, context: ApplyContext
        ):
            from .contract.kernel_suite import kernel_context

            return super().apply(
                prepared,
                key_index,
                table,
                rows,
                states,
                kernel_context("dop853", table, context.budget),
            )

    class ReferenceDrive(ReferenceDirect):
        """Use full-carrier integration at the first constant-tone candidate."""

        name = "dyson"

    class ReferenceIdle(ReferenceDirect):
        """Use full Lindblad integration for idle rows too."""

        name = "t4"

    src = source()
    probe = replace(
        track(3, EnvelopeSpec("constant"), duration=0.6),
        # Below the period carrier floor, distinct inputs still reach Dyson.
        start_ns=np.array([1.2, 1.3, 1.4]),
        frequency_ghz=np.full(3, 0.025),
        amplitude=np.array([1e-5, 2e-5, 0.3]),
        phase_rad=np.array([0.2, -0.5, 1.1]),
    )
    prefix = replace(
        track(3, EnvelopeSpec("constant"), duration=0.4),
        frequency_ghz=np.full(3, 1.2),
        amplitude=np.full(3, 0.003),
    )
    b = replace(
        batch(3),
        tracks=(prefix, probe),
        end_ns=probe.start_ns + probe.duration_ns,
    )
    dyson = ObservedDyson()
    actual = run(
        Engine(
            1,
            100000,
            kernels=(
                T4(),
                dyson,
                FloquetPeriodKernel(),
                T6Kernel(),
                DOP853Kernel(),
            ),
        ),
        src,
        (b,),
    )
    # The two weak later probes use arbitrary-input B2. The strong probe
    # rejects Dyson and reaches a real fallback, rather than retrying at apply.
    assert actual.paths.segments_by_kernel.get("dyson", 0) == 5
    assert (
        sum(
            count
            for name, count in actual.paths.segments_by_kernel.items()
            if name not in ("dyson", "t4")
        )
        == 1
    )
    assert all(start == 0 for start in dyson.bound_starts)
    assert dyson.bound_starts
    expected = run(
        Engine(
            1, 0, kernels=(ReferenceDrive(), ReferenceIdle(), FloquetPeriodKernel())
        ),
        src,
        (b,),
    )
    np.testing.assert_allclose(actual.densities, expected.densities, rtol=0, atol=1e-3)
    np.testing.assert_allclose(
        np.diagonal(actual.densities, axis1=1, axis2=2),
        np.diagonal(expected.densities, axis1=1, axis2=2),
        rtol=0,
        atol=1e-3,
    )


@pytest.mark.parametrize("bind", [False, True])
def test_shared_shaped_prefix_supplies_one_input_to_probe_and_later_segments(bind):
    class PrefixKernel(RecordingKernel):
        def __init__(self, name, delta):
            super().__init__(name)
            self.delta = delta
            self.applied_rows = 0

        def apply(
            self, prepared, key_index, table, rows, states, context: ApplyContext
        ):
            self.applied_rows += len(rows)
            states[:, 0, 0] += self.delta
            states[:, 1, 1] -= self.delta
            return super().apply(prepared, key_index, table, rows, states, context)

    class ProbeKernel(RecordingKernel):
        def __init__(self):
            super().__init__("dyson")
            self.known_inputs = []

        def bind_input(
            self,
            prepared,
            key_index,
            table,
            rows,
            source_initial,
            context: KernelContext,
        ):
            self.known_inputs.append(source_initial.copy())
            return BoundPayload(source_initial) if bind else None

    # A representative has low reuse and takes Sambe; the full batch's three
    # shaped rows take the ordinary cost order. Both are valid approximations.
    spectral = PrefixKernel("shaped_sambe", 0.05)
    direct = PrefixKernel("dop853", 0.1)
    probe = ProbeKernel()
    idle = RecordingKernel("t4")
    shaped = EnvelopeSpec(
        "sampled", sample_ns=np.array([0.0, 0.5, 1.0]), samples=np.array([0, 1, 0])
    )
    b = replace(
        batch(3),
        tracks=(
            track(3, shaped, duration=1),
            track(3, EnvelopeSpec("constant"), start=1, duration=1),
        ),
    )
    src = source()
    result = run(Engine(1, 0, kernels=(spectral, direct, probe, idle)), src, (b,))
    assert len(probe.known_inputs) == 1
    saved = probe.known_inputs[0]
    assert np.max(abs(saved - src.initial_density)) > 0.01
    for _, _, received in (*probe.calls, *idle.calls):
        np.testing.assert_array_equal(
            received, np.repeat(saved[None], len(received), axis=0)
        )
    np.testing.assert_array_equal(result.densities, np.repeat(saved[None], 3, axis=0))
    assert spectral.applied_rows + direct.applied_rows == 1, (
        "identical prefixes must be evolved once, not again in workers"
    )


def test_shared_prefix_paths_and_exceedances_count_each_adopting_timeline():
    class ExceededPrefix(RecordingKernel):
        def apply(
            self, prepared, key_index, table, rows, states, context: ApplyContext
        ):
            super().apply(prepared, key_index, table, rows, states, context)
            return np.ones(len(rows), dtype=bool)

    class RejectedDirect(RecordingKernel):
        def __init__(self):
            super().__init__("dop853", accepts=False)
            self.guarded_rows = 0

        def guard(self, table, rows, context: KernelContext):
            self.guarded_rows += len(rows)
            return super().guard(table, rows, context)

    spectral = ExceededPrefix("shaped_sambe")
    direct = RejectedDirect()
    gate = RecordingKernel("gate_cache")
    shaped = EnvelopeSpec(
        "sampled", sample_ns=np.array([0.0, 0.5, 1.0]), samples=np.array([0, 1, 0])
    )
    b = replace(
        batch(3),
        tracks=(
            track(3, shaped, duration=1),
            track(3, EnvelopeSpec("constant"), start=1, duration=1),
        ),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = run(
            Engine(
                1,
                0,
                kernels=(
                    spectral,
                    direct,
                    gate,
                    RecordingKernel("dyson"),
                    RecordingKernel("t4"),
                ),
            ),
            source(),
            (b,),
        )
    assert result.paths.segments_by_kernel == {"shaped_sambe": 3, "dyson": 3, "t4": 3}
    assert result.paths.points_by_class == {"shaped_sambe": 3, "dyson": 3, "t4": 3}
    assert result.paths.exceeded_by_kernel == {"shaped_sambe": 3}
    assert len(caught) == 1
    assert "3 segments (shaped_sambe: 3)" in str(caught[0].message)
    assert direct.guarded_rows == 0, "adopted prefix rows must not be reclassified"
    assert not gate.preparations, "adopted prefix rows must not be prepared again"


class InventoryKernel(RecordingKernel):
    """Observe whole-allocation ownership without retaining numerical payloads."""

    def __init__(self, name="t4"):
        super().__init__(name)
        self.payloads: list[weakref.ReferenceType[Payload]] = []
        self.live_before_build: list[int] = []

    @property
    def live_bytes(self):
        return sum(payload.nbytes for ref in self.payloads if (payload := ref()))

    def prepare(self, keys, table, rows, context: KernelContext):
        self.live_before_build.append(self.live_bytes)
        payload = super().prepare(keys, table, rows, context)
        self.payloads.append(weakref.ref(payload))
        return payload


@pytest.mark.parametrize("changed_model", [False, True])
def test_complete_inventory_releases_stale_on_offer_before_next_build(changed_model):
    kernel = InventoryKernel()
    src = source()
    other = (
        ModelSource(
            src.initial_density,
            lambda level: replace(src.model(level), identity="adjacent-flux"),
        )
        if changed_model
        else src
    )
    # Include metadata for the intended resident working set.
    with closing(
        Engine(
            1, 32 + 2 * RESIDENT_METADATA_BYTES, kernels=(kernel,), chunks_per_worker=2
        )
    ) as engine:
        seed = replace(batch(4), end_ns=np.array([3.0, 3.0, 7.0, 7.0]))
        engine.evolve(src, (seed,), cancel=Event())
        assert kernel.live_bytes == 32
        misses = replace(batch(3), end_ns=np.array([4.0, 5.0, 6.0]))
        engine.evolve(other, (misses,), cancel=Event())
        # One inline worker may own a 16-byte active build above the account.
        # Once offered, it must release stale before the next build starts.
        assert kernel.live_before_build[2:] == [32, 32, 32], (
            "resident plus pending must fit after every offer"
        )
        assert kernel.live_bytes <= 32


def test_inventory_keeps_unused_residents_when_new_products_fit():
    kernel = InventoryKernel()
    src = source()
    # Include metadata for the intended resident working set.
    with closing(
        Engine(1, 64 + 4 * RESIDENT_METADATA_BYTES, kernels=(kernel,))
    ) as engine:
        engine.evolve(src, (batch(2),), cancel=Event())
        unused = kernel.payloads[0]
        engine.evolve(src, (batch(2, duration=4),), cancel=Event())
        assert unused() is not None, "spare capacity must preserve stale reuse"
        again = replace(batch(2), points=np.arange(2) + 10)
        engine.evolve(src, (again,), cancel=Event())
        assert kernel.preparations == 2


def test_inventory_protects_whole_borrowed_batches_and_bounds_pending():
    kernel = InventoryKernel()
    src = source()
    # Keep one whole three-key batch, including its metadata, protected.
    with closing(
        Engine(1, 48 + RESIDENT_METADATA_BYTES, kernels=(kernel,), chunks_per_worker=2)
    ) as engine:
        # One indivisible allocation owns both keys, even when only 3 ns is used.
        seed = replace(batch(6), end_ns=np.array([3.0, 3.0, 7.0, 7.0, 8.0, 8.0]))
        engine.evolve(src, (seed,), cancel=Event())
        borrowed = kernel.payloads[0]
        assert borrowed() is not None
        mixed = replace(batch(4), end_ns=np.array([3.0, 4.0, 5.0, 6.0]))
        engine.evolve(src, (mixed,), cancel=Event())
        assert borrowed() is not None, "a partial hit must keep its entire batch"
        assert kernel.preparations == 6, "borrowed keys must not rebuild"
        assert kernel.live_bytes <= 48, "resident plus pending share one budget"
        assert max(kernel.live_before_build) <= 48


def test_prefix_pending_shares_capacity_with_future_suffix_hits():
    idle, driven = InventoryKernel(), InventoryKernel("dyson")
    src = source()
    # Include metadata for two separately retained products.
    with closing(
        Engine(
            1,
            32 + 2 * RESIDENT_METADATA_BYTES,
            kernels=(idle, driven),
            chunks_per_worker=1,
        )
    ) as engine:
        # This hit belongs to the later suffix, not the representative prefix.
        engine.evolve(src, (batch(2),), cancel=Event())
        suffix = idle.payloads[0]
        pulses = tuple(
            track(2, EnvelopeSpec("constant"), start=start, duration=1)
            for start in (1, 2)
        )
        timeline = replace(batch(2, duration=6), tracks=pulses)
        result = engine.evolve(src, (timeline,), cancel=Event())
        assert result.paths.segments_by_kernel == {"t4": 4, "dyson": 4}
        assert suffix() is not None, "prefix work must protect later suffix hits"
        assert idle.preparations == 2, "prefix work must not evict suffix keys"
        assert idle.live_bytes + driven.live_bytes <= 32, (
            "prefix products, suffix residents and worker offers share capacity"
        )
