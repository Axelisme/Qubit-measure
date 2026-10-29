"""Inventory scheduling observed through Engine and injected kernel collaborators."""

from contextlib import closing
from dataclasses import replace
from threading import Event

import numpy as np
import pytest
from zcu_tools.simulate.evolution import (
    Engine,
    EnvelopeSpec,
    ModelSource,
    SecularDrive,
    SecularModel,
)
from zcu_tools.simulate.evolution.kernel import ApplyContext, KernelContext

from ._engine import RecordingKernel, batch, source, track


class RelaxingKernel(RecordingKernel):
    """Cheap nonidentity dynamics, optionally with timeline-local prepare keys."""

    def __init__(self, name: str, *, local: bool = False):
        super().__init__(name)
        self.local = local

    def prepare_keys(self, table, rows, context: KernelContext):
        keys = np.empty(len(rows), dtype=[("duration", "f8"), ("owner", "i8")])
        keys["duration"] = table.duration_ns[rows]
        keys["owner"] = table.timeline[rows] if self.local else 0
        return keys

    def apply(self, prepared, key_index, table, rows, states, context: ApplyContext):
        transfer = states[:, 1, 1] * 0.125
        states[:, 0, 0] += transfer
        states[:, 1, 1] -= transfer
        return super().apply(prepared, key_index, table, rows, states, context)


def gate_train(n: int, gates: int = 40, *, simultaneous: bool = False):
    # Cold gate selection includes numerical admission. Use a smooth 128 ns
    # ramp, not a 3 ns spike that must fall back despite the injected guard.
    # Forty uses also amortize the gate build for the single-timeline case.
    duration = 128
    spacing = duration + 1
    envelope = EnvelopeSpec(
        "sampled",
        np.array([0, 1, 0], dtype=np.complex128),
        np.array([0.0, duration / 2, duration]),
    )
    tracks = tuple(
        replace(
            track(n, envelope, start=1 + spacing * gate, duration=duration),
            frequency_ghz=np.full(n, 1.1),
            amplitude=np.full(n, 0.001),
        )
        for gate in range(gates)
    )
    if simultaneous:
        tracks += tuple(replace(item, frequency_ghz=np.full(n, 0.7)) for item in tracks)
    return replace(batch(n, duration=spacing * gates + 1), tracks=tracks)


def evolve_train(
    n: int,
    chunks: int | None,
    *,
    local: bool = False,
    gates: int = 40,
    simultaneous: bool = False,
    workers: int = 1,
):
    idle = RelaxingKernel("t4", local=local)
    driven = RelaxingKernel(
        "multitone_gate" if simultaneous else "single_tone_gate", local=local
    )
    options = {} if chunks is None else {"chunks_per_worker": chunks}
    with closing(Engine(workers, kernels=(idle, driven), **options)) as engine:
        result = engine.evolve(
            source(), (gate_train(n, gates, simultaneous=simultaneous),), cancel=Event()
        )
    return result, len(idle.calls) + len(driven.calls)


@pytest.mark.parametrize("timelines", [1, 21])
@pytest.mark.parametrize("local", [False, True])
def test_chunk_strategies_preserve_density_bytes_and_routes(timelines, local):
    expected, _ = evolve_train(timelines, 6, local=local)
    assert expected.paths.segments_by_kernel == {
        "t4": timelines * 41,
        "single_tone_gate": timelines * 40,
    }
    assert not np.array_equal(expected.densities[0], source().initial_density)
    for chunks in (1, None):
        actual, _ = evolve_train(timelines, chunks, local=local)
        assert actual.densities.tobytes() == expected.densities.tobytes()
        assert actual.paths == expected.paths
        np.testing.assert_array_equal(actual.points, expected.points)


@pytest.mark.parametrize("simultaneous", [False, True])
@pytest.mark.parametrize("workers", [1, 2])
def test_complete_short_gate_inventory_batches_more_timelines_by_default(
    simultaneous, workers
):
    expected, legacy_calls = evolve_train(
        21, 6, simultaneous=simultaneous, workers=workers
    )
    actual, selected_calls = evolve_train(
        21, None, simultaneous=simultaneous, workers=workers
    )
    assert actual.densities.tobytes() == expected.densities.tobytes()
    assert actual.paths == expected.paths
    assert selected_calls < legacy_calls, "short gate inventory should use fewer chunks"


def test_integrator_heavy_inventory_keeps_existing_chunk_strategy():
    observations = []
    for options in ({"chunks_per_worker": 6}, {}):
        idle, driven = RelaxingKernel("t4"), RelaxingKernel("multitone_rwa")
        with closing(Engine(1, kernels=(idle, driven), **options)) as engine:
            result = engine.evolve(
                source(), (gate_train(21, simultaneous=True),), cancel=Event()
            )
        assert result.paths.segments_by_kernel["multitone_rwa"] == 21 * 40
        observations.append(
            (result.densities.tobytes(), len(idle.calls) + len(driven.calls))
        )
    assert observations[0] == observations[1]


def test_small_gate_inventory_keeps_existing_chunk_strategy():
    _, legacy_calls = evolve_train(21, 6, gates=2)
    _, selected_calls = evolve_train(21, None, gates=2)
    assert selected_calls == legacy_calls


def test_long_shaped_timelines_preserve_independent_phases_and_densities():
    count, pulses, duration = 4, 32, 20.0
    operator = np.array([[0, 0.012], [0.012, 0]], dtype=complex)
    model = SecularModel(
        (0,),
        "independent-shaped-timelines",
        np.array([0.0, 0.1]),
        np.eye(2, dtype=complex),
        (SecularDrive("drive", operator, np.zeros_like(operator)),),
        np.array([[0.0, 1e-5], [0.0, 0.0]]),
        np.zeros((2, 2)),
    )
    phases = np.arange(count) * 0.7
    gaps = 2.0 + np.arange(count)
    initial = np.array(
        [
            [[0.6, 0.1 * np.exp(1j * phase)], [0.1 * np.exp(-1j * phase), 0.4]]
            for phase in phases
        ]
    )
    envelope = EnvelopeSpec(
        "sampled",
        np.array([0.0, 1.0, 0.02], dtype=complex),
        np.array([0.0, duration / 2, duration]),
    )
    timeline = replace(
        batch(count),
        tracks=tuple(
            replace(
                track(count, envelope, duration=duration),
                start_ns=gaps + pulse * (duration + gaps),
                phase_rad=phases,
            )
            for pulse in range(pulses)
        ),
        end_ns=pulses * (duration + gaps) + gaps,
        start_ns=np.zeros(count),
        initial_densities=initial,
    )
    results = []
    with closing(ModelSource(initial[0], lambda _: model)) as src:
        for workers, chunks in ((1, 6), (1, None), (2, None)):
            with closing(Engine(workers, chunks_per_worker=chunks)) as engine:
                results.append(engine.evolve(src, (timeline,), cancel=Event()))
    expected = results[0]
    assert expected.paths.segments_by_kernel == {
        "t4": count * (pulses + 1),
        "shaped_sambe": count * pulses,
    }
    assert not np.allclose(expected.densities[0], expected.densities[1])
    for actual in results[1:]:
        np.testing.assert_allclose(
            actual.densities, expected.densities, rtol=1e-8, atol=1e-10
        )
        assert actual.paths.segments_by_kernel == expected.paths.segments_by_kernel
