"""Constant-tone full-density evolution across Engine lifecycle boundaries."""

from contextlib import closing
from dataclasses import replace

import numpy as np
from zcu_tools.simulate.evolution import (
    Engine,
    EnvelopeSpec,
    SegmentShift,
    TimelineBatch,
)

from .contract.accuracy_cases import coherent_source, drive_track, timeline
from .contract.reference import converged_reference
from .contract.synthetic_models import synthetic_model


def test_t6_density_is_independent_of_workers_order_and_history() -> None:
    count = 4
    source = coherent_source(synthetic_model())
    batch = timeline(count, 16.0)
    tone = drive_track(batch, EnvelopeSpec("constant"), frequency=0.7, gain=0.15)
    batch = replace(batch, tracks=(tone,))
    expected = converged_reference(source, batch).density
    tolerance = 1e-3
    for workers in (1, 2):
        with closing(Engine(workers, families=("t6",))) as engine:
            source.history.clear()
            for clear in (False, False, True):
                if clear:
                    source.history.clear()
                result = engine.evolve(source, (batch,), telemetry=True)
                assert result.paths.segments_by_kernel == {"t6": count}
                assert result.paths.routing["t6"].rows == count
                np.testing.assert_allclose(
                    result.densities, expected, rtol=0, atol=tolerance
                )
            reversed_batch = replace(
                batch,
                points=batch.points[::-1],
                tracks=(replace(tone, phase_rad=tone.phase_rad[::-1]),),
            )
            result = engine.evolve(source, (reversed_batch,))
            np.testing.assert_allclose(
                result.densities, expected[::-1], rtol=0, atol=tolerance
            )


def test_t6_direct_continuation_preserves_carrier_origin() -> None:
    source = coherent_source(synthetic_model())
    batch = timeline(3, 3.0)
    batch = replace(
        batch,
        tracks=(
            drive_track(batch, EnvelopeSpec("constant"), frequency=-0.7, gain=0.15),
        ),
        segment_boundaries=((1.25,),) * 3,
    )
    expected = converged_reference(source, batch).density
    with closing(Engine(1, families=("t6",))) as engine:
        prefix = engine.evolve(source, (replace(batch, end_ns=np.full(3, 1.25)),))
        continued = engine.evolve(
            source,
            (
                replace(
                    batch, start_ns=np.full(3, 1.25), initial_densities=prefix.densities
                ),
            ),
        )
    np.testing.assert_allclose(continued.densities, expected, rtol=0, atol=1e-3)


def test_translated_start_and_phase_preserve_direct_density() -> None:
    source = coherent_source(synthetic_model())
    batch = timeline(3, 2.0)
    tone = drive_track(batch, EnvelopeSpec("constant"), frequency=0.7, gain=0.15)
    batch = replace(batch, tracks=(tone,))
    expected = converged_reference(source, batch).density
    offset = 1.37
    translated = replace(
        batch,
        end_ns=batch.end_ns + offset,
        start_ns=np.full(3, offset),
        initial_densities=np.repeat(source.initial_density[None], 3, axis=0),
        tracks=(
            replace(
                tone,
                start_ns=tone.start_ns + offset,
                phase_rad=tone.phase_rad - 2 * np.pi * tone.frequency_ghz * offset,
            ),
        ),
    )
    with closing(Engine(1, families=("t6",))) as engine:
        result = engine.evolve(source, (translated,))
    np.testing.assert_allclose(result.densities, expected, rtol=0, atol=1e-3)


def test_t6_direct_uses_shifted_energies_and_bath() -> None:
    model = synthetic_model()
    source = coherent_source(model)
    delta = np.array([0.0, 0.021, 0.037])
    slopes = np.array([0.0, 0.01, 0.02])
    decay = (slopes[:, None] - slopes[None, :]) ** 2 / 2
    shifted_model = replace(
        model,
        identity="shifted-t6-model",
        energies_ghz=model.energies_ghz + delta,
        dephasing_per_ns=model.dephasing_per_ns + decay,
    )
    batch = timeline(3, 2.0)
    batch = replace(
        batch,
        tracks=(
            drive_track(batch, EnvelopeSpec("constant"), frequency=0.7, gain=0.15),
        ),
    )
    expected = converged_reference(coherent_source(shifted_model), batch).density
    shifted: TimelineBatch = replace(
        batch, segment_shifts=((SegmentShift(0, 2, delta, decay),),) * 3
    )
    with closing(Engine(2, families=("t6",))) as engine:
        result = engine.evolve(source, (shifted,))
    np.testing.assert_allclose(result.densities, expected, rtol=0, atol=1e-3)
