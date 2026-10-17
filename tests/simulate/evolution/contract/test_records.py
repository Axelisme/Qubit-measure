"""Construction is the boundary that rejects malformed engine inputs."""

from __future__ import annotations

from collections.abc import Callable, MutableMapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier

import numpy as np
import pytest
import zcu_tools.simulate.evolution as evolution

from .kernel_suite import synthetic_model


def test_model_preserves_directional_rates_and_owns_immutable_arrays() -> None:
    model = synthetic_model(4)
    rates = model.rates_per_ns.copy()
    copied = replace(model, rates_per_ns=rates)
    rates[:] = 0
    assert copied.rates_per_ns[0, 1] == 0.015
    assert copied.rates_per_ns[1, 0] == 0.002
    assert not copied.rates_per_ns.flags.writeable


@pytest.mark.parametrize(
    "fault",
    [
        "shape",
        "hermitian",
        "rates_diagonal",
        "gamma_symmetry",
        "gamma_diagonal",
        "negative_rate",
        "nonfinite",
        "basis",
        "energies",
        "drive_shape",
    ],
)
def test_model_rejects_invalid_secular_data(fault: str) -> None:
    model = synthetic_model()
    constructions: dict[str, tuple[Callable[[], object], str]] = {
        "shape": (lambda: replace(model, rates_per_ns=np.zeros((2, 2))), "shape"),
        "hermitian": (
            lambda: evolution.SecularDrive(
                "bad", np.triu(np.ones((3, 3), complex)), np.zeros((3, 3), complex)
            ),
            "Hermitian",
        ),
        "rates_diagonal": (
            lambda: replace(model, rates_per_ns=model.rates_per_ns + np.eye(3)),
            "diagonal",
        ),
        "gamma_symmetry": (
            lambda: replace(model, dephasing_per_ns=np.triu(np.ones((3, 3)), 1)),
            "symmetric",
        ),
        "gamma_diagonal": (
            lambda: replace(model, dephasing_per_ns=np.eye(3)),
            "diagonal",
        ),
        "negative_rate": (
            lambda: replace(model, rates_per_ns=-model.rates_per_ns),
            "nonnegative",
        ),
        "nonfinite": (
            lambda: replace(model, energies_ghz=np.array([0.0, 1.0, np.nan])),
            "finite",
        ),
        "basis": (
            lambda: replace(model, eigenvectors=2 * model.eigenvectors),
            "unitary",
        ),
        "energies": (
            lambda: replace(model, energies_ghz=np.array([0.0, 2.0, 1.0])),
            "ascending",
        ),
        "drive_shape": (
            lambda: replace(
                model,
                drives=(
                    evolution.SecularDrive(
                        "small", np.eye(2, dtype=complex), np.eye(2, dtype=complex)
                    ),
                ),
            ),
            "shape",
        ),
    }
    construct, keyword = constructions[fault]
    with pytest.raises(ValueError, match=keyword):
        construct()


def test_flux_records_validate_ragged_layout_and_hamiltonian() -> None:
    schedule = evolution.FluxSchedule(
        np.array([0, 2, 3]), np.array([0.0, 2.0, 0.0]), np.array([[0], [1], [0]])
    )
    assert schedule.level[1, 0] == 1
    waveform = evolution.FluxWaveform(
        np.array([0, 2, 4]), np.array([0.0, 2.0, 0.0, 3.0]), np.zeros((4, 1))
    )
    assert waveform.flux.shape == (4, 1)
    term = evolution.FluxTerm(0, "cos_2pi", np.eye(3, dtype=complex))
    expansion = evolution.FluxExpansion(np.zeros((3, 3), complex), (term,))
    assert expansion.terms[0].loop == 0
    with pytest.raises(ValueError, match="shape"):
        evolution.FluxExpansion(np.eye(2, dtype=complex), (term,))
    with pytest.raises(ValueError, match="Hermitian"):
        evolution.FluxTerm(0, "cos_2pi", np.triu(np.ones((3, 3), complex)))


@pytest.mark.parametrize("offsets", [[0, 3, 2, 3], [1, 2, 3], [0, 1, 4], [0, 0, 3]])
def test_ragged_offsets_reject_missing_or_reversed_segments(offsets: list[int]) -> None:
    with pytest.raises(ValueError, match="offsets"):
        evolution.FluxSchedule(
            np.array(offsets),
            np.array([0.0, 1.0, 2.0]),
            np.zeros((3, 1), dtype=np.int64),
        )
    with pytest.raises(ValueError, match="offsets"):
        evolution.FluxWaveform(
            np.array(offsets), np.array([0.0, 1.0, 2.0]), np.zeros((3, 1))
        )


def test_ragged_times_must_increase_inside_each_point() -> None:
    with pytest.raises(ValueError, match="increase"):
        evolution.FluxSchedule(
            np.array([0, 3]),
            np.array([0.0, 2.0, 1.0]),
            np.zeros((3, 1), dtype=np.int64),
        )
    with pytest.raises(ValueError, match="increase"):
        evolution.FluxWaveform(
            np.array([0, 3]), np.array([0.0, 2.0, 1.0]), np.zeros((3, 1))
        )


def test_timeline_shapes_and_envelope_content_identity() -> None:
    envelope = evolution.EnvelopeSpec("constant")
    assert envelope.identity == evolution.EnvelopeSpec("constant").identity
    sampled = evolution.EnvelopeSpec(
        "sampled", samples=np.array([1, 1j], complex), sample_ns=np.array([0.0, 1.0])
    )
    changed = replace(sampled, samples=np.array([1, -1j], complex))
    assert sampled.identity != changed.identity
    track = evolution.DriveTrack(
        "drive",
        np.zeros(2),
        np.ones(2),
        np.full(2, 0.1),
        np.zeros(2),
        np.ones(2),
        (envelope,),
        np.zeros(2, dtype=np.int64),
        np.ones(2, dtype=np.bool_),
    )
    flux = evolution.FluxSchedule(
        np.array([0, 1, 2]), np.zeros(2), np.zeros((2, 1), dtype=np.int64)
    )
    batch = evolution.TimelineBatch(np.array([2, 7]), (track,), flux, np.full(2, 3.0))
    assert batch.points.tolist() == [2, 7]
    with pytest.raises(ValueError, match="shape"):
        replace(track, duration_ns=np.ones(3))
    with pytest.raises(ValueError, match="index"):
        replace(track, envelope_index=np.full(2, 1, dtype=np.int64))
    with pytest.raises(ValueError, match="shape"):
        replace(batch, end_ns=np.ones(1))
    with pytest.raises(ValueError, match="zero"):
        replace(sampled, sample_ns=np.array([1.0, 0.0]))
    result = evolution.EvolutionResult(
        batch.points,
        np.repeat(
            (np.eye(3, dtype=np.complex128)[None] / 3).astype(np.complex128), 2, axis=0
        ),
        evolution.PathDiagnostics(),
    )
    with pytest.raises(ValueError, match="shape"):
        replace(result, densities=np.zeros((1, 3, 3), complex))


def test_model_source_caches_each_level_across_threads() -> None:
    calls: list[evolution.FluxLevel] = []

    def build(level: evolution.FluxLevel) -> evolution.SecularModel:
        calls.append(level)
        return replace(synthetic_model(), level=level, identity=f"model-{level}")

    source = evolution.ModelSource(
        (np.eye(3, dtype=np.complex128) / 3).astype(np.complex128), build
    )
    barrier = Barrier(2)

    def get_model(_: int) -> evolution.SecularModel:
        barrier.wait(timeout=10)  # All submitted workers reach this rendezvous.
        return source.model((0,))

    with ThreadPoolExecutor(max_workers=2) as pool:
        models = list(pool.map(get_model, range(2)))
    assert len(calls) == 1
    assert all(model is models[0] for model in models)
    assert source.model((1,)).level == (1,)
    assert calls == [(0,), (1,)]
    with pytest.raises(ValueError, match="expansion"):
        source.flux_expansion()
    expansion = evolution.FluxExpansion(np.eye(3, dtype=complex), ())
    assert (
        evolution.ModelSource(source.initial_density, build, expansion).flux_expansion()
        is expansion
    )


def test_model_source_propagates_builder_errors_without_caching_failure() -> None:
    calls = 0

    def build(level: evolution.FluxLevel) -> evolution.SecularModel:
        nonlocal calls
        calls += 1
        raise ValueError(f"secular bath unavailable at {level}")

    source = evolution.ModelSource(
        (np.eye(3, dtype=np.complex128) / 3).astype(np.complex128), build
    )
    for _ in range(2):
        with pytest.raises(ValueError, match="secular bath unavailable at"):
            source.model((0,))
    assert calls == 2


def test_sampled_envelope_interpolates_complex_values_and_holds_last_sample() -> None:
    envelope = evolution.EnvelopeSpec(
        "sampled",
        samples=np.array([0, 2 + 4j, 4 - 2j], dtype=np.complex128),
        sample_ns=np.array([0.0, 2.0, 3.0]),
    )
    actual = envelope.values(np.array([0.0, 0.5, 1.0, 2.5, 3.0, 7.0]))
    np.testing.assert_allclose(
        actual, [0, 0.5 + 1j, 1 + 2j, 3 + 1j, 4 - 2j, 4 - 2j], rtol=0, atol=1e-14
    )
    np.testing.assert_array_equal(
        evolution.EnvelopeSpec("constant").values(np.array([0.0, 3.0])), [1, 1]
    )


def test_sampled_envelope_requires_zero_time_origin() -> None:
    with pytest.raises(ValueError, match="zero"):
        evolution.EnvelopeSpec(
            "sampled",
            samples=np.ones(2, dtype=np.complex128),
            sample_ns=np.array([1.0, 2.0]),
        )


def test_exceeded_diagnostics_default_validation_and_immutable_copy() -> None:
    default = evolution.PathDiagnostics()
    assert getattr(default, "exceeded_by_kernel", None) == {}
    counts = {"dyson": 2, "t4": 0}
    diagnostics = replace(default, exceeded_by_kernel=counts)
    counts["dyson"] = 99
    assert diagnostics.exceeded_by_kernel == {"dyson": 2}
    assert not isinstance(diagnostics.exceeded_by_kernel, MutableMapping)
    for invalid in ({"dyson": -1}, {"": 1}):
        with pytest.raises(ValueError, match="names|nonnegative"):
            replace(default, exceeded_by_kernel=invalid)
