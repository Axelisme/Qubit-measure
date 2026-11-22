"""Complete Lindblad operators through the public Engine seam."""

from contextlib import closing
from dataclasses import replace
from pathlib import Path
from threading import Event

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from zcu_tools.simulate.boundary import Cancelled
from zcu_tools.simulate.evolution import (
    Engine,
    EnvelopeSpec,
    FluxSchedule,
    ModelSource,
    NoValidKernel,
    SecularDrive,
    SecularModel,
    SegmentShift,
    common_frame_diagnostic,
    multitone_rwa_diagnostic,
    read_frozen_input,
    reference_key,
    single_tone_rwa_diagnostic,
    write_frozen_input,
)

from ._engine import batch, track


def test_one_photon_half_life_preserves_reduced_qubit_coherence() -> None:
    kappa = 0.26
    # Photon-major basis |0,g>, |0,e>, |1,g>, |1,e>. The two decay
    # components belong to the same bath operator, not independent channels.
    a = np.zeros((4, 4), dtype=complex)
    a[:2, 2:] = np.sqrt(kappa) * np.eye(2)
    qubit = np.array([1, 1j]) / np.sqrt(2)
    expected = np.outer(qubit, qubit.conj())
    initial = np.zeros((4, 4), dtype=complex)
    initial[2:, 2:] = expected
    with (
        closing(
            ModelSource(
                initial,
                lambda level: SecularModel(
                    level,
                    "one-photon",
                    np.array([0.0, 0.0, 0.71, 0.71]),
                    np.eye(4, dtype=complex),
                    (),
                    np.zeros((4, 4)),
                    np.zeros((4, 4)),
                    collapse_operators=(a,),
                ),
            )
        ) as source,
        closing(Engine(1, backend="cpu")) as engine,
    ):
        result = engine.evolve(source, (batch(1, duration=np.log(2) / kappa),))

    rho = result.densities[0]
    np.testing.assert_allclose(np.trace(rho[2:, 2:]), 0.5, atol=2e-6, rtol=0)
    np.testing.assert_allclose(rho[:2, :2] + rho[2:, 2:], expected, atol=2e-6, rtol=0)


def _two_level_source(*jumps: NDArray[np.complex128]) -> ModelSource:
    return ModelSource(
        np.diag([0.0, 1.0]).astype(complex),
        lambda level: SecularModel(
            level,
            "two-level",
            np.array([0.0, 0.3]),
            np.eye(2, dtype=complex),
            (),
            np.zeros((2, 2)),
            np.zeros((2, 2)),
            collapse_operators=jumps,
        ),
    )


def test_full_loss_refuses_flux_changes_even_without_drive() -> None:
    timeline = replace(
        batch(1),
        flux=FluxSchedule(np.array([0, 2]), np.array([0.0, 1.0]), np.array([[0], [1]])),
    )
    with (
        closing(
            _two_level_source(np.array([[0, 0.2], [0, 0]], dtype=complex))
        ) as source,
        closing(Engine(1, backend="cpu")) as engine,
    ):
        for run in (engine.inspect_dispatch, engine.evolve):
            with pytest.raises(NoValidKernel, match="collapse operators.*fixed flux"):
                run(source, (timeline,))


@pytest.mark.parametrize("split", [False, True], ids=["independent", "touching-spans"])
def test_phase_sensitive_jumps_agree_with_independent_lab_density(split: bool) -> None:
    energies = np.array([0.0, 0.19, 0.57])
    vectors = np.fft.fft(np.eye(3), axis=0) / np.sqrt(3)
    shift = np.array([0.01, -0.03, 0.04])
    duration, frequency, theta, gain = 7.5, 0.42, -0.5, 0.8
    # Recycling has complex cross terms; L†L is not diagonal.
    jump = np.array([[0, 0.18j, 0.13], [0, 0, 0.22 - 0.11j], [0, 0, 0]], complex)
    second = np.diag([0.02, 0.07j, 0.03]).astype(complex)
    x = np.array([[0, 0.07, 0], [0.07, 0, 0.025j], [0, -0.025j, 0]], complex)
    y = np.array([[0, -0.04j, 0], [0.04j, 0, 0.02], [0, 0.02, 0]], complex)
    initial_vector = np.array([0.0, 1j, 1.0]) / np.sqrt(2)
    initial = np.outer(initial_vector, initial_vector.conj())

    # Build an independent lab-frame solve in fixed coordinates.
    def in_lab(matrix: NDArray[np.complex128]) -> NDArray[np.complex128]:
        return vectors @ matrix @ vectors.conj().T

    lab_h = in_lab(np.diag(energies + shift).astype(complex))
    lab_x, lab_y = in_lab(x), in_lab(y)
    lab_jumps = tuple(in_lab(item) for item in (jump, second))

    def rhs(time: float, flat: NDArray[np.complex128]) -> NDArray[np.complex128]:
        rho = flat.reshape(3, 3)
        angle = 2 * np.pi * frequency * time + theta
        h = lab_h + gain * (np.cos(angle) * lab_x + np.sin(angle) * lab_y)
        out = -2j * np.pi * (h @ rho - rho @ h)
        for operator in lab_jumps:
            adjoint = operator.conj().T
            loss = adjoint @ operator
            out += operator @ rho @ adjoint - (loss @ rho + rho @ loss) / 2
        return out.ravel()

    reference = solve_ivp(
        rhs,
        (0.0, duration),
        in_lab(initial).ravel(),
        method="DOP853",
        rtol=2e-12,
        atol=2e-14,
    )
    assert reference.success
    pulse = replace(
        track(1, EnvelopeSpec("constant"), duration=duration),
        amplitude=np.array([gain]),
        frequency_ghz=np.array([frequency]),
        phase_rad=np.array([theta]),
    )
    zero = np.zeros((3, 3))
    midpoint = 2.25
    boundaries = (0.0, midpoint, duration) if split else (0.0, duration)
    timeline = replace(
        batch(1, duration=duration),
        tracks=(pulse,),
        segment_boundaries=(boundaries[1:-1],),
        segment_shifts=(
            tuple(
                SegmentShift(a, b, shift, zero)
                for a, b in zip(boundaries[:-1], boundaries[1:], strict=True)
            ),
        ),
    )
    with (
        closing(
            ModelSource(
                initial,
                lambda level: SecularModel(
                    level,
                    "phased-jumps",
                    energies,
                    vectors,
                    (SecularDrive("drive", x, y),),
                    zero,
                    zero,
                    collapse_operators=(jump, second),
                ),
            )
        ) as source,
        closing(Engine(1, families=("dop853",), backend="cpu")) as engine,
    ):
        result = engine.evolve(source, (timeline,), budget=1e-8)
    np.testing.assert_allclose(
        in_lab(result.densities[0]), reference.y[:, -1].reshape(3, 3), rtol=0, atol=2e-7
    )


def test_full_loss_falls_back_to_cpu_direct_and_honors_allowlist() -> None:
    timeline = batch(1)
    with (
        closing(
            _two_level_source(np.array([[0, 0.2], [0, 0]], dtype=complex))
        ) as source,
        closing(_two_level_source()) as control,
        closing(Engine(1)) as short_list,
        closing(Engine(1, families=("t4",), backend="cpu")) as excluded,
    ):
        (row,) = short_list.inspect_dispatch(source, (timeline,)).rows
        assert row.family == "dop853"
        actual = short_list.evolve(source, (timeline,))
        assert actual.paths.execution_device == "cpu"
        assert actual.paths.segments_by_kernel == {"dop853": 1}
        baseline = short_list.evolve(control, (timeline,))
        assert baseline.paths.segments_by_kernel == {"t4": 1}
        np.testing.assert_array_equal(baseline.densities[0], control.initial_density)
        for run in (excluded.inspect_dispatch, excluded.evolve):
            with pytest.raises(
                NoValidKernel, match="collapse operators require CPU dop853"
            ) as error:
                run(source, (timeline,))
            assert error.value.attempted == ()


def test_secular_guard_diagnostics_refuse_full_loss() -> None:
    zero = np.zeros((2, 2), dtype=complex)
    with closing(
        _two_level_source(np.array([[0, 0.2], [0, 0]], dtype=complex))
    ) as source:
        model = replace(
            source.model((0,)),
            drives=(
                SecularDrive("a", zero, zero),
                SecularDrive("b", zero, zero),
            ),
        )
    for query in (
        lambda: single_tone_rwa_diagnostic(model, "a", 0.5, 0.01, 3.0),
        lambda: multitone_rwa_diagnostic(
            model, ("a", "b"), (0.5, 0.6), (0.01, 0.01), 3.0
        ),
        lambda: common_frame_diagnostic(
            model, ("a", "b"), (0.5, 0.6), (0.01, 0.01), 3.0
        ),
    ):
        with pytest.raises(ValueError, match="collapse operators"):
            query()


def test_cancelled_full_loss_does_not_register_reuse() -> None:
    stop = Event()
    timeline = replace(
        batch(1),
        flux=FluxSchedule(np.array([0, 1]), np.array([0.0]), np.array([[1]])),
    )
    with closing(
        _two_level_source(np.array([[0, 0.2], [0, 0]], dtype=complex))
    ) as original:

        def build(level: tuple[int, ...]) -> SecularModel:
            stop.set()
            return original.model(level)

        with (
            closing(ModelSource(original.initial_density, build)) as source,
            closing(Engine(1, backend="cpu")) as engine,
        ):
            with pytest.raises(Cancelled):
                engine.evolve(source, (timeline,), cancel=stop, reuse_key=b"cancelled")
            assert not engine.reuse(source, b"cancelled")
            stop.clear()
            assert engine.evolve(source, (timeline,)).paths.segments_by_kernel == {
                "dop853": 1
            }


def test_frozen_full_loss_evolves_identically_and_distinguishes_keys(
    tmp_path: Path,
) -> None:
    timeline = batch(1)
    jump = np.array([[0, 0.1 + 0.2j], [0, 0]], dtype=complex)
    key = tmp_path / "full.json"
    with (
        closing(_two_level_source(jump)) as source,
        closing(_two_level_source(2 * jump)) as changed,
        closing(Engine(1, backend="cpu")) as engine,
    ):
        write_frozen_input(key, source, timeline)
        replay = read_frozen_input(key)
        with closing(replay.source):
            digest = reference_key(source, timeline).encode()
            actual = engine.evolve(source, (timeline,), reuse_key=digest)
            assert engine.reuse(source, digest)
            assert not engine.reuse(changed, digest)
            assert reference_key(changed, timeline) != digest.decode()
            assert reference_key(replay.source, replay.batch) == digest.decode()
            rebuilt = engine.evolve(replay.source, (replay.batch,))
            np.testing.assert_array_equal(rebuilt.densities, actual.densities)
