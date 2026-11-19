"""Full-density numerical contracts through the public evolution engine."""

from contextlib import closing
from dataclasses import replace
from threading import Event

import numpy as np
import pytest
from zcu_tools.simulate.evolution import Engine, EnvelopeSpec, ModelSource

from ._engine import RecordingKernel, batch, source, track
from .contract.kernel_suite import synthetic_model


@pytest.mark.parametrize("budget", [2e-6, 5e-7], ids=["within-budget", "over-budget"])
def test_evolve_checks_readout_density_correction_against_budget(budget: float) -> None:
    expected = np.array([[0.5, 0.5j, 0], [-0.5j, 0.5, 0], [0, 0, 0]])
    invalid = expected.copy()
    invalid[0, 1] += 1e-6j
    invalid[1, 0] -= 1e-6j

    class EndpointKernel(RecordingKernel):
        def apply(self, prepared, key_index, table, rows, states, context):
            states[:] = invalid
            return np.zeros(len(rows), dtype=np.bool_)

    timeline = replace(batch(1), points=np.array([17]))
    with (
        closing(source()) as src,
        closing(Engine(1, kernels=(EndpointKernel("t4"),))) as engine,
    ):
        if budget < 1e-6:
            with pytest.raises(
                RuntimeError, match=r"point 17.*density.*exceeds budget"
            ):
                engine.evolve(src, (timeline,), budget=budget)
            return

        result = engine.evolve(src, (timeline,), budget=budget)

    # Building a continuation runs the unchanged density input validator.
    replace(timeline, start_ns=timeline.end_ns, initial_densities=result.densities)
    np.testing.assert_allclose(result.densities[0], expected, rtol=0, atol=1e-14)
    assert np.max(np.abs(result.densities[0] - invalid)) == pytest.approx(1e-6)


@pytest.mark.parametrize("coherent", [False, True], ids=["mixed", "coherent"])
@pytest.mark.parametrize("gap", [0.0, 1e-8, 0.13])
@pytest.mark.parametrize("bath", ["none", "symmetric", "cycle"])
def test_weak_constant_drive_preserves_full_density_with_degenerate_modes(
    coherent: bool, gap: float, bath: str
) -> None:
    from scipy.integrate import solve_ivp

    model = synthetic_model()
    rates = np.zeros((3, 3))
    if bath == "symmetric":
        rates[:] = 0.004
        np.fill_diagonal(rates, 0)
    elif bath == "cycle":
        # A directed cycle has complex-conjugate population eigenmodes.
        rates[0, 1] = rates[1, 2] = rates[2, 0] = 0.004
    slopes = np.array([0.0, 0.03, 0.03]) if bath != "none" else np.zeros(3)
    phases = np.exp(1j * np.array([0.0, 0.4, -0.7]))
    rotation = phases[:, None] * phases.conj()[None, :]
    model = replace(
        model,
        energies_ghz=np.arange(3) * gap,
        rates_per_ns=rates,
        dephasing_per_ns=(slopes[:, None] - slopes[None, :]) ** 2 / 2,
        drives=(
            replace(
                model.drives[0],
                in_phase_ghz=model.drives[0].in_phase_ghz * rotation,
                quadrature_ghz=0.6 * model.drives[0].quadrature_ghz * rotation,
            ),
        ),
    )
    weights = np.array([0.5, 0.3, 0.2])
    vector = np.sqrt(weights) * phases
    initial = np.outer(vector, vector.conj()) if coherent else np.diag(weights)
    src = ModelSource(
        initial.astype(complex), lambda level: replace(model, level=level)
    )
    pulse = replace(
        track(4, EnvelopeSpec("constant"), duration=2.5),
        amplitude=np.array([1e-8, -2e-8, 0.0, 3e-8]),
        frequency_ghz=np.array([0.1, 0.15, 0.2, 0.12]),
        phase_rad=np.array([0.3, -0.6, 1.1, 0.0]),
    )
    b = replace(batch(4, duration=2.5), tracks=(pulse,))
    expected = []
    for gain, frequency, phase in zip(
        pulse.amplitude, pulse.frequency_ghz, pulse.phase_rad, strict=True
    ):

        def rhs(time, flat, gain=gain, frequency=frequency, phase=phase):
            rho = flat.reshape(3, 3)
            angle = 2 * np.pi * frequency * time + phase
            h = np.diag(model.energies_ghz).astype(complex)
            h += gain * (
                np.cos(angle) * model.drives[0].in_phase_ghz
                + np.sin(angle) * model.drives[0].quadrature_ghz
            )
            result = -2j * np.pi * (h @ rho - rho @ h)
            result -= model.dephasing_per_ns * rho
            for target, origin in np.argwhere(rates > 0):
                jump = np.zeros((3, 3), complex)
                jump[target, origin] = np.sqrt(rates[target, origin])
                loss = jump.conj().T @ jump
                result += jump @ rho @ jump.conj().T - (loss @ rho + rho @ loss) / 2
            return result.ravel()

        solution = solve_ivp(
            rhs,
            (0, 2.5),
            src.initial_density.ravel(),
            method="DOP853",
            rtol=2e-12,
            atol=2e-14,
        )
        assert solution.success
        expected.append(solution.y[:, -1].reshape(3, 3))
    outputs = []
    for workers in (1, 2):
        with closing(Engine(workers)) as engine:
            actual = engine.evolve(src, (b,), budget=1e-9, cancel=Event())
        assert actual.paths.segments_by_kernel == {"dyson": 3, "t4": 1}
        assert actual.paths.exceeded_by_kernel == {}
        np.testing.assert_allclose(actual.densities, expected, rtol=0, atol=1e-9)
        np.testing.assert_allclose(
            actual.densities,
            actual.densities.conj().swapaxes(-1, -2),
            rtol=0,
            atol=1e-13,
        )
        np.testing.assert_allclose(
            np.trace(actual.densities, axis1=-2, axis2=-1), 1, rtol=0, atol=1e-13
        )
        outputs.append(actual.densities)
    np.testing.assert_array_equal(outputs[0], outputs[1])
