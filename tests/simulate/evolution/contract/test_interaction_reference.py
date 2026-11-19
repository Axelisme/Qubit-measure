"""Public density, convergence and input-domain contracts of the exact reference."""

import json
from contextlib import closing
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from zcu_tools.simulate.evolution import (
    EnvelopeSpec,
    FluxSchedule,
    ModelSource,
    RoutingHistory,
    SecularDrive,
    SecularModel,
    TimelineBatch,
    read_frozen_input,
    write_frozen_input,
)

from . import interaction_reference as solver
from .accuracy_cases import (
    AccuracyCase,
    accuracy_case,
    drive_track,
    history_case,
    multisegment_case,
    multitone_phase_case,
    timeline,
)
from .guard_edge_cases import EDGE_FAMILIES, guard_edge_case
from .interaction_reference import interaction_reference
from .reference import SolverTolerances, converged_reference, density_error

pytestmark = pytest.mark.sim

_MATRIX_NAMES = (
    "idle_coherence",
    "weak_plateau",
    "strong_plateau",
    "shaped_complex",
    "two_carriers",
    "continuous_flux",
    "joint_gate_inventory",
    "low_carrier",
    "linear_shaped",
    "linear_plateau",
    "nearby_carriers",
    "adiabatic_ramp",
    *("guard_edge/" + family for family in EDGE_FAMILIES),
    "multisegment/final",
    "history/2",
    "history/8",
)


def _matrix_case(name: str) -> AccuracyCase:
    if name.startswith("guard_edge/"):
        return guard_edge_case(name.split("/")[1])
    if name.startswith("history/"):
        return history_case(int(name.split("/")[1]), RoutingHistory())
    if name == "multisegment/final":
        return multisegment_case()
    return accuracy_case(name)


@pytest.mark.parametrize("name", _MATRIX_NAMES)
def test_accuracy_matrix_matches_lab_frame(
    name: str, request: pytest.FixtureRequest
) -> None:
    case = _matrix_case(name)
    if case.batch.flux_waveform is not None:
        pytest.skip(
            "continuous flux waveform is outside the fixed-schedule reference domain"
        )
    _assert_lab_parity(case.source, case.batch, name, request)


def _assert_lab_parity(
    source: ModelSource,
    batch: TimelineBatch,
    name: str,
    request: pytest.FixtureRequest,
) -> None:
    result = interaction_reference(source, batch)
    expected = converged_reference(source, batch)
    error = density_error(result.density, expected.density)
    request.node.user_properties.append(
        (
            "interaction_comparison",
            json.dumps(
                {
                    "case": name,
                    "points": len(batch.points),
                    "density_error": error,
                    "interaction_convergence": result.convergence_error,
                    "lab_convergence": expected.convergence_error,
                }
            ),
        )
    )
    assert error <= 2e-6, f"{name}: density error {error:.9g} > 2e-6"


def test_resonant_rabi_matches_analytic_lab_density() -> None:
    frequency, gain, phase, end = 2.0, 0.2, 0.31, 0.8
    x = np.array([[0, 1], [1, 0]], complex)
    y = np.array([[0, 1j], [-1j, 0]], complex)
    model = SecularModel(
        (0,),
        "rabi-doublet",
        np.array([0.0, frequency]),
        np.eye(2, dtype=complex),
        (SecularDrive("drive", x, y),),
        np.zeros((2, 2)),
        np.zeros((2, 2)),
    )
    source = ModelSource(np.diag([0, 1]).astype(complex), lambda level: model)
    batch = timeline(1, end)
    track = replace(
        drive_track(batch, EnvelopeSpec("constant"), frequency=frequency, gain=gain),
        phase_rad=np.array([phase]),
    )
    angle = 2 * np.pi * gain * end
    state = np.array(
        [
            -1j * np.exp(1j * phase) * np.sin(angle),
            np.exp(-2j * np.pi * frequency * end) * np.cos(angle),
        ]
    )
    result = interaction_reference(source, replace(batch, tracks=(track,)))
    np.testing.assert_allclose(
        result.density, np.outer(state, state.conj())[None], atol=2e-10, rtol=0
    )
    assert result.convergence_error <= 2e-6


def test_continuation_keeps_absolute_carrier_and_local_envelope_time(
    request: pytest.FixtureRequest,
) -> None:
    case = accuracy_case("shaped_complex")
    start = np.full(2, 0.6)
    initial = np.array([np.diag([1, 0, 0]), np.diag([0, 1, 0])], complex)
    batch = replace(
        case.batch,
        segment_boundaries=((0.6,),) * 2,
        start_ns=start,
        initial_densities=initial,
    )
    _assert_lab_parity(case.source, batch, "continuation", request)


def test_single_tolerance_chains_resume_from_frozen_density(tmp_path: Path) -> None:
    case = multitone_phase_case("drag", 3, gain=20.0)
    # Cut during the second pair, at a schedule boundary. Both sampled drives
    # continue across it with their original origins and nonzero carrier phase.
    cut = 5.5
    batch = replace(case.batch, segment_boundaries=((cut,),))
    first = replace(batch, end_ns=np.array([cut]))
    paths = [tmp_path / f"{chain}.json" for chain in ("coarse", "fine")]
    # Loose in this synthetic case so reusing the other chain's density is
    # detectable, even though both solves land close to the converged reference.
    tolerances = (SolverTolerances(1e-4, 1e-6), SolverTolerances(2e-12, 2e-14))
    with closing(case.source):
        for path, tolerance in zip(paths, tolerances, strict=True):
            density = solver.interaction_density(case.source, first, tolerance)
            continuation = replace(
                batch, start_ns=np.array([cut]), initial_densities=density
            )
            write_frozen_input(path, case.source, continuation)
        for path, tolerance in zip(paths, tolerances, strict=True):
            expected = interaction_reference(
                case.source, batch, coarse=tolerance, fine=tolerance
            ).density
            checkpoint = read_frozen_input(path)
            with closing(checkpoint.source):
                result = solver.interaction_density(
                    checkpoint.source, checkpoint.batch, tolerance
                )
            np.testing.assert_allclose(result, expected, rtol=0, atol=1e-12)


def test_nonzero_schedule_level_returns_density_in_dc_basis(
    request: pytest.FixtureRequest,
) -> None:
    case = accuracy_case("linear_plateau")
    base = case.source.model((0,))
    dc_vectors = np.array([[1, 1j], [1j, 1]], complex) / np.sqrt(2)
    rotation = np.array(
        [[np.cos(0.4), -np.sin(0.4)], [np.sin(0.4), np.cos(0.4)]], complex
    )
    dc = replace(base, eigenvectors=dc_vectors)
    shifted = replace(
        base,
        level=(2,),
        identity="shifted-doublet",
        eigenvectors=dc_vectors @ rotation,
        energies_ghz=np.array([0.1, 0.47]),
        rates_per_ns=base.rates_per_ns * 1000,
        dephasing_per_ns=base.dephasing_per_ns * 1000,
    )
    source = ModelSource(
        case.source.initial_density, lambda level: dc if level == (0,) else shifted
    )
    batch = replace(case.batch, flux=replace(case.batch.flux, level=np.full((2, 1), 2)))
    _assert_lab_parity(source, batch, "nonzero_level", request)


def test_shaped_multitone_sequence_matches_lab_frame(
    request: pytest.FixtureRequest,
) -> None:
    case = accuracy_case("joint_gate_inventory")
    first, second = case.batch.tracks
    envelope = EnvelopeSpec(
        "sampled",
        np.array([0.2 + 0.3j, 1 - 0.2j, 0.5 + 0.4j, 0.0]),
        np.array([0.0, 0.2, 0.4, 0.9]),
    )
    first = replace(
        first,
        start_ns=np.full(20, 0.1),
        duration_ns=np.full(20, 0.9),
        envelopes=(envelope,),
        amplitude=np.full(20, 4.0),
    )
    second = replace(
        second,
        start_ns=np.full(20, 0.4),
        duration_ns=np.full(20, 0.7),
        amplitude=np.full(20, 2.0),
        active=np.arange(20) % 2 == 0,
    )
    third = replace(second, port=first.port, frequency_ghz=np.full(20, 0.13))
    batch = replace(
        case.batch,
        tracks=(first, second, third),
        segment_boundaries=((0.3, 1.2),) * 20,
    )
    _assert_lab_parity(case.source, batch, "shaped_multitone_sequence", request)


def test_rejects_schedule_level_changes_within_shot() -> None:
    case = accuracy_case("weak_plateau")
    flux = FluxSchedule(
        np.array([0, 2, 4]),
        np.array([0.0, 0.7, 0.0, 0.7]),
        np.array([[0], [1], [0], [1]]),
    )
    with pytest.raises(ValueError, match="flux.*changes"):
        interaction_reference(case.source, replace(case.batch, flux=flux))


@pytest.mark.parametrize("constant", [False, True])
def test_rejects_continuous_waveform_even_when_constant(constant: bool) -> None:
    case = accuracy_case("continuous_flux")
    batch = case.batch
    if constant:
        waveform = batch.flux_waveform
        assert waveform is not None
        batch = replace(
            batch, flux_waveform=replace(waveform, flux=np.zeros_like(waveform.flux))
        )
    with pytest.raises(ValueError, match="continuous flux waveform.*unsupported"):
        interaction_reference(case.source, batch)


def test_rejects_unconverged_density() -> None:
    case = accuracy_case("linear_plateau")
    batch = replace(
        case.batch,
        end_ns=np.full(2, 0.413),
        tracks=(replace(case.batch.tracks[0], amplitude=np.full(2, 5000.0)),),
    )
    with pytest.raises(AssertionError, match="reference did not converge"):
        interaction_reference(case.source, batch, coarse=SolverTolerances(0.1, 0.1))
