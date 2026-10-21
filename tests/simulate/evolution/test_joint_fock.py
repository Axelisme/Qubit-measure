"""Joint-Fock spectra through the explicit numeric request, without acquisition."""

from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.simulate import evolution
from zcu_tools.simulate.evolution import JointFockResult

RESULT_ARRAYS = (
    "populations",
    "trace_error",
    "hermitian_error",
    "min_eigenvalue",
    "edge_population",
    "sampled_pointer_means",
    "initial_trace_error",
    "initial_edge_population",
    "vacuum_trace_error",
    "vacuum_edge_population",
)


@pytest.fixture
def raw_result() -> JointFockResult:
    """Finite observations with RK4 defects that the result must not repair."""
    return JointFockResult(
        populations=np.linspace(-0.2, 1.3, 24).reshape(2, 2, 3, 2),
        cutoffs=(1, 3),
        trace_error=np.full((2, 2, 3), 0.2),
        hermitian_error=np.full((2, 2, 3), 0.3),
        min_eigenvalue=np.full((2, 2, 3), -0.4),
        edge_population=np.full((2, 2, 3), -0.5),
        sampled_pointer_means=np.full((2, 3), 0.6),
        initial_trace_error=np.full((2, 2), 0.7),
        initial_edge_population=np.full((2, 2), 1.8),
        vacuum_trace_error=np.full(2, 0.9),
        vacuum_edge_population=np.full(2, 1.1),
        state_indices=(0, 2),
    )


@pytest.mark.parametrize("name", RESULT_ARRAYS)
def test_result_owns_unrepaired_array_copies(raw_result: JointFockResult, name: str):
    shape = getattr(raw_result, name).shape
    minimum = (
        -0.2
        if name
        in (
            "populations",
            "min_eigenvalue",
            "edge_population",
            "initial_edge_population",
            "vacuum_edge_population",
        )
        else 0.1
    )
    caller_array = np.linspace(minimum, 1.3, np.prod(shape)).reshape(shape)
    expected = caller_array.copy()
    result = replace(raw_result, **{name: caller_array})
    np.testing.assert_array_equal(getattr(result, name), expected)
    caller_array[...] = 42
    np.testing.assert_array_equal(getattr(result, name), expected)


@pytest.mark.parametrize("name", RESULT_ARRAYS)
def test_result_arrays_are_read_only(raw_result: JointFockResult, name: str):
    with pytest.raises(ValueError, match="read-only"):
        getattr(raw_result, name).flat[0] = 42


@pytest.mark.parametrize(
    ("name", "shape"),
    [
        ("populations", (2, 2, 3)),
        ("populations", (2, 1, 3, 2)),
        ("populations", (0, 2, 3, 2)),
        ("populations", (2, 2, 0, 2)),
        ("populations", (2, 2, 3, 3)),
        ("trace_error", (2, 2, 1)),
        ("hermitian_error", (2, 3, 2)),
        ("min_eigenvalue", (2, 2, 3, 1)),
        ("edge_population", (1, 2, 3)),
        ("sampled_pointer_means", (2, 2)),
        ("initial_trace_error", (2,)),
        ("initial_edge_population", (2, 3)),
        ("vacuum_trace_error", (2, 1)),
        ("vacuum_edge_population", (1,)),
    ],
)
def test_result_rejects_inconsistent_shapes(
    raw_result: JointFockResult, name: str, shape: tuple[int, ...]
):
    with pytest.raises(ValueError, match=name):
        replace(raw_result, **{name: np.zeros(shape)})


@pytest.mark.parametrize("name", RESULT_ARRAYS)
@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
def test_result_rejects_nonfinite_observations(
    raw_result: JointFockResult, name: str, nonfinite: float
):
    invalid = getattr(raw_result, name).copy()
    invalid.flat[-1] = nonfinite
    with pytest.raises(ValueError, match=f"{name}.*finite"):
        replace(raw_result, **{name: invalid})


@pytest.mark.parametrize(
    "name",
    [
        "trace_error",
        "hermitian_error",
        "sampled_pointer_means",
        "initial_trace_error",
        "vacuum_trace_error",
    ],
)
def test_result_rejects_negative_defect_magnitudes_and_pointer_means(
    raw_result: JointFockResult, name: str
):
    invalid = getattr(raw_result, name).copy()
    invalid.flat[-1] = -0.1
    with pytest.raises(ValueError, match=f"{name}.*nonnegative"):
        replace(raw_result, **{name: invalid})


@pytest.mark.parametrize("cutoffs", [(1,), (1, 0), (1, -1), (1, 1.5), (1, True)])
def test_result_rejects_invalid_cutoff_inventory(raw_result: JointFockResult, cutoffs):
    with pytest.raises(ValueError, match="cutoffs"):
        replace(raw_result, cutoffs=cutoffs)


@pytest.mark.parametrize(
    "indices", [(), (0, 1), (2, 0), (0, 0), (0, 1, 2, 3), (0.0, 2.0)]
)
def test_result_rejects_inconsistent_state_indices(
    raw_result: JointFockResult, indices
):
    with pytest.raises(ValueError, match="state_indices"):
        replace(raw_result, state_indices=indices)


def test_vacuum_spectrum_keeps_finite_pi_and_probe_preparations_independent():
    assert hasattr(evolution, "JointFockRequest"), "Explicit joint request is missing"
    assert hasattr(evolution, "joint_fock_spectrum"), "Joint numeric entry is missing"
    from zcu_tools.simulate.evolution import JointFockRequest, joint_fock_spectrum

    request = JointFockRequest(
        frequencies_ghz=np.array([[0.08, 0.12], [0.081, 0.121]]),
        kappa_external_per_ns=np.array([0.01, 0.02]),
        kappa_internal_per_ns=np.array([0.002, 0.003]),
        resonator_carriers_ghz=np.array([0.09, 0.10]),
        qubit_carriers_ghz=np.array([0.02, 0.02]),
        qubit_frequency_ghz=0.02,
        incident_per_sqrt_ns=0.0,
        down_per_ns=0.0,
        up_per_ns=0.0,
        gamma2_per_ns=0.0,
        pi_duration_ns=100.0,
        pi_rabi_rad_per_ns=np.pi / 100,
        prefill_duration_ns=0.0,
        probe_duration_ns=100.0,
        probe_rabi_rad_per_ns=np.pi / 200,
    )
    result = joint_fock_spectrum(request)
    # A resonant pi/2 probe maps both the ground and finite-pi branches to Pe=1/2.
    assert result.populations.shape == (2, 2, 2, 2)  # rf, preparation, qf, g/e
    np.testing.assert_allclose(result.populations[..., 1], 0.5, atol=3e-6)
    np.testing.assert_allclose(result.populations.sum(axis=-1), 1, atol=1e-12)
    np.testing.assert_allclose(result.trace_error, 0, atol=1e-12)
    assert np.max(result.hermitian_error) < 1e-12
    assert np.min(result.min_eigenvalue) > -1e-6
    # Repeated calls and repeated probe coordinates never carry a previous state.
    np.testing.assert_array_equal(
        joint_fock_spectrum(request).populations, result.populations
    )
    assert result.cutoffs == (1, 1)  # zero sampled mean -> N=0, conservative N+1
    assert np.max(result.edge_population) == pytest.approx(0.0, abs=1e-14)
