"""Physical drive operators and complex crosstalk at the numeric composition seam."""

import numpy as np
import pytest
from zcu_tools.experiment.ftf.numeric import drive_sources, joint_parameters, resonators
from zcu_tools.resources.ftf import parse_ftf
from zcu_tools.simulate.ftf import JointModel

from tests.resources._ftf_data import ftf_document

pytestmark = pytest.mark.sim


@pytest.mark.parametrize("explicit", [False, True])
def test_charge_drive_matrices_preserve_full_charge_bitwise(explicit: bool) -> None:
    document = ftf_document()
    lines = document["drive_lines"]
    assert isinstance(lines, dict)
    if explicit:
        lines["mw"]["operator"] = "charge"
    try:
        settings = parse_ftf(document)
    except ValueError as error:
        pytest.fail(f"legacy and explicit charge lines must validate: {error}")
    joint = JointModel(
        joint_parameters(settings), np.array([0.32, 0.21, 0.43]), temperature_k=0.0
    )
    (source,) = drive_sources(settings, joint)
    # The pre-operator contract sums all three full-charge terms in this order.
    zero = np.zeros_like(joint.hamiltonian(joint.dc_flux))
    expected_x = sum(
        (c * joint.charge_operator(i) for i, c in enumerate((0.02, 0.0, -0.01))),
        zero.copy(),
    )
    expected_y = sum(
        (c * joint.charge_operator(i) for i, c in enumerate((-0.01, 0.0, -0.02))),
        zero.copy(),
    )
    np.testing.assert_array_equal(source.in_phase_ghz, expected_x)
    np.testing.assert_array_equal(source.quadrature_ghz, expected_y)
    assert source.name == "mw"


def test_mixed_lines_use_dc_flux_derivative_and_signed_complex_crosstalk() -> None:
    document = ftf_document()
    lines = document["drive_lines"]
    assert isinstance(lines, dict)
    lines["flux"] = {
        "gen_ch": 3,
        "operator": "flux",
        "flux_drive_phi0_per_gain": {
            "f1": {"real": 0.002, "imag": -0.001},
            "coupler": {"real": -0.003, "imag": 0.004},
            "f2": {"real": 0.005, "imag": 0.002},
        },
    }
    try:
        settings = parse_ftf(document)
    except ValueError as error:
        pytest.fail(f"charge and flux lines must coexist: {error}")
    for dc in (np.array([0.32, 0.21, 0.43]), np.array([0.14, 0.36, 0.48])):
        joint = JointModel(joint_parameters(settings), dc, temperature_k=0.0)
        sources = drive_sources(settings, joint)
        assert [s.name for s in sources] == ["mw", "flux"]
        coefficients = (0.002 - 0.001j, -0.003 + 0.004j, 0.005 + 0.002j)
        zero = np.zeros_like(joint.hamiltonian(dc))
        derivatives = tuple(
            (joint.hamiltonian(dc + step) - joint.hamiltonian(dc - step)) / 2e-5
            for step in 1e-5 * np.eye(3)
        )
        expected_x = sum(
            (c.real * d for c, d in zip(coefficients, derivatives, strict=True)),
            zero.copy(),
        )
        expected_y = sum(
            (-c.imag * d for c, d in zip(coefficients, derivatives, strict=True)),
            zero.copy(),
        )
        np.testing.assert_allclose(
            sources[1].in_phase_ghz, expected_x, rtol=2e-8, atol=1e-10
        )
        np.testing.assert_allclose(
            np.asarray(sources[1].quadrature_ghz), expected_y, rtol=2e-8, atol=1e-10
        )
        # Secular conversion rotates these fixed-DC operators, not a new derivative
        # at the displaced level. This covers the actual SecularDrive seam too.
        model = joint.secular_model(
            (3, -2, 4), sources, resonators=resonators(settings)
        )
        v = model.eigenvectors
        np.testing.assert_allclose(
            model.drives[1].in_phase_ghz,
            v.conj().T @ expected_x @ v,
            rtol=2e-8,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            model.drives[1].quadrature_ghz,
            v.conj().T @ expected_y @ v,
            rtol=2e-8,
            atol=1e-10,
        )
