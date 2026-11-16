"""Public dressed-resonator builder contracts with synthetic native parents."""

from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose
from zcu_tools.simulate.resonator_mode import ResonatorModeInput, build_resonator_mode


def synthetic_input() -> ResonatorModeInput:
    """Four parent levels with a noncontiguous retained set and one drive port."""
    return ResonatorModeInput(
        hamiltonian_ghz=np.diag([0, 2, 7, 13]).astype(complex),
        coupling_ghz=np.zeros((4, 4), dtype=complex),
        drive_operators_ghz={"charge": np.ones((4, 4), dtype=complex)},
        retained_labels=((1, 1), (0, 0), (1, 0), (0, 1)),
        frequency_ghz=3.0,
        decay_per_ns=0.04,
        qubit_rates_per_ns=np.zeros((4, 4)),
        level=(0,),
    )


def test_uncoupled_model_orders_labels_projects_drives_and_only_loses_photons():
    result = build_resonator_mode(synthetic_input())
    assert result.index_to_label == ((0, 0), (1, 0), (0, 1), (1, 1))
    assert result.label_to_index == {(0, 0): 0, (1, 0): 1, (0, 1): 2, (1, 1): 3}
    assert_allclose(result.model.energies_ghz, [0, 2, 3, 5])
    assert_allclose(result.model.eigenvectors, np.eye(4))
    assert_allclose(result.model.dephasing_per_ns, 0)
    assert result.model.drives[0].name == "charge"
    assert_allclose(
        result.model.drives[0].in_phase_ghz,
        [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]],
    )
    assert_allclose(result.model.drives[0].quadrature_ghz, 0)
    assert_allclose(
        result.model.rates_per_ns,
        [[0, 0, 0.04, 0], [0, 0, 0, 0.04], [0, 0, 0, 0], [0, 0, 0, 0]],
    )


def test_coupled_native_parent_labels_drives_and_qubit_loss_match_two_blocks():
    # The two 2x2 parity blocks have gaps 5 GHz and 1 GHz and coupling .4 GHz.
    ce, se = np.cos(0.5 * np.arctan(0.8 / 5)), np.sin(0.5 * np.arctan(0.8 / 5))
    co, so = np.cos(0.5 * np.arctan(0.8)), np.sin(0.5 * np.arctan(0.8))
    # Rows: |00>, |01>, |10>, |11>. Columns: ascending dressed states.
    dressed = np.array(
        [[ce, 0, 0, se], [0, -so, co, 0], [0, co, so, 0], [-se, 0, 0, ce]]
    )
    native_rotation = np.eye(4, dtype=complex)
    native_rotation[:2, :2] = np.array([[1, 1j], [1j, 1]]) / np.sqrt(2)
    coupling = np.zeros((4, 4), dtype=complex)
    coupling[0, 1] = coupling[1, 0] = 0.4
    projector = np.diag([1, 0, 0, 0]).astype(complex)
    inputs = synthetic_input()
    rates = np.zeros((4, 4))
    rates[0, 1] = 0.07
    inputs = replace(
        inputs,
        photon_cutoff=2,
        hamiltonian_ghz=native_rotation
        @ inputs.hamiltonian_ghz
        @ native_rotation.conj().T,
        coupling_ghz=native_rotation @ coupling @ native_rotation.conj().T,
        drive_operators_ghz={
            "ground": native_rotation @ projector @ native_rotation.conj().T
        },
        qubit_rates_per_ns=rates,
    )
    result = build_resonator_mode(inputs)
    model = result.model
    assert result.index_to_label == ((0, 0), (1, 0), (0, 1), (1, 1))
    assert_allclose(
        model.energies_ghz,
        [
            (5 - np.hypot(5, 0.8)) / 2,
            (5 - np.hypot(1, 0.8)) / 2,
            (5 + np.hypot(1, 0.8)) / 2,
            (5 + np.hypot(5, 0.8)) / 2,
        ],
    )
    # The named bare state is the strict majority in each analytic eigenvector.
    for index, bare_row in enumerate([0, 2, 1, 3]):
        assert dressed[bare_row, index] ** 2 > 0.5
    projected_ground = dressed.T @ np.diag([1, 1, 0, 0]) @ dressed
    assert_allclose(
        np.abs(model.drives[0].in_phase_ghz), np.abs(projected_ground), atol=1e-14
    )
    resonator_only = build_resonator_mode(
        replace(inputs, qubit_rates_per_ns=np.zeros((4, 4)))
    ).model
    assert_allclose(resonator_only.rates_per_ns[np.tril_indices(4)], 0, atol=1e-28)
    expected_resonator = 0.04 * np.array(
        [
            [0, (ce * so) ** 2, (ce * co) ** 2, 0],
            [0, 0, 0, (co * ce) ** 2],
            [0, 0, 0, (so * ce) ** 2],
            [0, 0, 0, 0],
        ]
    )
    assert_allclose(resonator_only.rates_per_ns, expected_resonator, atol=1e-28)
    bare_jump = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
    expected_qubit = 0.07 * np.abs(dressed.T @ bare_jump @ dressed) ** 2
    np.fill_diagonal(expected_qubit, 0)
    assert_allclose(
        model.rates_per_ns - resonator_only.rates_per_ns, expected_qubit, atol=1e-18
    )


def test_half_mixed_retained_label_reports_label_and_overlap():
    inputs = synthetic_input()
    coupling = inputs.coupling_ghz.copy()
    coupling[0, 1] = coupling[1, 0] = 0.4
    inputs = replace(inputs, frequency_ghz=2, photon_cutoff=2, coupling_ghz=coupling)
    with pytest.raises(ValueError, match=r"label \(1, 0\).*overlap") as error:
        build_resonator_mode(inputs)
    assert float(str(error.value).split()[-1]) == pytest.approx(0.5)


def test_repeated_retained_label_reports_label_and_overlap():
    inputs = replace(synthetic_input(), retained_labels=((0, 0), (0, 0)))
    with pytest.raises(ValueError, match=r"label \(0, 0\).*overlap"):
        build_resonator_mode(inputs)


@pytest.mark.parametrize(
    "changed_field",
    [
        "parent",
        "coupling",
        "drive",
        "port",
        "labels",
        "frequency",
        "decay",
        "cutoff",
        "qubit_rates",
        "level",
    ],
)
def test_identity_covers_every_numeric_input_and_label_rule(changed_field):
    inputs = synthetic_input()
    parent = inputs.hamiltonian_ghz + np.eye(4)
    coupling = np.diag([0.02, 0.01, 0, 0]).astype(complex)
    rates = np.zeros((4, 4))
    rates[0, 1] = 0.01
    variants = {
        "parent": replace(inputs, hamiltonian_ghz=parent),
        "coupling": replace(inputs, coupling_ghz=coupling),
        "drive": replace(
            inputs, drive_operators_ghz={"charge": np.eye(4, dtype=complex)}
        ),
        "port": replace(
            inputs, drive_operators_ghz={"other": inputs.drive_operators_ghz["charge"]}
        ),
        "labels": replace(inputs, retained_labels=inputs.retained_labels[:-1]),
        "frequency": replace(inputs, frequency_ghz=3.1),
        "decay": replace(inputs, decay_per_ns=0.05),
        "cutoff": replace(inputs, photon_cutoff=4),
        "qubit_rates": replace(inputs, qubit_rates_per_ns=rates),
        "level": replace(inputs, level=(1,)),
    }
    original = build_resonator_mode(inputs).model
    assert build_resonator_mode(synthetic_input()).model.identity == original.identity
    assert (
        build_resonator_mode(variants[changed_field]).model.identity
        != original.identity
    )


def test_equal_drive_mappings_have_equal_identity_and_port_order():
    inputs = synthetic_input()
    first = {"x": np.eye(4, dtype=complex), "y": inputs.drive_operators_ghz["charge"]}
    reverse = dict(reversed(list(first.items())))
    left = build_resonator_mode(replace(inputs, drive_operators_ghz=first)).model
    right = build_resonator_mode(replace(inputs, drive_operators_ghz=reverse)).model
    assert left.identity == right.identity
    assert tuple(d.name for d in left.drives) == tuple(d.name for d in right.drives)
    for a, b in zip(left.drives, right.drives, strict=True):
        assert_allclose(a.in_phase_ghz, b.in_phase_ghz)


@pytest.mark.parametrize(
    "invalid", ["parent", "coupling", "rates_shape", "diagonal_rates", "label"]
)
def test_invalid_numeric_contract_fails_at_builder(invalid):
    inputs = synthetic_input()
    nonhermitian = inputs.hamiltonian_ghz.copy()
    nonhermitian[0, 1] = 1j
    variants = {
        "parent": replace(inputs, hamiltonian_ghz=nonhermitian),
        "coupling": replace(inputs, coupling_ghz=nonhermitian),
        "rates_shape": replace(inputs, qubit_rates_per_ns=np.zeros((3, 3))),
        "diagonal_rates": replace(inputs, qubit_rates_per_ns=np.eye(4)),
        "label": replace(inputs, retained_labels=((-1, 0),)),
    }
    with pytest.raises(ValueError, match="Hermitian|rates|label"):
        build_resonator_mode(variants[invalid])


def test_unretained_parent_level_contributes_virtual_energy_shift():
    inputs = synthetic_input()
    coupling = inputs.coupling_ghz.copy()
    coupling[0, 3] = coupling[3, 0] = 0.4
    inputs = replace(
        inputs,
        coupling_ghz=coupling,
        photon_cutoff=2,
        retained_labels=((0, 0), (0, 1)),
        drive_operators_ghz={"ground": np.diag([1, 0, 0, 0]).astype(complex)},
    )
    result = build_resonator_mode(inputs)
    # |00> mixes with discarded |31>; |01> mixes with discarded |30>.
    assert_allclose(
        result.model.energies_ghz,
        [(16 - np.hypot(16, 0.8)) / 2, (16 - np.hypot(10, 0.8)) / 2],
    )
    assert result.index_to_label == ((0, 0), (0, 1))
    assert np.all(np.diag(result.model.drives[0].in_phase_ghz).real > 0.5)


def test_two_photons_decay_twice_as_fast_as_one_photon():
    inputs = replace(synthetic_input(), retained_labels=((0, 0), (0, 1), (0, 2)))
    model = build_resonator_mode(inputs).model
    assert_allclose(model.rates_per_ns, [[0, 0.04, 0], [0, 0, 0.08], [0, 0, 0]])
