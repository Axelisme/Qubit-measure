"""Offline batch construction and persistent evolution through public seams."""

import numpy as np
from scipy.linalg import expm
from zcu_tools.simulate.evolution import Basis, DriveSource, NumericModel


def test_intel_joint_complex128_batch_matches_independent_exponential(tmp_path):
    from zcu_tools.simulate.evolution import CPUReference, IntelOpenCL, PlateauFamily

    x = np.array([[0, 1], [1, 0]], dtype=complex)
    y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    basis = Basis("batch", ("g", "e"), np.eye(2, dtype=complex))
    model = NumericModel(
        "batch",
        basis,
        np.diag([0, 0.017]).astype(complex),
        (
            DriveSource(
                "a",
                np.asarray(0.012 * x, dtype=np.complex128),
                np.asarray(0.012 * y, dtype=np.complex128),
            ),
            DriveSource(
                "b",
                np.asarray(0.009 * y, dtype=np.complex128),
                np.asarray(0.009 * x, dtype=np.complex128),
            ),
        ),
    )
    family = PlateauFamily(model, 8.0, (1 + 0.3j, -0.4 + 0.2j))
    scales = np.array([-0.75, -0.13, 0.0, 0.37, 0.8])
    expected = np.array(
        [
            expm(
                -2j
                * np.pi
                * 8
                * (
                    model.hamiltonian_ghz
                    + s * (0.012 * x + 0.0036 * y - 0.0036 * y + 0.0018 * x)
                )
            )
            for s in scales
        ]
    )
    cpu = CPUReference().compute(family, scales)
    gpu = IntelOpenCL(tmp_path / "compiler").compute(family, scales)
    np.testing.assert_allclose(cpu.propagators, expected, atol=2e-12, rtol=0)
    np.testing.assert_allclose(gpu.propagators, expected, atol=2e-12, rtol=0)
    assert gpu.device.startswith("Intel")
    assert gpu.gpu_bytes > 0
    assert gpu.transfer_seconds >= 0
