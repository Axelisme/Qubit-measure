"""Numerical publication gates at the public table-validation seam."""

import numpy as np
import pytest
from zcu_tools.simulate.evolution import (
    Basis,
    CPUReference,
    DriveSource,
    NumericModel,
    PlateauFamily,
    ValidationLimits,
)
from zcu_tools.simulate.evolution.validation import interpolate, validate_table


def test_trace_growth_is_rejected_even_within_elementwise_unitarity_limit():
    n = 8
    basis = Basis("trace", tuple(map(str, range(n))), np.eye(n, dtype=complex))
    coherent = np.ones((n, n), dtype=complex)
    model = NumericModel(
        "trace",
        basis,
        np.zeros((n, n), dtype=complex),
        (DriveSource("drive", 0.01 * coherent / n),),
    )
    family = PlateauFamily(model, 2.0, (1 + 0j,))
    nodes = 0.1 * np.cos(np.linspace(np.pi, 0, 7))
    values = CPUReference().compute(family, nodes).propagators @ (
        np.eye(n) + 1.4e-12 * coherent
    )
    u = interpolate(nodes, values, -0.1 + 0.2 * 0.431)
    long_u = np.linalg.matrix_power(u, 512)
    # A coherent rank-one gain can hide in small individual matrix elements.
    unitarity_error = np.max(np.abs(long_u.conj().T @ long_u - np.eye(n)))
    initial = np.full(n, np.exp(0.3j) / np.sqrt(n))
    evolved = long_u @ initial
    density = np.outer(evolved, evolved.conj())
    assert unitarity_error <= 2e-9
    assert abs(np.trace(density) - 1) > 2e-9

    with pytest.raises(ValueError, match="trace"):
        validate_table(family, nodes, values, ValidationLimits())
