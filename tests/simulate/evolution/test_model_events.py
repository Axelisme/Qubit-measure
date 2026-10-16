"""Numeric model discontinuities partition evolution independent of pulse events."""

import numpy as np
from zcu_tools.simulate.evolution import (
    Basis,
    DensityState,
    Evolution,
    NumericModel,
    SolverSettings,
)


def test_model_event_times_preserve_short_dynamic_interval() -> None:
    basis = Basis("two", ("a", "b"), np.eye(2, dtype=complex))
    initial = DensityState(basis, np.diag([1.0, 0.0]).astype(complex))
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)

    def extra(t: float):
        return np.asarray(
            x / 8 if 400.0 <= t < 402.0 else np.zeros_like(x), dtype=np.complex128
        )

    model = NumericModel(
        "short",
        basis,
        np.zeros_like(x),
        hamiltonian_at=extra,
        event_times_ns=(400.0, 402.0),
    )
    result = Evolution(model, SolverSettings(max_step_ns=10000.0)).evolve(
        initial, end_ns=1000.0
    )
    np.testing.assert_allclose(result.density, np.diag([0.0, 1.0]), atol=1e-9)
