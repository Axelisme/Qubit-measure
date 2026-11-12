"""Synthetic bath/drive models shared by Engine and kernel contract consumers."""

import numpy as np
import zcu_tools.simulate.evolution as evolution


def synthetic_model(size: int = 3) -> evolution.SecularModel:
    """Return an M=3/4 model with bidirectional decay and pure dephasing."""
    rates = np.zeros((size, size))
    rates[0, 1], rates[1, 0], rates[1, 2] = 0.015, 0.002, 0.007
    slopes = np.arange(size) * 0.03
    x = np.diag(np.ones(size - 1), 1) + np.diag(np.ones(size - 1), -1)
    y = 1j * (np.diag(np.ones(size - 1), 1) - np.diag(np.ones(size - 1), -1))
    return evolution.SecularModel(
        (0,),
        f"synthetic-{size}",
        np.arange(size, dtype=float) * 0.13,
        np.eye(size, dtype=complex),
        (
            evolution.SecularDrive(
                "drive", x.astype(np.complex128), y.astype(np.complex128)
            ),
        ),
        rates,
        (slopes[:, None] - slopes[None, :]) ** 2 / 2,
    )
