"""Checked raw two-phase Fourier representation of full-drive gate endpoints."""

import numpy as np
from numpy.typing import NDArray

from ..kernel import SegmentTable
from ..secular import SecularModel
from ._dop853 import Tolerances
from ._gate_endpoints import build_multitone_endpoints

PHASE_GRID = (9, 9)
PHASE_RTOL = 4e-7
PHASE_DEFECT = 1.9e-7
FULL_DRIVE_GATE_POLICY = (
    "dual-phase-full-drive-v1",
    PHASE_GRID,
    PHASE_RTOL,
    PHASE_DEFECT,
    4,
    64,
)


def build(
    model: SecularModel, table: SegmentTable, row: int
) -> NDArray[np.complex128] | None:
    """Return checked (4,9,9,M,M) Fourier coefficients, else reject with None.

    Integrate a 9x9 raw phase grid and two sets of 32 independent off-grid
    phase pairs in the same batch, at rtol 4e-7 and atol 4e-9. Require every
    quarter's full-unitary Frobenius interpolation defect <=1.9e-7 before
    playback. This depth-500 error allocation is empirical, not a global bound
    on unobserved phases. No upgrade is attempted after failed validation.
    Numerical integration failures propagate rather than masquerade as refusal.
    """
    axes = [np.arange(count) * (2 * np.pi / count) for count in PHASE_GRID]
    nodes = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 2)
    # Irrational sequences avoid the integration nodes and rational-grid aliases.
    indices = np.arange(1, 33)[:, None]
    probes = (
        2
        * np.pi
        * np.remainder(
            np.concatenate(
                [
                    indices * np.sqrt([2.0, 3.0]),
                    indices * np.sqrt([5.0, 7.0]) + np.array([0.137, 0.739]),
                ]
            ),
            1,
        )
    )
    controls = Tolerances(PHASE_RTOL, PHASE_RTOL * 0.01, np.inf, np.inf, 0)
    endpoints = build_multitone_endpoints(
        model,
        table,
        row,
        np.asarray(np.concatenate([nodes, probes]), dtype=np.float64),
        controls,
    )
    size = len(model.energies_ghz)
    samples = endpoints[:, : len(nodes)].reshape(4, *PHASE_GRID, size, size)
    coefficients = np.fft.fftn(samples, axes=(1, 2)) / len(nodes)
    modes = [np.fft.fftfreq(count) * count for count in PHASE_GRID]
    predicted = np.einsum(
        "pa,pb,sabij->spij",
        np.exp(1j * probes[:, :1] * modes[0]),
        np.exp(1j * probes[:, 1:] * modes[1]),
        coefficients,
    )
    defect = float(
        np.max(np.linalg.norm(predicted - endpoints[:, len(nodes) :], axis=(-2, -1)))
    )
    if not np.isfinite(defect) or defect > PHASE_DEFECT:
        return None
    coefficients.setflags(write=False)
    return coefficients
