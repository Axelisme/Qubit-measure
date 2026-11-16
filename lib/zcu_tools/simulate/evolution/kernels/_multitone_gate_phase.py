"""Checked raw two-phase Fourier representation of full-drive gate endpoints."""

import numpy as np
from numpy.typing import NDArray

from ..kernel import SegmentTable
from ..secular import SecularModel
from ._dop853 import Tolerances
from ._gate_endpoints import build_multitone_endpoints

PHASE_GRID = (9, 10)
PHASE_RTOL = 2.75e-7
PHASE_STAGES = ((PHASE_GRID, PHASE_RTOL), ((13, 13), 1e-8))
PHASE_DEFECT = 1.9e-7
FULL_DRIVE_GATE_POLICY = (
    "dual-phase-full-drive-v2",
    PHASE_STAGES,
    0.01,  # atol/rtol
    PHASE_DEFECT,
    4,
    64,
)


def build(
    model: SecularModel, table: SegmentTable, row: int
) -> NDArray[np.complex128] | None:
    """Return checked (4,Na,Nb,M,M) coefficients, or None after two refusals.

    All envelopes start at 9x10, rtol/atol 2.75e-7/2.75e-9. Failed validation
    rebuilds once at 13x13, 1e-8/1e-10. On the three-level 4 ns sampled case
    at gain 2, the maximum quarter defect falls from 1.413e-5 to 1.534e-8.
    The larger grid resolves higher phase harmonics; tighter integration leaves
    margin below 1.9e-7 rather than spending that margin on solver error.
    This bounded route is independent of envelope shape or imaginary samples.
    Each stage integrates its nodes and two independent sets of 32 off-grid
    phase pairs together, then checks every quarter's full-unitary Frobenius
    defect <=1.9e-7. The depth-500 allocation is empirical, not a global bound
    on unobserved phases. Numerical failures propagate, never trigger upgrades.
    """
    for grid, rtol in PHASE_STAGES:
        coefficients = _build_stage(model, table, row, grid, rtol)
        if coefficients is not None:
            return coefficients
    return None


def _build_stage(
    model: SecularModel,
    table: SegmentTable,
    row: int,
    grid: tuple[int, int],
    rtol: float,
) -> NDArray[np.complex128] | None:
    """Fit and validate one full-drive grid; None means interpolation refusal."""
    axes = [np.arange(count) * (2 * np.pi / count) for count in grid]
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
    controls = Tolerances(rtol, rtol * 0.01, np.inf, np.inf, 0)
    endpoints = build_multitone_endpoints(
        model,
        table,
        row,
        np.asarray(np.concatenate([nodes, probes]), dtype=np.float64),
        controls,
    )
    size = len(model.energies_ghz)
    samples = endpoints[:, : len(nodes)].reshape(4, *grid, size, size)
    coefficients = np.fft.fftn(samples, axes=(1, 2)) / len(nodes)
    modes = [np.fft.fftfreq(count) * count for count in grid]
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
