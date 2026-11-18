"""Raw two-phase Fourier representation of full-drive gate endpoints."""

import numpy as np
from numpy.typing import NDArray

from ..kernel import SegmentTable
from ..secular import SecularModel
from ._dop853 import Tolerances
from ._gate_endpoints import GATE_SUBSEGMENTS, build_multitone_endpoints

PHASE_GRID = (5, 6)
PHASE_RTOL = 2.75e-7
FULL_DRIVE_GATE_POLICY = (
    "dual-phase-full-drive-v3",
    PHASE_GRID,
    PHASE_RTOL,
    0.01,  # atol/rtol
    GATE_SUBSEGMENTS,
)


def build(model: SecularModel, table: SegmentTable, row: int) -> NDArray[np.complex128]:
    """Integrate a positive dual-tone row and return read-only (4,5,6,M,M) coefficients.

    All keys use one grid and rtol/atol 2.75e-7/2.75e-9. The caller checks
    playback at the density level, including phases outside the grid.
    Integration failures propagate.
    """
    axes = [np.arange(count) * (2 * np.pi / count) for count in PHASE_GRID]
    nodes = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 2)
    controls = Tolerances(PHASE_RTOL, PHASE_RTOL * 0.01, np.inf, np.inf, 0)
    endpoints = build_multitone_endpoints(
        model, table, row, np.asarray(nodes, dtype=np.float64), controls
    )
    size = len(model.energies_ghz)
    samples = endpoints.reshape(GATE_SUBSEGMENTS, *PHASE_GRID, size, size)
    coefficients = np.fft.fftn(samples, axes=(1, 2)) / len(nodes)
    coefficients.setflags(write=False)
    return coefficients
