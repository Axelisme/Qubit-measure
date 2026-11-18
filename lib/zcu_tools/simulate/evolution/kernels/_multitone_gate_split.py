"""Full Lindblad density probes for the full-drive dual-phase gate guard."""

import numpy as np
from numpy.typing import NDArray

from ..kernel import SegmentTable
from ..secular import SecularModel
from ._dop853_data import pack_models, pack_segments
from .dop853 import DOP853Plan, EndpointData, advance_endpoints

# One density trajectory per phase pair. Include (0, 0) and off-grid pairs;
# sample all four quarter turns on each axis without taking the full product.
SPLIT_PHASES = (np.pi / 2) * np.array([[0, 0], [1, 2], [2, 3], [3, 1]])
SPLIT_PHASES.setflags(write=False)
SPLIT_REFERENCE_BUDGET = 1e-8  # DOP853 local rtol/atol 3e-10/3e-12


def probe_densities(size: int) -> NDArray[np.complex128]:
    """Probe low/high populations and real/complex coherences, without caller state."""
    kets = np.empty((len(SPLIT_PHASES), size), dtype=np.complex128)
    kets[:2] = 0
    kets[0, 0] = kets[1, -1] = 1
    kets[2] = 1 / np.sqrt(size)
    kets[3] = (1j ** np.arange(size)) / np.sqrt(size)
    return kets[:, :, None] * kets[:, None, :].conj()


def advance_probes(
    model: SecularModel,
    table: SegmentTable,
    row: int,
    initial: NDArray[np.complex128],
    budget: float,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Return tight-reference and Engine-budget DOP853 probe endpoints.

    Compact the representative into independent timelines at local zero.
    Both solves start with the same probe densities and envelope clocks.
    Use the ordinary isolated-row interaction frame and DOP853 budget policy.
    The input densities are not modified. Integration errors propagate.
    """
    rows = np.arange(len(SPLIT_PHASES), dtype=np.int64)
    same_row = np.full_like(rows, row)
    probes = SegmentTable(
        timeline=rows,
        start_ns=np.zeros(len(rows)),
        duration_ns=table.duration_ns[same_row],
        level=table.level[same_row],
        tones=table.tones[same_row],
        tone_port=table.tone_port[same_row, :2],
        tone_frequency_ghz=table.tone_frequency_ghz[same_row, :2],
        tone_amplitude=table.tone_amplitude[same_row, :2],
        tone_theta_rad=SPLIT_PHASES,
        tone_envelope=np.tile([0, 1], (len(rows), 1)),
        tone_envelope_offset_ns=table.tone_envelope_offset_ns[same_row, :2],
        envelopes=tuple(table.envelopes[e] for e in table.tone_envelope[row, :2]),
    )
    data = EndpointData(pack_models([model]), pack_segments(probes))
    row_models = (rows, np.zeros_like(rows))
    plan = DOP853Plan(np.full(len(rows), "interaction"), np.full_like(rows, -1))
    reference, fallback = initial.copy(), initial.copy()
    advance_endpoints(data, row_models, reference, SPLIT_REFERENCE_BUDGET, plan)
    advance_endpoints(data, row_models, fallback, budget, plan)
    return reference, fallback
