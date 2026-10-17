"""Amortized shaped preparation work, separate from numerical validity guards."""

from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from .kernel import Kernel, SegmentTable
from .kernels.gate_cache import GateCacheKernel
from .kernels.period_table import PeriodTableKernel


def _usage(
    kernel: Kernel, table: SegmentTable, rows: NDArray[np.int64]
) -> NDArray[np.int64]:
    _, inverse, counts = np.unique(
        kernel.prepare_keys(table, rows), return_inverse=True, return_counts=True
    )
    return counts[inverse]


def preference(
    table: SegmentTable, rows: NDArray[np.int64], kernels: Mapping[str, Kernel]
) -> NDArray[np.int64]:
    """Order gate, period table and direct solver by estimated amortized work.

    Scores are relative work units calibrated on 16-level FTF models, not clock
    measurements at dispatch. Preparation is charged per gate key or per port/
    frequency/amplitude-band key. Guards still decide numerical validity.
    Canonical keys also allow missing required registry entries to fail later
    with the ordinary segment-specific error. Return (len(rows), 3) permutations
    of indices 0=gate cache, 1=period table, 2=DOP853, cheapest first.
    """
    gate = kernels.get("gate_cache", GateCacheKernel())
    period = kernels.get("period_table", PeriodTableKernel())
    frequency = np.abs(table.tone_frequency_ghz[rows, 0])
    duration = table.duration_ns[rows]
    # Budget-scaled controls reduce high-carrier work. Preserve the
    # low-carrier frequency term, where lab-frame stability limits integration.
    carrier_work = 1 + 0.3 * frequency - 0.07 * np.maximum(0, frequency - 1)
    direct = 0.8 * duration * carrier_work
    gate_work = duration * (60 + 3 * frequency) / _usage(gate, table, rows) + 0.04
    safe_frequency = np.where(frequency > 0, frequency, 1)
    cycles = duration * frequency
    tail = np.maximum(0, duration - np.floor(cycles + 1e-10) / safe_frequency)
    complex_envelopes = np.array([np.any(e.samples.imag != 0) for e in table.envelopes])
    complex_rows = complex_envelopes[table.tone_envelope[rows, 0]]
    period_work = (
        (150 + 550 / safe_frequency) / _usage(period, table, rows)
        + np.where(complex_rows, 0.064, 0.008) * cycles
        + 4 * tail * (1 + 0.3 * frequency)
    )
    period_work[frequency == 0] = np.inf  # Direct DC has no table benefit.
    return np.argsort(np.column_stack((gate_work, period_work, direct)), axis=1)


def spectral_candidates(
    table: SegmentTable, rows: NDArray[np.int64], kernels: Mapping[str, Kernel]
) -> NDArray[np.int64]:
    """Keep spectral builds out of already amortized ordinary band tables.

    A flattop frequency key normally occurs in its two ramps. Shared shaped
    gates and amplitude bands keep the ordinary gate/period/direct cost order.
    Only model-independent keys enter this filter; numerical guards follow.
    """
    period = kernels.get("period_table", PeriodTableKernel())
    return rows[_usage(period, table, rows) <= 2]
