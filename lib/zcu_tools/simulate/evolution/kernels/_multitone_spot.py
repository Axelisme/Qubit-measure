"""Complete-evolve candidate and playback selection for sampled multitone RWA."""

import numpy as np
from numpy.typing import NDArray

from ..kernel import KernelContext, SegmentTable
from ..workplan import FactKey
from ._rwa_numeric import admit_multitone_rows
from ._workload_facts import TableIdentity, envelope_bounds, quadrature_norms

# Revisioned with the family; the probe is an empirical check, not a channel bound.
SPOT_POLICY = ("sampled-spot-v1", 0.1)


def sampled_candidates(table: SegmentTable, context: KernelContext) -> tuple[int, ...]:
    """List complete-table sampled pairs that survive row-local omission proxies."""
    key = FactKey(
        ("multitone-sampled-candidates", TableIdentity(table), context.budget), tuple
    )

    def build() -> tuple[int, ...]:
        pairs = np.flatnonzero(table.tones == 2)
        rows = pairs[~table.constant_envelopes(pairs)]
        return tuple(
            int(row) for row in rows[admit_multitone_rows(table, rows, context)]
        )

    return context.work_plan.fact(key, build)


def _spot_key(
    table: SegmentTable, row: int, context: KernelContext
) -> tuple[object, ...]:
    model = table.model(context.source, row)
    # Both solvers play in row-local time. Absolute start is irrelevant, but
    # effective start phase must match; full-drive error is not phase covariant.
    return (
        model.identity,
        model.level,
        table.shift(row),
        float(table.duration_ns[row]),
        tuple(table.tone_port[row, :2]),
        tuple(table.tone_frequency_ghz[row, :2]),
        tuple(table.tone_amplitude[row, :2]),
        tuple(table.tone_theta_rad[row, :2]),
        tuple(table.tone_envelope_offset_ns[row, :2]),
        tuple(table.envelopes[e].identity for e in table.tone_envelope[row, :2]),
    )


def _curvature(samples: NDArray[np.complex128], times: NDArray[np.float64]) -> float:
    spacing = np.diff(times)
    if len(spacing) < 2:
        return 0.0
    slopes = np.diff(samples) / spacing
    return float(np.max(np.abs(np.diff(slopes) / ((spacing[1:] + spacing[:-1]) / 2))))


def spot_rows(
    table: SegmentTable, candidates: tuple[int, ...], context: KernelContext
) -> tuple[int, ...]:
    """Return the strongest, steepest and most curved candidates, once per key.

    Use peak-weighted operator norm for drive strength, peak-normalized linear
    slope and change in slope for envelope shape. Shape metrics compare each
    tone in its native sample grid. Strength breaks shape-score ties so a
    gain sweep with one shape need not probe a second row. Remaining ties
    follow complete-table order.
    """
    key = FactKey(("multitone-sampled-spots", TableIdentity(table), candidates), tuple)
    return context.work_plan.fact(key, lambda: _spot_rows(table, candidates, context))


def _spot_rows(
    table: SegmentTable, candidates: tuple[int, ...], context: KernelContext
) -> tuple[int, ...]:
    if not candidates:
        return ()
    bounds = envelope_bounds(table, context.work_plan)
    slopes = np.zeros(len(table.envelopes))
    curvatures = np.zeros_like(slopes)
    for index, env in enumerate(table.envelopes):
        peak = bounds.peaks[index]
        if env.kind == "sampled" and peak > 0:
            slopes[index] = bounds.slopes[index] / peak
            curvatures[index] = _curvature(env.samples, env.sample_ns) / peak
    scores = np.empty((len(candidates), 3))
    for index, row in enumerate(candidates):
        envs = table.tone_envelope[row, :2]
        model = table.model(context.source, row)
        strength = 0.0
        for tone, env in enumerate(envs):
            port = int(table.tone_port[row, tone])
            norms = quadrature_norms(model, port, context.work_plan)
            strength += (
                abs(table.tone_amplitude[row, tone])
                * bounds.peaks[env]
                * (norms.in_phase + norms.quadrature)
            )
        scores[index] = strength, np.max(slopes[envs]), np.max(curvatures[envs])
    selected: dict[tuple[object, ...], int] = {}
    for score in scores.T:
        ties = np.flatnonzero(score == np.max(score))
        row = candidates[ties[np.argmax(scores[ties, 0])]]
        selected.setdefault(_spot_key(table, row, context), row)
    return tuple(selected.values())


def probe_density(size: int) -> NDArray[np.complex128]:
    """Construct a fixed pure state with populations/coherences on low levels."""
    count = min(size, 4)
    vector = np.zeros(size, dtype=np.complex128)
    vector[:count] = np.exp(1j * np.arange(count) ** 2) / np.sqrt(count)
    return np.outer(vector, vector.conj())
