"""Complete-evolve candidate and playback selection for sampled multitone RWA."""

from dataclasses import dataclass
from functools import cache
from math import gcd

import numpy as np
from numpy.typing import NDArray

from ..kernel import KernelContext, SegmentTable
from ..workplan import FactKey
from ._rwa_numeric import admit_multitone_rows
from ._workload_facts import TableIdentity, envelope_bounds

# Revisioned with the family; the probe is an empirical check, not a channel bound.
SPOT_POLICY = ("sampled-spot-v1", 0.1)


def sampled_candidates(table: SegmentTable, context: KernelContext) -> tuple[int, ...]:
    """List complete-table sampled pairs that survive row-local omission proxies."""
    key = FactKey(
        ("multitone-sampled-candidates", TableIdentity(table), context.budget), tuple
    )

    def build() -> tuple[int, ...]:
        pairs = np.flatnonzero((table.tones == 2) & ~table.full_loss)
        rows = pairs[~table.constant_envelopes(pairs)]
        return tuple(
            int(row) for row in rows[admit_multitone_rows(table, rows, context)]
        )

    return context.work_plan.fact(key, build)


@dataclass(frozen=True)
class SampledGroup:
    """Complete-table sampled rows sharing one amplitude-only admission decision.

    rows lists all proxy-eligible members in table order. spot_rows lists the
    distinct complete-table rows to probe before admitting any of these members.
    """

    rows: tuple[int, ...]
    spot_rows: tuple[int, ...]


@dataclass(frozen=True)
class SampledAdmission:
    """Evolve-local inventory for pricing and numerical admission.

    groups lists candidates in first-member order. by_row maps complete-table
    row indices to positions in groups, with -1 for noncandidates.
    """

    groups: tuple[SampledGroup, ...]
    by_row: NDArray[np.int64]


def _coordinates(table: SegmentTable, row: int) -> tuple[object, ...]:
    # The solvers play in row-local time. A cut retains its pulse-local offset;
    # absolute start is irrelevant after the planner folds it into theta.
    return (
        tuple(table.level[row]),
        table.shift(row),
        float(table.duration_ns[row]),
        tuple(table.tone_port[row, :2]),
        tuple(table.tone_frequency_ghz[row, :2]),
        tuple(table.tone_amplitude[row, :2] < 0),
        tuple(table.tone_theta_rad[row, :2]),
        tuple(table.tone_envelope_offset_ns[row, :2]),
    )


def _shape_groups(table: SegmentTable, candidates: tuple[int, ...]) -> list[list[int]]:
    @cache
    def normalized(index: int) -> tuple[str, bytes, tuple[int, ...]]:
        env = table.envelopes[index]
        # Align the exact binary denominators of I/Q and remove one positive
        # scale. Float division can both hide real differences and create them.
        ratios = [
            float(value).as_integer_ratio() for value in env.samples.view(np.float64)
        ]
        denominator = max((d for _, d in ratios), default=1)
        integers = [n * (denominator // d) for n, d in ratios]
        divisor = gcd(*integers) or 1
        return env.kind, env.sample_ns.tobytes(), tuple(n // divisor for n in integers)

    by_coordinates: dict[tuple[object, ...], list[int]] = {}
    for row in candidates:
        by_coordinates.setdefault(_coordinates(table, row), []).append(row)
    groups = []
    for rows in by_coordinates.values():
        envelopes = table.tone_envelope[rows, :2]
        # Planning already interns exact content. Gain-only sweeps need no
        # normalization, and independent geometry cannot share probes anyway.
        if np.all(envelopes == envelopes[0]):
            groups.append(rows)
            continue
        by_shape: dict[tuple[object, ...], list[int]] = {}
        for row, indices in zip(rows, envelopes, strict=True):
            shape = tuple(normalized(int(index)) for index in indices)
            by_shape.setdefault(shape, []).append(row)
        groups.extend(by_shape.values())
    return sorted(groups, key=lambda rows: rows[0])


def sampled_groups(table: SegmentTable, context: KernelContext) -> SampledAdmission:
    """Plan proxy-eligible groups and probes for the complete fixed-flux table.

    Use the table's exact model and row-local tone geometry. Sampled envelopes
    must have identical native grids and exactly proportional I/Q samples with
    no phase rotation. For each tone, select the member with the largest
    gain magnitude times envelope peak, breaking ties in table order. Probe
    a member only once when it has both maxima. The immutable result contains
    no propagated states and lives in the open context.work_plan. Failures
    resolving models or computing proxies propagate.
    """
    key = FactKey(
        ("multitone-sampled-groups", TableIdentity(table), context.budget),
        SampledAdmission,
    )

    def build() -> SampledAdmission:
        members = _shape_groups(table, sampled_candidates(table, context))
        groups = []
        by_row = np.full(len(table.tones), -1, dtype=np.int64)
        for index, rows in enumerate(members):
            by_row[rows] = index
            groups.append(SampledGroup(tuple(rows), _spot_rows(table, rows, context)))
        return SampledAdmission(tuple(groups), by_row)

    return context.work_plan.fact(key, build)


def _spot_rows(
    table: SegmentTable, rows: list[int], context: KernelContext
) -> tuple[int, ...]:
    peaks = envelope_bounds(table, context.work_plan).peaks
    amplitude = (
        np.abs(table.tone_amplitude[rows, :2]) * peaks[table.tone_envelope[rows, :2]]
    )
    # The first row at each peak wins a tie, including repeated numeric keys.
    return tuple(dict.fromkeys(rows[index] for index in np.argmax(amplitude, axis=0)))


def probe_density(size: int) -> NDArray[np.complex128]:
    """Construct a fixed pure state with populations/coherences on low levels."""
    count = min(size, 4)
    vector = np.zeros(size, dtype=np.complex128)
    vector[:count] = np.exp(1j * np.arange(count) ** 2) / np.sqrt(count)
    return np.outer(vector, vector.conj())
