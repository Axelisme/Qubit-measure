"""Complete-evolve physical carrier and rise/fall spectral inventories."""

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .kernel import KernelContext, SegmentTable
from .secular import ModelSource
from .workplan import CarrierFamilyPlan, FactKey, RampFamilyPlan, WorkPlan

type _RampCoordinate = tuple[str, tuple[int, ...], int, float, float, int]

# At +0.3 GHz a zero crossing with m=2 had frozen-envelope error 0.01243 > 0.01.
_COMPLEX_MOTION_LIMIT = 0.0025


@dataclass(frozen=True)
class _RampRows:
    rows: NDArray[np.int64]
    family_indices: NDArray[np.int64]
    families: tuple[RampFamilyPlan, ...]


def _row_index(
    families: tuple[RampFamilyPlan, ...], pairs: list[tuple[int, int]]
) -> _RampRows:
    pairs.sort()
    return _RampRows(
        np.array([row for row, _ in pairs], dtype=np.int64),
        np.array([index for _, index in pairs], dtype=np.int64),
        families,
    )


def populate_carrier_plan(
    table: SegmentTable, source: ModelSource, plan: WorkPlan
) -> None:
    """Publish carrier/ramp inventories and row lookup for the full evolve table.

    Called once before workers start. Families retain all candidate rows, even
    those later rejected. The immutable row index is scoped to table identity;
    renumbered prefix tables cannot use its global rows.
    """
    ramps = plan_ramp_families(table, source, plan)
    plan.publish_ramp_families(ramps)
    plan.publish_carrier_families(plan_carrier_families(ramps))
    if ramps:
        plan.fact(
            FactKey(("shaped-ramp-rows", id(table)), _RampRows),
            lambda: _row_index(
                ramps,
                [(row, index) for index, ramp in enumerate(ramps) for row in ramp.rows],
            ),
        )


def subdivisions(
    table: SegmentTable,
    rows: NDArray[np.int64],
    source: ModelSource,
    work_plan: WorkPlan,
) -> NDArray[np.int64]:
    """Return the existing motion/split subperiod count for each selected row."""
    # Kernel registration imports this planner; defer numerical helper imports.
    from .kernels._shaped import drive_bounds
    from .kernels._workload_facts import envelope_bounds

    strength, gamma = drive_bounds(table, rows, source, work_plan)
    bounds = envelope_bounds(table, work_plan)
    peaks, slopes = bounds.peaks, bounds.slopes
    envelope = table.tone_envelope[rows, 0]
    gain = np.abs(table.tone_amplitude[rows, 0])
    frequency = np.maximum(np.abs(table.tone_frequency_ghz[rows, 0]), 1e-20)
    motion = 2 * np.pi * gain * strength * slopes[envelope] / frequency**2
    split = (
        table.duration_ns[rows]
        * gamma
        * (gain * peaks[envelope] * strength / frequency) ** 2
    )
    motion_limit = np.where(bounds.real[envelope], 0.01, _COMPLEX_MOTION_LIMIT)
    counts = np.maximum(
        1, np.ceil(np.sqrt(np.maximum(motion / motion_limit, split / 0.01)))
    )
    # Unrepresentable counts saturate above the guard's 32-subperiod limit.
    result = np.full(len(rows), np.iinfo(np.int64).max, dtype=np.int64)
    representable = counts < 2.0**63
    result[representable] = counts[representable].astype(np.int64)
    return result


def _coordinates(
    table: SegmentTable,
    rows: NDArray[np.int64],
    source: ModelSource,
    work_plan: WorkPlan,
) -> Iterator[_RampCoordinate]:
    from .kernels._workload_facts import envelope_bounds

    peaks = envelope_bounds(table, work_plan).peaks
    bounds = np.abs(table.tone_amplitude[rows, 0]) * peaks[table.tone_envelope[rows, 0]]
    counts = subdivisions(table, rows, source, work_plan)
    for row, bound, count in zip(rows, bounds, counts, strict=True):
        model = source.model(tuple(int(v) for v in table.level[row]))
        # Match band_keys' zero-gain DC identity, including ineligible candidates.
        frequency = float(table.tone_frequency_ghz[row, 0]) if bound > 0 else 0.0
        yield (
            model.identity,
            model.level,
            int(table.tone_port[row, 0]),
            frequency,
            float(bound),
            int(count),
        )


def plan_ramp_families(
    table: SegmentTable, source: ModelSource, work_plan: WorkPlan
) -> tuple[RampFamilyPlan, ...]:
    """Group all single-tone shaped candidates, before numerical admission.

    Duration, slope and prepare key do not partition a spectral family. The
    returned rows always index this complete table, not a worker/prefix slice.
    No harmonic, amplitude-node or convergence decision is made here.
    """
    rows = np.flatnonzero(table.tones == 1)
    rows = rows[~table.constant_envelopes(rows)]
    if not len(rows):
        return ()
    groups: dict[tuple[str, tuple[int, ...], int, float, float, int], list[int]] = {}
    for row, identity in zip(
        rows, _coordinates(table, rows, source, work_plan), strict=True
    ):
        groups.setdefault(identity, []).append(int(row))
    return tuple(
        RampFamilyPlan(*identity, tuple(members))
        for identity, members in groups.items()
    )


def plan_carrier_families(
    ramps: tuple[RampFamilyPlan, ...],
) -> tuple[CarrierFamilyPlan, ...]:
    """Select profitable physical groups from the complete ramp inventory.

    Only >9 unique nonzero same-sign carriers with max(abs(f))/min(abs(f))
    <= 1.1 qualify. Wider, mixed-sign and small groups use exact builds. This
    policy is independent of guards, workers, cache warmth and build order.
    """
    groups: dict[
        tuple[str, tuple[int, ...], int, float, int], list[RampFamilyPlan]
    ] = {}
    for ramp in ramps:
        identity = (
            ramp.model_identity,
            ramp.level,
            ramp.port,
            ramp.bound,
            ramp.subdivisions,
        )
        groups.setdefault(identity, []).append(ramp)
    families = []
    for identity, members in groups.items():
        grid = np.unique([ramp.frequency_ghz for ramp in members])
        magnitudes = np.abs(grid)
        if (
            len(grid) > 9
            and np.all(grid * grid[0] > 0)
            and magnitudes.max() <= 1.1 * magnitudes.min()
        ):
            families.append(
                CarrierFamilyPlan(
                    *identity,
                    tuple(float(f) for f in grid),
                    tuple(sorted(row for ramp in members for row in ramp.rows)),
                )
            )
    return tuple(families)


def ramp_families_for_rows(
    table: SegmentTable, rows: NDArray[np.int64], context: KernelContext
) -> tuple[RampFamilyPlan, ...]:
    """Resolve selected rows to planned families, including renumbered prefixes.

    Physical coordinates, not global row numbers, identify prefix consumers.
    Standalone kernel calls with an empty WorkPlan plan their own complete table.
    Returned records align with rows; a missing planned coordinate raises KeyError.
    """
    index = context.work_plan.fact(
        FactKey(("shaped-ramp-rows", id(table)), _RampRows),
        lambda: _reindex_ramps(table, context),
    )
    positions = np.searchsorted(index.rows, rows)
    if np.any(positions >= len(index.rows)) or np.any(index.rows[positions] != rows):
        raise KeyError("row has no planned ramp family")
    return tuple(index.families[index.family_indices[p]] for p in positions)


def _reindex_ramps(table: SegmentTable, context: KernelContext) -> _RampRows:
    families = context.work_plan.ramp_families
    if not families:
        families = plan_ramp_families(table, context.source, context.work_plan)
        return _row_index(
            families,
            [(row, index) for index, ramp in enumerate(families) for row in ramp.rows],
        )
    coordinates = {
        (
            family.model_identity,
            family.level,
            family.port,
            family.frequency_ghz,
            family.bound,
            family.subdivisions,
        ): index
        for index, family in enumerate(families)
    }
    rows = np.flatnonzero(table.tones == 1)
    rows = rows[~table.constant_envelopes(rows)]
    # Prefix tables are renumbered. Resolve their physical coordinates once,
    # then every build uses the same cheap table-local row lookup as the main scan.
    pairs = [
        (int(row), coordinates[key])
        for row, key in zip(
            rows,
            _coordinates(table, rows, context.source, context.work_plan),
            strict=True,
        )
    ]
    return _row_index(families, pairs)
