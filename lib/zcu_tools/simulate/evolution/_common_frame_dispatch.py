"""Cold common-frame choices from complete coefficient-key reuse, not cache warmth."""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ._multitone_dispatch import populate_multitone_inventory
from .kernel import KernelContext, SegmentTable
from .kernels._workload_facts import envelope_bounds
from .kernels.common_frame import (
    coefficient_keys,
    common_frame_cost_viable,
    common_frame_strengths,
)
from .kernels.multitone_rwa import MultitoneRWAKernel, multitone_retention
from .workplan import FactKey


@dataclass(frozen=True)
class CommonFramePrice:
    """One coefficient group's cold prices in relative scheduling units.

    key identifies coefficient physics. common and original are mean consumer
    prices, including the complete common build once across the group's rows.
    These estimates choose work, never numerical acceptance or cache admission.
    """

    key: bytes
    common: float
    original: float


@dataclass(frozen=True)
class CommonFramePricing:
    """Small immutable complete-acquire price inventory, not numerical tensors.

    keys is a row-aligned V32 coefficient-key vector, zero for non-two-tone rows.
    prices contains one mean-price pair per distinct coefficient key.
    """

    keys: NDArray[np.void]
    prices: tuple[CommonFramePrice, ...]


def _constant_rwa_price(table: SegmentTable, row: int, context: KernelContext) -> float:
    """Price admitted constant integration by occupied blocks and fast edges.

    Native 16-state dimension-52 unexpanded rows cost about 3.8 ms at 1 us,
    versus 12.9 ms for cold common build/apply. In the existing relative common
    units this is .12 versus .45, not the dense M**3 estimate of 2.58.
    Restored edges charge their actual interaction-picture frequencies. The
    50 MHz floor prices the constant solver's 16 ns step ceiling, not accuracy.
    """
    retained = multitone_retention(table, row, context)
    assert retained.dimension is not None
    model = table.model(context.source, row)
    frequencies = table.tone_frequency_ghz[row, :2]
    bandwidth = max(
        (
            abs(model.energies_ghz[i] - model.energies_ghz[j] + frequency)
            for frequency, edges in zip(frequencies, retained.edges, strict=True)
            for i, j in edges
        ),
        default=0.0,
    )
    bandwidth = max(0.05, float(np.ptp(frequencies)), bandwidth)
    blocks = retained.dimension / 9
    return (
        0.003 * blocks**1.5
        + 0.000014 * table.duration_ns[row] * blocks * bandwidth / 0.05
    )


def _prices(
    table: SegmentTable, context: KernelContext, *, price_rwa: bool
) -> CommonFramePricing:
    source = context.source
    admitted = (
        populate_multitone_inventory(
            table, source, context.work_plan, context.budget
        ).admitted
        if price_rwa
        else ()
    )
    rows = np.flatnonzero(table.tones == 2)
    keys = coefficient_keys(table, rows, source)
    constant = table.constant_envelopes(rows)
    grouped: dict[bytes, list[tuple[float, float, float]]] = defaultdict(list)
    for row, key, fixed in zip(rows, keys, constant, strict=True):
        model = table.model(source, row)
        size = len(model.energies_ghz) / 3
        duration = float(table.duration_ns[row])
        original = 0.001 + 0.000040 * duration
        if fixed:
            original = min(original, 0.003 + 0.000014 * duration)
        original *= size**3
        if fixed and row in admitted:
            original = min(original, _constant_rwa_price(table, int(row), context))
        grouped[key.tobytes()].append(
            (
                0.0035 * size**2,
                (0.0005 + 0.000028 * duration) * size**1.5,
                original,
            )
        )
    prices = tuple(
        CommonFramePrice(
            key,
            (values[0][0] + sum(v[1] for v in values)) / len(values),
            sum(v[2] for v in values) / len(values),
        )
        for key, values in sorted(grouped.items())
    )
    aligned = np.zeros(len(table.timeline), dtype="V32")
    aligned[rows] = keys
    return CommonFramePricing(aligned, prices)


def populate_common_frame_plan(
    table: SegmentTable, context: KernelContext, *, price_rwa: bool
) -> CommonFramePricing:
    """Publish complete-group prices lazily; prefix/worker subsets only borrow.

    Three-level calibration charges a 3.5 ms coefficient build including its
    handoff, plus .5 ms/row and 28 us/ns apply. Legacy constant prices use the
    lower direct/RWA estimate; admitted constant RWA uses retained dimension,
    restored bandwidth and duration instead of dense cubic scaling. Shaped
    rows retain dense integration prices. Small-matrix CF4
    prices scale as M**1.5 and coefficient work as M**2. These are scheduling
    estimates, not runtime bounds. No tensors live in this inventory.
    """
    return context.work_plan.fact(
        FactKey(("common-frame-prices", price_rwa), CommonFramePricing),
        lambda: _prices(table, context, price_rwa=price_rwa),
    )


def _cheap_rwa(
    table: SegmentTable,
    rows: NDArray[np.int64],
    context: KernelContext,
    rwa: MultitoneRWAKernel | None,
) -> NDArray[np.bool_]:
    """Keep admitted unexpanded graphs before paying common-frame risk work."""
    cheap = np.zeros(len(rows), dtype=bool)
    if rwa is None:
        return cheap
    if table is context.work_plan.common_frame_table:
        inventory = populate_multitone_inventory(
            table, context.source, context.work_plan, context.budget
        )
        return np.isin(rows, inventory.cheap)
    admitted = rwa.guard(table, rows, context)
    for index in np.flatnonzero(admitted):
        retained = multitone_retention(table, int(rows[index]), context)
        model = table.model(context.source, int(rows[index]))
        cheap[index] = (
            retained.dimension is not None
            and retained.dimension <= max(9, 2 * len(model.energies_ghz))
            and not any(retained.edges)
        )
    return cheap


def preferred(
    table: SegmentTable,
    rows: NDArray[np.int64],
    context: KernelContext,
    rwa: MultitoneRWAKernel | None,
) -> NDArray[np.bool_]:
    """Price complete builds, reuse and rejection risk, never accuracy.

    Prefix/worker subsets borrow complete-group mean prices. A cheap admitted
    unexpanded RWA graph wins before any common cost-feature construction.
    Remaining rows price wasted cold builds using first-order SW strength,
    weighted by actual gains and envelope peaks. The empirical eighth-power
    risk near eta=.08 is a cost forecast, not an admission/accuracy veto.
    Exact coherent .03 / bath .01 admission stays with the numerical owner.
    Cache warmth never changes these choices.
    """
    selected = ~_cheap_rwa(table, rows, context, rwa)
    if not np.any(selected):
        return selected
    complete = context.work_plan.common_frame_table
    if complete is None:
        raise RuntimeError("common-frame pricing requires complete geometry")
    pricing = populate_common_frame_plan(complete, context, price_rwa=rwa is not None)
    keys = (
        pricing.keys[rows]
        if table is complete
        else coefficient_keys(table, rows, context.source)
    )
    groups: dict[bytes, list[int]] = defaultdict(list)
    for index in np.flatnonzero(selected):
        groups[keys[index].tobytes()].append(int(index))
    peaks = envelope_bounds(table, context.work_plan).peaks
    for price in pricing.prices:
        indices = np.array(groups.get(price.key, ()), dtype=np.int64)
        if not len(indices):
            continue
        if price.common >= price.original:
            selected[indices] = False
            continue
        representative = int(rows[indices[0]])
        model = table.model(context.source, representative)
        ports = tuple(int(v) for v in table.tone_port[representative, :2])
        if not common_frame_cost_viable(
            model, ports, table.tone_frequency_ghz[representative, :2]
        ):
            selected[indices] = False
            continue
        strengths = context.work_plan.fact(
            FactKey(("common-frame-cost-strength", price.key), np.ndarray),
            lambda row=representative, model=model, ports=ports: common_frame_strengths(
                model, ports, table.tone_frequency_ghz[row, :2]
            ),
        )
        if not np.isfinite(strengths).all():
            selected[indices] = False
            continue
        current = rows[indices]
        gains = (
            np.abs(table.tone_amplitude[current, :2])
            * peaks[table.tone_envelope[current, :2]]
        )
        eta = np.max(gains @ strengths, axis=1)
        risk = np.minimum(1.0, np.minimum(1.0, eta / 0.08) ** 8)
        selected[indices] = price.common + risk * price.original < price.original
    return selected
