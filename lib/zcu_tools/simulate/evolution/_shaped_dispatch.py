"""Amortized shaped preparation work, separate from numerical validity guards."""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ._carrier_plan import plan_ramp_families
from .kernel import Kernel, KernelContext, SegmentTable
from .kernels._rwa_gate import canonical_keys, gate_segments
from .kernels._workload_facts import envelope_bounds, fixed_flux
from .kernels.gate_cache import GateCacheKernel
from .kernels.multitone_rwa import retained_edges
from .kernels.period_table import PeriodTableKernel
from .kernels.single_tone_gate import SingleToneGateKernel
from .secular import ModelSource
from .workplan import FactKey, WorkPlan


def _usage(
    kernel: Kernel, table: SegmentTable, rows: NDArray[np.int64], context: KernelContext
) -> NDArray[np.int64]:
    _, inverse, counts = np.unique(
        kernel.prepare_keys(table, rows, context),
        return_inverse=True,
        return_counts=True,
    )
    return counts[inverse]


def preference(
    table: SegmentTable,
    rows: NDArray[np.int64],
    kernels: Mapping[str, Kernel],
    context: KernelContext,
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
    gate_work = (
        duration * (60 + 3 * frequency) / _usage(gate, table, rows, context) + 0.04
    )
    safe_frequency = np.where(frequency > 0, frequency, 1)
    cycles = duration * frequency
    tail = np.maximum(0, duration - np.floor(cycles + 1e-10) / safe_frequency)
    complex_envelopes = ~envelope_bounds(table, context.work_plan).real
    complex_rows = complex_envelopes[table.tone_envelope[rows, 0]]
    period_work = (
        (150 + 550 / safe_frequency) / _usage(period, table, rows, context)
        + np.where(complex_rows, 0.064, 0.008) * cycles
        + 4 * tail * (1 + 0.3 * frequency)
    )
    period_work[frequency == 0] = np.inf  # Direct DC has no table benefit.
    return np.argsort(np.column_stack((gate_work, period_work, direct)), axis=1)


@dataclass(frozen=True)
class SingleGateChoice:
    """Lowest estimated cold plan for one physical Sambe spectral family.

    keys: canonical keys selected for covariant gates; other rows retain Sambe.
    uses: complete-acquire row count for this family.
    gate_seconds: total mixed-plan build, playback and guard/key CPU estimate.
    sambe_seconds: all-Sambe cold CPU estimate for the same complete inventory.
    preferred: at least one canonical key uses a gate in the cheaper plan.
    """

    keys: tuple[bytes, ...]
    uses: int
    gate_seconds: float
    sambe_seconds: float
    preferred: bool


def _cold_key_cost(
    table: SegmentTable, row: int, count: int, context: KernelContext, subdivisions: int
) -> tuple[float, float]:
    """Estimate cold gate build/play and Sambe variable playback for one key."""
    model = context.source.model(tuple(int(v) for v in table.level[row]))
    scale = (len(model.energies_ghz) / 16) ** 3
    duration = float(table.duration_ns[row])
    frequency = abs(float(table.tone_frequency_ghz[row, 0]))
    env_index = table.tone_envelope[row, 0]
    env = table.envelopes[env_index]
    peak = envelope_bounds(table, context.work_plan).peaks[env_index]
    motion = 0.0
    if len(env.samples) > 1 and peak > 0:
        motion = float(
            np.max(np.abs(np.diff(env.samples) / np.diff(env.sample_ns))) / peak
        )
    retained = retained_edges(
        model, int(table.tone_port[row, 0]), float(table.tone_frequency_ghz[row, 0])
    )
    strength = (
        abs(table.tone_amplitude[row, 0]) * peak * 2 * np.linalg.norm(retained, 2)
    )
    kernel = SingleToneGateKernel()
    admitted = kernel.guard(table, np.array([row], dtype=np.int64), context)[0]
    gate = np.inf
    if admitted:
        segments = gate_segments(table, row, context)
        build = 0.10 * duration / 256 * (
            1 + 4 * motion + 0.1 * 2 * np.pi * duration * strength
        ) + 0.001 * (segments - 4)
        gate = scale * (build + count * (0.0000075 * segments + 0.00008))
    steps = duration * frequency * subdivisions
    whole_steps = np.floor(steps + 1e-10)
    tail = (
        duration - whole_steps / (frequency * subdivisions) if frequency else duration
    )
    # Smooth real Sambe uses four amplitude coefficients, seven Fourier modes
    # and two Newton polar iterations. Phase contraction is paid per row and
    # subdivision; Clenshaw, polar and D-U-D are paid per subperiod.
    phase = 0.0000002 * 4 * 7 * subdivisions
    subperiod = 0.0000005 * 4 + 0.000004 * 2 + 0.00001
    fractional = 0.0009 * tail if frequency >= 1 else (0.00010 if tail > 1e-10 else 0)
    sambe = scale * count * (phase + whole_steps * subperiod + fractional + 0.00002)
    return float(gate), float(sambe)


def populate_single_gate_plan(
    table: SegmentTable, source: ModelSource, plan: WorkPlan, budget: float = 1e-3
) -> tuple[SingleGateChoice, ...]:
    """Choose mixed gate/Sambe cold plans from complete-acquire spectral families.

    Each canonical gate key pays its own numerically admitted build plus n-step
    playback. The alternative pays actual Sambe subdivisions, four-coefficient/
    seven-mode phase contraction, two polar iterations, D-U-D and fractional
    tails. Peak gain and subdivision count partition shared Sambe construction,
    not an entire carrier/shape domain. Within each family, compare all Sambe,
    individually profitable gates plus residual Sambe, and all gates (which can
    avoid the shared spectral build). Unsupported keys never pay a gate build.

    Calibration at 16 levels: four-key compiled Gaussian prepares measured
    0.241/0.316 CPU-s (integer/half); retain the conservative 0.10 CPU-s per
    256 ns key before shape/drive multipliers. Each extra restart costs 0.001 s.
    Four-U playback measured 0.0592/1445 = 41 us/row; allow 30 us plus 80 us for
    guard/key work. Sambe's three-subdivision, four-coefficient, seven-mode,
    two-polar half-flux workload measured 4.6533 s/1445 rows. Charge 20 us per
    subperiod, 0.2 us per phase coefficient/mode and 0.008 s per spectral build.
    Fractional tails cost 0.9 ms/ns above 1 GHz or 0.1 ms for a spectral tail.
    Cubic dimension scaling applies to both alternatives. These are conservative
    relative CPU estimates, not numerical certificates or resident-cache prices.
    Prefix/worker subsets borrow this decision; no future acquire is predicted.
    """

    def build() -> tuple[SingleGateChoice, ...]:
        context = KernelContext(source, budget, plan)
        families = plan.ramp_families or plan_ramp_families(table, source, plan)
        if not families:
            return ()
        canonical = canonical_keys(table, context)
        choices = []
        for family in families:
            group: dict[tuple[bytes, bool], tuple[int, int]] = {}
            for row in family.rows:
                key = canonical[row]
                identity = (key, fixed_flux(table, row))
                representative, count = group.get(identity, (row, 0))
                group[identity] = (representative, count + 1)
            scale = (len(source.model(family.level).energies_ghz) / 16) ** 3
            shared = scale * 0.008
            costs = [
                (key, *_cold_key_cost(table, row, count, context, family.subdivisions))
                for (key, _fixed), (row, count) in group.items()
            ]
            all_sambe = shared + sum(sambe for _, _, sambe in costs)
            selected = {key for key, gate, sambe in costs if gate < sambe}
            mixed = sum(min(gate, sambe) for _, gate, sambe in costs)
            if any(gate >= sambe for _, gate, sambe in costs):
                mixed += shared
            all_gate = sum(gate for _, gate, _ in costs)
            if all_gate < mixed:
                mixed = all_gate
                selected = {key for key, _, _ in costs}
            if mixed >= all_sambe:
                mixed = all_sambe
                selected = set()
            choices.append(
                SingleGateChoice(
                    tuple(sorted(selected)),
                    len(family.rows),
                    float(mixed),
                    float(all_sambe),
                    bool(selected),
                )
            )
        return tuple(choices)

    return plan.fact(FactKey(("single-tone-gate-mixed-cold-v2", budget), tuple), build)


def single_gate_preferred(
    table: SegmentTable, rows: NDArray[np.int64], context: KernelContext
) -> NDArray[np.bool_]:
    """Borrow complete-acquire mixed cold choices before row-local guards."""
    preferred = {
        key
        for choice in populate_single_gate_plan(
            table, context.source, context.work_plan, context.budget
        )
        for key in choice.keys
    }
    if not preferred:
        return np.zeros(len(rows), dtype=bool)
    canonical = canonical_keys(table, context)
    return np.array([canonical[row] in preferred for row in rows], dtype=bool)
