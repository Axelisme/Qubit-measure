"""Amortized shaped preparation work, separate from numerical validity guards."""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .kernel import Kernel, KernelContext, SegmentTable
from .kernels._rwa_gate import canonical_key
from .kernels._workload_facts import envelope_bounds
from .kernels.gate_cache import GateCacheKernel
from .kernels.multitone_rwa import retained_edges
from .kernels.period_table import PeriodTableKernel
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


_SINGLE_GATE_PLAN = FactKey(("single-tone-gate-cold-cost-v1",), tuple)


@dataclass(frozen=True)
class SingleGateChoice:
    """Cold costs in relative CPU seconds for one physical carrier/shape domain.

    keys: canonical gate keys in this domain from the complete acquire.
    uses: total rows, including all phases and absolute starts.
    gate_seconds/sambe_seconds: summed builds, playback and guard/key work.
    preferred: gate estimate has at least 20 percent cold cost headroom.
    """

    keys: tuple[bytes, ...]
    uses: int
    gate_seconds: float
    sambe_seconds: float
    preferred: bool


def populate_single_gate_plan(
    table: SegmentTable, source: ModelSource, plan: WorkPlan
) -> tuple[SingleGateChoice, ...]:
    """Fix single-tone gate vs Sambe costs from the entire acquire inventory.

    Estimates use joint dimension, actual duration, gain*retained-drive strength,
    sampled shape motion, carrier periods and fractional-tail type. Each gate
    key pays a cold four-U build; a Sambe carrier/shape domain pays one spectral
    build. No numerical admission, experiment identity or cache warmth is read.
    Prefix and worker subsets borrow these decisions without recounting reuse.
    """

    def build() -> tuple[SingleGateChoice, ...]:
        rows = np.flatnonzero(
            (table.tones == 1)
            & ~table.constant_envelopes(np.arange(len(table.timeline)))
        )
        if not len(rows):
            return ()
        domains: dict[
            tuple[str, tuple[int, ...], int, float, str],
            dict[bytes, tuple[int, int]],
        ] = {}
        for row in rows:
            model = source.model(tuple(int(v) for v in table.level[row]))
            env = table.envelopes[table.tone_envelope[row, 0]]
            domain = (
                model.identity,
                model.level,
                int(table.tone_port[row, 0]),
                float(table.tone_frequency_ghz[row, 0]),
                env.identity,
            )
            key = canonical_key(table, int(row), source)
            group = domains.setdefault(domain, {})
            representative, count = group.get(key, (int(row), 0))
            group[key] = (representative, count + 1)
        choices = []
        bounds = envelope_bounds(table, plan)
        for group in domains.values():
            gate, sambe, uses = 0.0, 0.0, 0
            scale = 1.0
            for row, count in group.values():
                model = source.model(tuple(int(v) for v in table.level[row]))
                scale = (len(model.energies_ghz) / 16) ** 3
                duration = float(table.duration_ns[row])
                frequency = abs(float(table.tone_frequency_ghz[row, 0]))
                env_index = table.tone_envelope[row, 0]
                env = table.envelopes[env_index]
                # Smooth real Gaussian calibration: ~0.20--0.28 CPU-s/key at
                # 16 levels and 254--256 ns. Shape and drive enter explicitly;
                # this is a cost model, not an RWA admission certificate.
                peak = bounds.peaks[env_index]
                motion = 0.0
                if len(env.samples) > 1 and peak > 0:
                    motion = float(
                        np.max(np.abs(np.diff(env.samples) / np.diff(env.sample_ns)))
                        / peak
                    )
                retained = retained_edges(
                    model,
                    int(table.tone_port[row, 0]),
                    float(table.tone_frequency_ghz[row, 0]),
                )
                strength = (
                    abs(table.tone_amplitude[row, 0])
                    * peak
                    * 2
                    * np.linalg.norm(retained, 2)
                )
                build = (
                    0.22
                    * duration
                    / 256
                    * (1 + 4 * motion + 0.1 * 2 * np.pi * duration * strength)
                )
                # Gate: four joint matrix actions plus phase and row guard/key.
                gate += scale * (build + count * (0.00003 + 0.00008))
                cycles = duration * frequency
                tail = (
                    duration - np.floor(cycles + 1e-10) / frequency
                    if frequency
                    else duration
                )
                # High carriers use ODE fractional tails; below 1 GHz Sambe
                # uses spectral tails and more expensive subperiod playback.
                playback = (0.00000075 if frequency >= 1 else 0.000008) * cycles
                fractional = 0.0009 * tail if frequency >= 1 else 0.00010
                sambe += scale * count * (playback + fractional + 0.00002)
                uses += count
            sambe += scale * 0.008  # shared carrier/shape spectral construction
            choices.append(
                SingleGateChoice(
                    tuple(sorted(group)),
                    uses,
                    float(gate),
                    float(sambe),
                    bool(gate <= 0.8 * sambe),
                )
            )
        return tuple(choices)

    return plan.fact(_SINGLE_GATE_PLAN, build)


def single_gate_preferred(
    table: SegmentTable, rows: NDArray[np.int64], context: KernelContext
) -> NDArray[np.bool_]:
    """Borrow complete-acquire cold choices for current rows before guards."""
    preferred = {
        key
        for choice in populate_single_gate_plan(
            table, context.source, context.work_plan
        )
        if choice.preferred
        for key in choice.keys
    }
    if not preferred:
        return np.zeros(len(rows), dtype=bool)
    return np.array(
        [canonical_key(table, int(row), context.source) in preferred for row in rows],
        dtype=bool,
    )
