"""Cold prices for omission-expanded multitone RWA and full-drive integration."""

from hashlib import sha256

import numpy as np

from .kernel import KernelContext, SegmentTable
from .kernels.multitone_rwa import MultitoneRWAKernel, multitone_retention
from .secular import ModelSource
from .workplan import FactKey, WorkPlan


def populate_multitone_plan(
    table: SegmentTable, source: ModelSource, plan: WorkPlan, budget: float = 1e-3
) -> frozenset[int]:
    """Return global rows whose expanded RWA costs more than direct DOP853.

    Uses the complete acquire once, before prefix and worker subsets. Numerical
    admission remains the kernel's responsibility. Only admitted rows that
    restore omitted edges need a comparison; unexpanded RWA retains its order.
    Prepare work is amortized over source-cumulative exact physics-key usage,
    including this inventory. No resident-cache state or timing probes enter
    the decision. The frozen row selection is shared by all dispatch stages.

    Calibration uses the 16-level half-flux close-carrier mux model, 256--512 ns
    constant/Gaussian rows at pi and twice-pi gains. Expanded dimensions 174/256
    and fastest restored frequencies 7.84/8.66 GHz measured RWA apply around
    0.33--0.41/0.85--0.93 CPU-s. The 60 us per ns-GHz coefficient leaves headroom
    over these measured coefficients, since thermal/native and pure-state
    inventories have different adaptive-step costs. Guard allowance is 3 ms
    per occupied block unit, prepare 2 ms amortized per exact key. Direct uses
    the existing two-tone 80 us bandwidth/action model, with a 0.7 factor for
    sampled envelopes (measured Gaussian/constant direct ratio 0.65--0.69).
    These conservative cold prices are scheduling estimates, not universal
    runtime bounds.
    """

    def build() -> frozenset[int]:
        if budget < 1e-3:
            return frozenset()
        context = KernelContext(source, budget, plan)
        kernel = MultitoneRWAKernel()
        candidates = np.flatnonzero(table.tones >= 2)
        admitted = candidates[kernel.guard(table, candidates, context)]
        rows = np.array(
            [
                row
                for row in admitted
                if any(multitone_retention(table, int(row), context).edges)
            ],
            dtype=np.int64,
        )
        if not len(rows):
            return frozenset()
        physics = kernel.prepare_keys(table, rows, context)
        keys = []
        inventory: dict[bytes, int] = {}
        for row, key in zip(rows, physics, strict=True):
            model = source.model(tuple(int(v) for v in table.level[row]))
            identity = sha256(
                b"multitone-cold" + model.identity.encode() + key.tobytes()
            ).digest()
            keys.append(identity)
            inventory[identity] = inventory.get(identity, 0) + 1
        usage = source.record_gate_usage(inventory)
        return frozenset(
            int(row)
            for row, key in zip(rows, keys, strict=True)
            if _direct_cheaper(table, int(row), context, usage[key])
        )

    return plan.fact(FactKey(("multitone-cold-plan", budget), frozenset), build)


def _direct_cheaper(
    table: SegmentTable, row: int, context: KernelContext, uses: int
) -> bool:
    model = context.source.model(tuple(int(v) for v in table.level[row]))
    retained = multitone_retention(table, row, context)
    assert retained.dimension is not None
    size = len(model.energies_ghz)
    scale = (size / 16) ** 3
    count = int(table.tones[row])
    frequencies = table.tone_frequency_ghz[row, :count]
    strength = sum(
        abs(table.tone_amplitude[row, tone])
        * (
            np.linalg.norm(model.drives[port].in_phase_ghz, 2)
            + np.linalg.norm(model.drives[port].quadrature_ghz, 2)
        )
        for tone, port in enumerate(table.tone_port[row, :count])
    )
    duration = table.duration_ns[row]
    bandwidth = float(np.ptp(model.energies_ghz)) + max(abs(frequencies)) + 2 * strength
    action = 1 + 4 * np.sqrt(strength) + 4 * strength
    tolerance = (1e-3 / context.budget) ** 0.125
    sampled = any(
        table.envelopes[e].kind != "constant" for e in table.tone_envelope[row, :count]
    )
    direct = (
        8e-5
        * duration
        * bandwidth
        * action
        * scale
        * tolerance
        * (0.7 if sampled else 1.0)
    )
    restored_frequency = max(
        abs(model.energies_ghz[i] - model.energies_ghz[j] + frequency)
        for frequency, edges in zip(frequencies, retained.edges, strict=True)
        for i, j in edges
    )
    occupancy = retained.dimension / size**2
    # Expanded graphs incur fast interaction-picture oscillations and additional
    # coefficient/guard work even when their omission certificate is excellent.
    prepare = 0.002 * scale * (1 + occupancy) / uses
    guard = 0.003 * scale * (1 + occupancy)
    apply = (
        6e-5
        * duration
        * max(0.05, restored_frequency)
        * action
        * (1 + occupancy)
        * scale
        * tolerance
    )
    return bool(direct < guard + prepare + apply)
