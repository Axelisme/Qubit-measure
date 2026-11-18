"""Cold dissipative DC channel prices, independent of numerical admission."""

import numpy as np
from numpy.typing import NDArray

from .kernel import KernelContext, SegmentTable
from .kernels.slow_envelope import SlowEnvelopeKernel
from .workplan import FactKey


def direct_preferred(
    table: SegmentTable, rows: NDArray[np.int64], context: KernelContext
) -> frozenset[int]:
    """Return complete-acquire slow rows cheaper to integrate with DOP853.

    rows contains every structurally slow row, not a prefix/worker subset.
    Only dissipative zero-carrier rows pay the dense Lindblad channel charge.
    Exact prepare keys amortize construction over this acquire, never history
    or resident cache state. Model lookup and key grouping build no channels.
    The measured single-thread expm charges are 0.035/1.71/19.1/120 s at
    M=16/32/48/64. Use 1.71*(M/32)**6 for O((M**2)**3) construction and
    0.000422*(M/32)**4 per application. Direct uses the existing two-tone
    bandwidth/action scheduling model; neither estimate certifies admission.
    """

    def build() -> frozenset[int]:
        selected = rows[table.tone_frequency_ghz[rows, 0] == 0]
        if not len(selected):
            return frozenset()
        _, first, inverse, counts = np.unique(
            SlowEnvelopeKernel().prepare_keys(table, selected, context),
            return_index=True,
            return_inverse=True,
            return_counts=True,
        )
        preferred: set[int] = set()
        for group, representative in enumerate(first):
            row = selected[representative]
            model = table.model(context.source, row)
            if not (np.any(model.rates_per_ns) or np.any(model.dephasing_per_ns)):
                continue
            size = len(model.energies_ghz)
            drive = model.drives[table.tone_port[row, 0]]
            strength = abs(table.tone_amplitude[row, 0]) * (
                np.linalg.norm(drive.in_phase_ghz, 2)
                + np.linalg.norm(drive.quadrature_ghz, 2)
            )
            direct = (
                5.67e-5
                * table.duration_ns[row]
                * (float(np.ptp(model.energies_ghz)) + 2 * strength)
                * (1 + 4 * np.sqrt(strength) + 4 * strength)
                * (size / 16) ** 0.324
                * (1e-3 / context.budget) ** 0.125
            )
            channel = 1.71 * (size / 32) ** 6 / counts[group]
            channel += 0.000422 * (size / 32) ** 4
            consumers = selected[inverse == group]
            context.work_plan.routing.expect(
                "dop853", table, consumers, np.full(len(consumers), direct)
            )
            context.work_plan.routing.expect(
                "slow_envelope", table, consumers, np.full(len(consumers), channel)
            )
            if direct < channel:
                preferred.update(int(value) for value in consumers)
        return frozenset(preferred)

    return context.work_plan.fact(
        FactKey(("slow-dc-direct", id(table), context.budget), frozenset), build
    )
