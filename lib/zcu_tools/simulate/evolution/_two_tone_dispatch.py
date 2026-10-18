"""Calibrated constant two-tone costs, independent of numerical admission."""

import numpy as np
from numpy.typing import NDArray

from .kernel import SegmentTable
from .kernels.floquet import FloquetKernel
from .secular import ModelSource


def direct_preferred(
    table: SegmentTable,
    rows: NDArray[np.int64],
    source: ModelSource,
    budget: float,
) -> NDArray[np.bool_]:
    """Return cheapest-first choices for all constant two-tone candidate rows.

    This calibrated cost model uses duration, dimension, spectral width, carrier
    and gain-weighted drive norms to estimate DOP853 steps. Floquet charges a
    modal apply, phase-sample frame work and complete 2 MHz probe-bin certificates,
    amortized over *all* rows sharing its canonical frame key. Certificate work
    ramps with drive action; this predicts cost, not a guard outcome. Local ODE
    work scales with budget**(-1/8). No guard, trial integration, cache inspection
    or timing occurs here.

    Coefficients were calibrated on 16-level FTF S9 throughput (16 ns, weak),
    S9 qualitative (514 ns, strong), and 16--512 ns / .02--.5 gain boundary rows
    at .5 and 7.29 GHz. They are estimates outside that calibration, not universal
    timing predictions. Every candidate still needs its own numerical guard;
    rejection tries the other family. Input rows must include the acquire-wide
    candidate set, never a worker subset. Return a bool array aligned with rows.
    """
    preferred = np.zeros(len(rows), dtype=bool)
    if not len(rows):
        return preferred
    keys = FloquetKernel().prepare_keys(table, rows)
    _, inverse = np.unique(keys, return_inverse=True)
    for group in range(int(inverse.max()) + 1):
        positions = np.flatnonzero(inverse == group)
        selected = rows[positions]
        row = selected[0]
        model = source.model(tuple(int(v) for v in table.level[row]))
        scale = len(model.energies_ghz) / 16
        width = float(np.ptp(model.energies_ghz))
        norms = np.array(
            [
                np.linalg.norm(d.in_phase_ghz, 2) + np.linalg.norm(d.quadrature_ghz, 2)
                for d in model.drives
            ]
        )
        gains = np.abs(table.tone_amplitude[selected, :2])
        strength = np.sum(gains * norms[table.tone_port[selected, :2]], axis=1)
        stark = np.argmax(gains, axis=1)
        frequency = table.tone_frequency_ghz[selected, :2]
        stark_frequency = np.abs(frequency[np.arange(len(selected)), stark])
        duration = table.duration_ns[selected]
        tolerance_work = (1e-3 / budget) ** 0.125
        direct = (
            8e-5
            * duration
            * (width + np.max(np.abs(frequency), axis=1) + 2 * strength)
            * (1 + 4 * np.sqrt(strength) + 4 * strength)
            * scale**3
            * tolerance_work
        )
        probe = 1 - stark
        bins = np.empty(len(selected), dtype=[("port", "i8"), ("bin", "f8")])
        bins["port"] = table.tone_port[selected, probe]
        bins["bin"] = np.round(frequency[np.arange(len(selected)), probe] / 0.002)
        bin_count = len(np.unique(bins))
        # Smooth expected certificate charge, without predicting guard acceptance.
        certificate_fraction = np.minimum(1, duration * strength / 0.15)
        safe_frequency = np.maximum(stark_frequency, 1e-12)
        samples = np.minimum(
            4096,
            2
            ** np.ceil(
                np.log2(np.maximum(128, 32 * (width + 2 * strength) / safe_frequency))
            ),
        )
        frame = 1.4 * samples / 512 * scale**5
        certificates = 1.05 * bin_count * scale**6 * certificate_fraction
        floquet = 0.07 * scale**6 + (frame + certificates) / len(selected)
        preferred[positions] = direct < floquet
    return preferred
