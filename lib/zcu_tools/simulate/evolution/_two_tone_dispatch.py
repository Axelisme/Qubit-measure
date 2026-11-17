"""Calibrated constant two-tone costs, independent of numerical admission."""

import numpy as np
from numpy.typing import NDArray

from .kernel import KernelContext, SegmentTable
from .kernels._floquet_policy import (
    MAX_PHASE_SAMPLES,
    phase_sample_count,
    probe_bin_index,
)
from .kernels.floquet import FloquetKernel


def first_candidate(
    table: SegmentTable,
    rows: NDArray[np.int64],
    context: KernelContext,
) -> NDArray[np.str_]:
    """Return the cheaper family name for each constant two-tone candidate row.

    This calibrated cost model uses duration, dimension, spectral width, carrier
    and gain-weighted drive norms to estimate DOP853 steps. Floquet charges a
    modal apply, phase-sample frame work and complete 2 MHz probe-bin certificates,
    amortized over *all* rows sharing its canonical frame key. Certificate work
    ramps with drive action; this predicts cost, not a guard outcome. Local ODE
    work scales with budget**(-1/8). No guard, trial integration, cache inspection
    or timing occurs here.

    The 2026-11-15 calibration uses the committed FTF_MODEL in
    tools/family_pareto_settings.py with 16 retained levels: ten interleaved
    three-pair controls, 32--2000 ns, 1--11-point duration and detuning
    inventories, Stark gains .2--.5 with a 1e-4 probe, and balanced distinct-port
    gains .1--.2.
    Frame/bin charges fit log CPU on the eight admitted controls; modal apply
    uses 8 ms/row (observed 2--11 ms). DOP853's coefficient fits all ten controls;
    the dimension exponent retains the earlier 8--20-level fluxonium fit.
    These are cost estimates, not universal timing or admission predictions.
    Cold distinct-port balanced drives were also calibrated
    on the 16-level half-flux mux model: 128--512 ns, gains .1--.2, carriers
    .295/.597 GHz, eleven probe points spanning 2 MHz. At 512 ns and gain .2,
    measured frame/certificate work was 4.32/8.60 CPU-s versus 1.4/2.1 estimated.
    Their multipliers ramp to 3.1/4.1 with both drive actions and balance; this
    charges construction, not a prediction that a strong probe must be rejected.
    Every candidate still needs its own guard;
    rejection tries the other family. Input rows must include the acquire-wide
    candidate set, never a worker subset. Return a row-aligned string array:
    'dop853' when direct integration is cheaper, otherwise 'floquet'.
    """
    source, budget = context.source, context.budget
    preferred = np.full(len(rows), "floquet", dtype="U16")
    if not len(rows):
        return preferred
    keys = FloquetKernel().prepare_keys(table, rows, context)
    _, inverse = np.unique(keys, return_inverse=True)
    for group in range(int(inverse.max()) + 1):
        positions = np.flatnonzero(inverse == group)
        selected = rows[positions]
        row = selected[0]
        model = table.model(source, row)
        scale = len(model.energies_ghz) / 16
        width = float(np.ptp(model.energies_ghz))
        norms = np.array(
            [
                np.linalg.norm(d.in_phase_ghz, 2) + np.linalg.norm(d.quadrature_ghz, 2)
                for d in model.drives
            ]
        )
        gains = np.abs(table.tone_amplitude[selected, :2])
        ports = table.tone_port[selected, :2]
        drive_strength = gains * norms[ports]
        strength = np.sum(drive_strength, axis=1)
        stark = np.argmax(gains, axis=1)
        frequency = table.tone_frequency_ghz[selected, :2]
        stark_frequency = np.abs(frequency[np.arange(len(selected)), stark])
        duration = table.duration_ns[selected]
        tolerance_work = (1e-3 / budget) ** 0.125
        # 2026-11-15: the ten tool-owned controls above need 1.633 times the
        # earlier single-fluxonium CPU estimate (geometric mean). Keep its
        # 0.324 dimension power: bandwidth already grows with dimension.
        direct = (
            5.67e-5
            * duration
            * (width + np.max(np.abs(frequency), axis=1) + 2 * strength)
            * (1 + 4 * np.sqrt(strength) + 4 * strength)
            * scale**0.324
            * tolerance_work
        )
        probe = 1 - stark
        bins = np.empty(len(selected), dtype=[("port", "i8"), ("bin", "f8")])
        bins["port"] = table.tone_port[selected, probe]
        bins["bin"] = probe_bin_index(frequency[np.arange(len(selected)), probe])
        bin_count = len(np.unique(bins))
        # Smooth expected certificate charge, without predicting guard acceptance.
        certificate_fraction = np.minimum(1, duration * strength / 0.15)
        safe_frequency = np.maximum(stark_frequency, 1e-12)
        samples = np.minimum(
            MAX_PHASE_SAMPLES,
            phase_sample_count(width + 2 * strength, safe_frequency),
        )
        # A weak probe does not pay the dual-strong cold surcharge. Balance is
        # gain-weighted operator strength, not raw DAC gain or port independence.
        weakest = np.min(drive_strength, axis=1)
        strongest = np.max(drive_strength, axis=1)
        balance = np.divide(
            weakest, strongest, out=np.zeros_like(weakest), where=strongest > 0
        )
        dual_work = (
            (ports[:, 0] != ports[:, 1])
            * np.minimum(1, 4 * balance)
            * np.minimum(1, duration * weakest / 0.15)
        )
        # Log-CPU fit: 4.181 s per 512 frame samples, 0.274 s per probe bin.
        # The fit moves shared work out of the old 1.05 s/bin certificate charge.
        frame = 4.18 * samples / 512 * scale**5 * (1 + 2.1 * dual_work)
        certificates = (
            0.274 * bin_count * scale**6 * certificate_fraction * (1 + 3.1 * dual_work)
        )
        floquet = 0.008 * scale**6 + (frame + certificates) / len(selected)
        context.work_plan.routing.expect("dop853", table, selected, direct)
        context.work_plan.routing.expect("floquet", table, selected, floquet)
        preferred[positions] = np.where(direct < floquet, "dop853", "floquet")
    return preferred
