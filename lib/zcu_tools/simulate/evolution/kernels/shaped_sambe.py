"""Low-reuse real-envelope periods from finite Hilbert Fourier matrices."""

from dataclasses import dataclass

import numpy as np
from numba import njit
from numpy.typing import NDArray
from scipy.linalg import expm

from ..declarations import (
    AlgorithmIdentity,
    Eligibility,
    GuardDeclaration,
    KernelDeclaration,
    ReuseDeclaration,
)
from ..kernel import GuardPreparation, GuardResult, Prepared, SegmentTable
from ..secular import ModelSource, SecularModel
from ._period_table import (
    PeriodEntry,
    PeriodPlayback,
    PeriodPrepared,
    SecularMaps,
    UnitaryTable,
    envelopes,
    periods,
)
from ._shaped import drive_bounds, envelope_bounds
from .dop853 import DOP853Kernel
from .period_table import PeriodTableKernel, advance_tail


def _spectral(
    model: SecularModel,
    port: int,
    frequency: float,
    gain: float,
    modes: int,
    duration: float,
) -> NDArray[np.complex128]:
    """Return Fourier blocks of U(duration, theta), from a central Sambe input."""
    drive = model.drives[port]
    coupling = gain * (drive.in_phase_ghz - 1j * drive.quadrature_ghz) / 2
    return _spectral_blocks(model.energies_ghz, coupling, frequency, modes, duration)


@njit(nogil=True, cache=True)
def _spectral_blocks(energies, coupling, frequency, modes, duration):
    """Run matrix assembly, eigensolve and reconstruction without the GIL."""
    size = len(energies)
    count = 2 * modes + 1
    matrix = np.zeros((count * size, count * size), dtype=np.complex128)
    for k in range(count):
        block = slice(k * size, (k + 1) * size)
        matrix[block, block] = np.diag(energies + (k - modes) * frequency)
        if k:
            previous = slice((k - 1) * size, k * size)
            matrix[block, previous] = coupling
            matrix[previous, block] = coupling.conj().T
    values, vectors = np.linalg.eigh(matrix)
    central = np.ascontiguousarray(vectors[modes * size : (modes + 1) * size])
    evolved = (vectors * np.exp(-2j * np.pi * values * duration)) @ central.conj().T
    result = np.zeros((16, size, size), dtype=np.complex128)
    for index in range(count):
        k = index - modes
        block = evolved[index * size : (index + 1) * size]
        result[k % 16] = np.exp(2j * np.pi * k * frequency * duration) * block
    return result


def _coefficients(
    model: SecularModel,
    port: int,
    frequency: float,
    bound: float,
    duration: float,
    orders: tuple[int, int],
) -> NDArray[np.complex128]:
    nodes, modes = orders
    angles = np.pi * (np.arange(nodes) + 0.5) / nodes
    amplitudes = bound * (1 + np.cos(angles)) / 2
    values = np.array(
        [_spectral(model, port, frequency, a, modes, duration) for a in amplitudes]
    )
    transform = 2 / nodes * np.cos(np.arange(nodes)[:, None] * angles)
    transform[0] /= 2
    coefficients = np.zeros(
        (1, 8, 16, len(model.energies_ghz), len(model.energies_ghz)), dtype=complex
    )
    coefficients[0, :nodes] = np.einsum("ab,bkij->akij", transform, values)
    return coefficients


class _SpectralFamily:
    """Reuse immutable spectral builds within one guard/prepare call."""

    def __init__(self) -> None:
        self.errors: dict[int, float] = {}
        self.polynomials: dict[
            tuple[int, int], tuple[NDArray[np.complex128], float, float]
        ] = {}

    def error(
        self,
        model: SecularModel,
        port: int,
        frequency: float,
        bound: float,
        step: float,
        modes: int,
    ) -> float:
        if modes not in self.errors:
            fine = _spectral(model, port, frequency, bound, modes + 1, step)
            coarse = _spectral(model, port, frequency, bound, modes, step)
            self.errors[modes] = float(
                np.sum(np.linalg.norm(fine - coarse, axis=(-2, -1)))
            )
        return self.errors[modes]

    def polynomial(
        self,
        model: SecularModel,
        port: int,
        frequency: float,
        bound: float,
        step: float,
        orders: tuple[int, int],
    ) -> tuple[NDArray[np.complex128], float, float]:
        if orders not in self.polynomials:
            nodes, _ = orders
            coefficients = _coefficients(model, port, frequency, bound, step, orders)
            last = float(
                np.sum(np.linalg.norm(coefficients[:, nodes - 1], axis=(-2, -1)))
            )
            previous = float(
                np.sum(np.linalg.norm(coefficients[:, nodes - 2], axis=(-2, -1)))
            )
            ratio = last / max(previous, 1e-30)
            interpolation = last * ratio / max(1 - ratio, 1e-30) if nodes == 4 else last
            coefficients.setflags(write=False)
            self.polynomials[orders] = coefficients, interpolation, ratio
        return self.polynomials[orders]


@dataclass(frozen=True)
class _Data:
    # Period tables own arrays. Models/ports identify the frozen-tail Hamiltonian.
    periods: PeriodPrepared
    models: tuple[SecularModel, ...]
    ports: tuple[int, ...]

    @property
    def nbytes(self) -> int:
        # Models belong to ModelSource; only newly owned arrays are charged.
        return self.periods.nbytes


class ShapedSambeKernel:
    """Finite-Fourier real-envelope propagation for low-reuse shaped keys.

    Keys contain level, port, signed GHz carrier, peak absolute gain,
    duration in ns and gain-weighted envelope slope. They fix subperiod resolution.
    Comparing
    adjacent harmonic cutoffs (2/3, then 3/4) estimates Fourier truncation.
    Four/eight amplitude nodes control interpolation. Guard preparation is
    call-local and never reads density. Rejections retain the old shaped
    decision order. Apply uses frozen-envelope secular Strang steps, never
    retries, clips or normalizes density. Complex envelopes and DC are rejected.
    """

    declaration = KernelDeclaration(
        identity=AlgorithmIdentity("shaped_sambe", "spectral-reuse-v2"),
        eligibility=Eligibility(1, 1, "shaped"),
        guard=GuardDeclaration(("model-build",), builds_model=True),
        accuracy="guarded-approximation",
        reuse=ReuseDeclaration("shaped_sambe/spectral-reuse-v2/serial"),
        fallback="legacy-order",
    )

    name = "shaped_sambe"
    guard_builds = True

    def prepare_keys(
        self,
        table: SegmentTable,
        rows: NDArray[np.int64],
    ) -> NDArray[np.void]:
        """Key peak gain, weighted slope and duration to fix subperiod resolution."""
        base = PeriodTableKernel().prepare_keys(table, rows)
        keys = np.empty(
            len(rows),
            dtype=base.dtype.descr
            + [("duration", "f8"), ("bound", "f8"), ("slope", "f8")],
        )
        for name in base.dtype.names or ():
            keys[name] = base[name]
        peaks, slopes = envelope_bounds(table)
        keys["slope"] = (
            np.abs(table.tone_amplitude[rows, 0]) * slopes[table.tone_envelope[rows, 0]]
        )
        keys["bound"] = (
            np.abs(table.tone_amplitude[rows, 0]) * peaks[table.tone_envelope[rows, 0]]
        )
        keys["duration"] = table.duration_ns[rows]
        return keys

    def _subdivisions(
        self,
        table: SegmentTable,
        rows: NDArray[np.int64],
        source: ModelSource,
    ) -> NDArray[np.int64]:
        strength, gamma = drive_bounds(table, rows, source)
        peaks, slopes = envelope_bounds(table)
        envelope = table.tone_envelope[rows, 0]
        gain = np.abs(table.tone_amplitude[rows, 0])
        frequency = np.maximum(np.abs(table.tone_frequency_ghz[rows, 0]), 1e-20)
        motion = 2 * np.pi * gain * strength * slopes[envelope] / frequency**2
        split = (
            table.duration_ns[rows]
            * gamma
            * (gain * peaks[envelope] * strength / frequency) ** 2
        )
        return np.maximum(
            1, np.ceil(np.sqrt(np.maximum(motion / 0.01, split / 0.01)))
        ).astype(np.int64)

    def _eligible(
        self,
        table: SegmentTable,
        rows: NDArray[np.int64],
        source: ModelSource,
        budget: float,
    ) -> NDArray[np.bool_]:
        """Check density-free bandwidth, motion and split bounds before building."""
        del budget
        strength, _ = drive_bounds(table, rows, source)
        keys = self.prepare_keys(table, rows)
        frequency = np.abs(keys["frequency"])
        real = np.array([np.all(e.samples.imag == 0) for e in table.envelopes])
        peaks, slopes = envelope_bounds(table)
        envelope = table.tone_envelope[rows, 0]
        resolved = slopes[envelope] <= 2 * np.maximum(
            peaks[envelope], 1e-30
        ) * frequency * self._subdivisions(table, rows, source)
        return (
            (frequency >= 0.03)
            & (keys["bound"] > 0)
            & resolved
            & real[table.tone_envelope[rows, 0]]
            & (
                keys["bound"] * strength
                <= 0.5 * frequency * self._subdivisions(table, rows, source)
            )
            & (self._subdivisions(table, rows, source) <= 32)
        )

    def guard(
        self,
        table: SegmentTable,
        rows: NDArray[np.int64],
        source: ModelSource,
        budget: float,
    ) -> NDArray[np.bool_]:
        """Return the same model-only convergence mask as guard_prepared."""
        return self.guard_prepared(table, rows, source, budget).accepted

    def _build(
        self,
        keys: NDArray[np.void],
        table: SegmentTable,
        rows: NDArray[np.int64],
        source: ModelSource,
        budget: float,
    ) -> tuple[_Data, NDArray[np.float64]]:
        entries, models, ports, errors = [], [], [], []
        subdivisions = self._subdivisions(table, rows, source)
        fallback = DOP853Kernel()
        # A timeline's rise/fall ramps arrive together. Only this call owns the
        # workspace; accepted payloads retain their immutable coefficients.
        families: dict[
            tuple[str, tuple[int, ...], int, float, float, int, float, str],
            _SpectralFamily,
        ] = {}
        for index, key in enumerate(keys):
            model = source.model(tuple(int(v) for v in key["level"]))
            port, f, bound = (
                int(key["port"]),
                float(key["frequency"]),
                float(key["bound"]),
            )
            m = int(subdivisions[index])
            step = 1 / (abs(f) * m)
            # Start with a cubic amplitude polynomial; refine its estimated
            # omitted tail before accepting a key.
            nodes = 4
            cycles = max(1, table.duration_ns[rows[index]] / step)
            modes = 2
            # Compare the limiting amplitude, not 128 carrier-phase ODEs.
            # The bandwidth guard and the retained polynomial's last degree
            # accompany this calibrated (not rigorous) convergence estimate.
            family_key = (
                model.identity,
                model.level,
                port,
                f,
                bound,
                m,
                budget,
                self.declaration.identity.revision,
            )
            family = families.get(family_key)
            if family is None:
                family = _SpectralFamily()
                families[family_key] = family
            error = family.error(model, port, f, bound, step, modes)
            if error * cycles > budget / 4:
                modes = 3
                error = family.error(model, port, f, bound, step, modes)
            coefficients, interpolation, ratio = family.polynomial(
                model, port, f, bound, step, (nodes, modes)
            )
            if ratio > 0.25 or interpolation * cycles > budget / 8:
                nodes = 8
                coefficients, interpolation, _ = family.polynomial(
                    model, port, f, bound, step, (nodes, modes)
                )
            errors.append((error + interpolation) * cycles)
            outgoing = model.rates_per_ns.sum(axis=0)
            populations = np.asarray(
                expm(step / 2 * (model.rates_per_ns - np.diag(outgoing))), dtype=float
            )
            coherences = np.exp(
                -step
                / 2
                * ((outgoing[:, None] + outgoing[None, :]) / 2 + model.dephasing_per_ns)
            )
            for array in (coefficients, populations, coherences):
                array.setflags(write=False)
            representative = rows[index : index + 1]
            direct = fallback.prepare(
                fallback.prepare_keys(table, representative),
                table,
                representative,
                source,
                min(budget, 1e-8),
            )
            # Integer-period blocks avoid thousands of tiny matrix products on
            # long high-carrier ramps. Subperiod playback retains every phase.
            drive = model.drives[port]
            strength = max(
                np.linalg.norm(drive.in_phase_ghz, 2),
                np.linalg.norm(drive.quadrature_ghz, 2),
            )
            frozen_ns = min(
                20.0,
                np.sqrt(0.05 / max(2 * np.pi * strength * float(key["slope"]), 1e-30)),
            )
            block = max(1, int(frozen_ns * abs(f))) if m == 1 else 1
            entries.append(
                PeriodEntry(
                    direct,
                    min(budget, 1e-8),
                    f,
                    bound,
                    coefficients,
                    populations,
                    coherences,
                    m,
                    block,
                    True,
                    modes,
                )
            )
            models.append(model)
            ports.append(port)
        return _Data(
            PeriodPrepared(tuple(entries)), tuple(models), tuple(ports)
        ), np.asarray(errors)

    def guard_prepared(
        self,
        table: SegmentTable,
        rows: NDArray[np.int64],
        source: ModelSource,
        budget: float,
    ) -> GuardResult:
        """Build eligible keys once, hand accepted model-only periods to Engine."""
        accepted = self._eligible(table, rows, source, budget)
        selected = rows[accepted]
        preparations = []
        if len(selected):
            keys, representatives, inverse = np.unique(
                self.prepare_keys(table, selected),
                return_index=True,
                return_inverse=True,
            )
            prepared, errors = self._build(
                keys, table, selected[representatives], source, budget
            )
            passed = errors[inverse] <= budget / 2
            accepted[np.flatnonzero(accepted)] = passed
            preparations.append(GuardPreparation(keys, prepared))
        return GuardResult(accepted, tuple(preparations))

    def prepare(
        self,
        keys: NDArray[np.void],
        table: SegmentTable,
        rows: NDArray[np.int64],
        source: ModelSource,
        budget: float,
    ) -> Prepared:
        """Build immutable spectral periods from aligned representative rows."""
        return self._build(keys, table, rows, source, budget)[0]

    def cost(
        self,
        table: SegmentTable,
        rows: NDArray[np.int64],
    ) -> NDArray[np.float64]:
        """Estimate carrier substeps; Engine owns the sole parallel layer."""
        return 1 + table.duration_ns[rows] * np.abs(table.tone_frequency_ghz[rows, 0])

    def bind_input(
        self,
        prepared: Prepared,
        key_index: int,
        table: SegmentTable,
        rows: NDArray[np.int64],
        source_initial: NDArray[np.complex128],
        budget: float,
    ) -> None:
        """Decline input binding; every prepared array is model-only."""
        del prepared, key_index, table, rows, source_initial, budget

    def apply(
        self,
        prepared: Prepared,
        key_index: NDArray[np.int64],
        table: SegmentTable,
        rows: NDArray[np.int64],
        states: NDArray[np.complex128],
        budget: float,
    ) -> NDArray[np.bool_]:
        """Advance real envelopes; fractional steps use frozen spectral tails.

        Incompatible payloads raise ValueError; nonfinite output raises
        RuntimeError. Return an all-False accuracy-exceedance mask.
        """
        del budget
        if not isinstance(prepared, _Data):
            raise ValueError("shaped_sambe requires its own prepared payload")
        for index, entry in enumerate(prepared.periods.entries):
            mask = key_index == index
            if not np.any(mask):
                continue
            selected = rows[mask]
            chunk = states[mask].copy()
            step = 1 / (abs(entry.frequency) * entry.subdivisions)
            cycles = np.floor(table.duration_ns[selected] / step).astype(np.int64)
            elapsed = (np.arange(int(cycles.max()))[None, :] + 0.5) * step
            elapsed = np.asarray(
                np.broadcast_to(elapsed, (len(selected), elapsed.shape[1])), dtype=float
            )
            periods(
                chunk,
                PeriodPlayback(
                    np.zeros(len(selected), dtype=np.int64),
                    envelopes(table, selected, elapsed),
                    table.tone_theta_rad[selected, 0],
                    cycles,
                ),
                UnitaryTable(
                    entry.bound,
                    entry.coefficients,
                    entry.subdivisions,
                    entry.block_cycles,
                    entry.project_unitary,
                    1 if entry.frequency > 0 else -1,
                    entry.harmonics,
                ),
                SecularMaps(entry.populations, entry.coherences),
            )
            if abs(entry.frequency) >= 1:
                # Sub-ns tails cost less in the existing compiled density
                # solver than another Fourier-matrix eigendecomposition.
                advance_tail(
                    entry,
                    table,
                    selected,
                    np.asarray(cycles * step, dtype=np.float64),
                    chunk,
                )
                states[mask] = chunk
                continue
            model, port = prepared.models[index], prepared.ports[index]
            for j, row in enumerate(selected):
                start = cycles[j] * step
                tail = table.duration_ns[row] - start
                if tail <= 1e-10:
                    continue
                amp = float(
                    envelopes(table, np.array([row]), np.array([[start + tail / 2]]))[
                        0, 0
                    ].real
                )
                blocks = _spectral(model, port, entry.frequency, amp, 3, tail)
                theta = (
                    table.tone_theta_rad[row, 0] + 2 * np.pi * entry.frequency * start
                )
                harmonics = np.fft.fftfreq(16) * 16
                u = np.einsum("k,kij->ij", np.exp(1j * harmonics * theta), blocks)
                # Polar projection changes U, not density or populations.
                left, _, right = np.linalg.svd(u)
                u = left @ right
                ratio = tail / step
                population = np.asarray(
                    expm(
                        tail
                        / 2
                        * (model.rates_per_ns - np.diag(model.rates_per_ns.sum(axis=0)))
                    ),
                    dtype=complex,
                )
                coherence = entry.coherences**ratio
                rho = chunk[j]
                for half in range(2):
                    diagonal = population @ np.diag(rho)
                    rho = rho * coherence
                    np.fill_diagonal(rho, diagonal)
                    if half == 0:
                        rho = u @ rho @ u.conj().T
                chunk[j] = rho
            if not np.isfinite(chunk).all():
                raise RuntimeError("shaped_sambe produced nonfinite density")
            states[mask] = chunk
        return np.zeros(len(rows), dtype=np.bool_)
