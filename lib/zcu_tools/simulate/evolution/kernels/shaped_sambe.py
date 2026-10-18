"""Smooth-envelope periods from shared finite Hilbert Fourier matrices."""

from dataclasses import dataclass

import numpy as np
from numba import njit
from numpy.typing import NDArray
from scipy.linalg import expm

from .._carrier_plan import ramp_families_for_rows
from .._carrier_plan import subdivisions as ramp_subdivisions
from ..kernel import (
    ApplyContext,
    GuardPreparation,
    GuardResult,
    KernelContext,
    Prepared,
    SegmentTable,
    prepared_key_indices,
)
from ..secular import SecularModel
from ..workplan import RampFamilyPlan
from ._carrier_reuse import CarrierWorkspace
from ._period_table import (
    PeriodEntry,
    PeriodPlayback,
    PeriodPrepared,
    SecularMaps,
    UnitaryTable,
    advance_tail,
    band_keys,
    envelopes,
    periods,
)
from ._sambe import eigensystem
from ._secular import coherence_decay, population_generator
from ._shaped import drive_bounds
from ._workload_facts import envelope_bounds
from .dop853 import prepare_models


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
    values, vectors, central = eigensystem(energies, coupling, frequency, modes)
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
    """One model/port/carrier/peak-gain/subperiod spectral family.

    The constructor fixes all physical build inputs. Only harmonic cutoffs and
    amplitude-node counts vary within its caches. One guard/prepare call owns
    the instance; accepted payloads may retain its immutable coefficient arrays.
    """

    def __init__(
        self,
        model: SecularModel,
        port: int,
        frequency: float,
        bound: float,
        step: float,
        workspace: CarrierWorkspace | None,
    ) -> None:
        self.workspace = workspace
        self.model = model
        self.port = port
        self.frequency = frequency
        self.bound = bound
        self.step = step
        self.errors: dict[int, float] = {}
        self.polynomials: dict[
            tuple[int, int], tuple[NDArray[np.complex128], float, float]
        ] = {}

    def _physical(self, kind: str, orders: tuple[int, int]) -> NDArray[np.complex128]:
        """Build or interpolate physical arrays with all noncarrier inputs fixed."""
        subdivisions = int(round(1 / (abs(self.frequency) * self.step)))

        def build(frequency: float) -> NDArray[np.complex128]:
            step = 1 / (abs(frequency) * subdivisions)
            if kind == "peak":
                modes = orders[1]
                return np.array(
                    [
                        _spectral(self.model, self.port, frequency, self.bound, k, step)
                        for k in (modes, modes + 1)
                    ]
                )
            return _coefficients(
                self.model, self.port, frequency, self.bound, step, orders
            )

        workspace = self.workspace
        if workspace is None:
            return build(self.frequency)
        identity = (
            self.model.identity,
            self.model.level,
            self.port,
            self.bound,
            subdivisions,
            kind,
            orders,
        )
        return workspace.evaluate(identity, self.frequency, build, group=identity[:5])

    def error(self, modes: int) -> float:
        if modes not in self.errors:
            coarse, fine = self._physical("peak", (0, modes))
            self.errors[modes] = float(
                np.sum(np.linalg.norm(fine - coarse, axis=(-2, -1)))
            )
        return self.errors[modes]

    def polynomial(
        self,
        orders: tuple[int, int],
    ) -> tuple[NDArray[np.complex128], float, float]:
        if orders not in self.polynomials:
            nodes, _ = orders
            coefficients = self._physical("polynomial", orders)
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
    keys: NDArray[np.void]
    errors: NDArray[np.float64]

    @property
    def nbytes(self) -> int:
        # Models belong to ModelSource; only newly owned arrays are charged.
        return self.periods.nbytes + self.keys.nbytes + self.errors.nbytes


class ShapedSambeKernel:
    """Finite-Fourier smooth-envelope propagation for shaped keys at any reuse.

    Keys contain level, port, signed GHz carrier, peak absolute gain,
    duration in ns, gain-weighted slope and a complex-envelope flag.
    They fix subperiod resolution, including the tighter complex motion limit.
    Comparing
    adjacent harmonic cutoffs (2/3, then 3/4) estimates Fourier truncation.
    Four/eight amplitude nodes control interpolation. Guard preparation is
    call-local and never reads density. Rejections retain the old shaped
    decision order. Apply uses frozen-envelope secular Strang steps, never
    retries, clips or normalizes density. Complex amplitudes use their magnitude
    polynomial and add their angle to carrier phase. DC is rejected.
    """

    revision = "carrier-physical-v3"
    name = "shaped_sambe"
    guard_builds = True

    def prepare_keys(
        self,
        table: SegmentTable,
        rows: NDArray[np.int64],
        context: KernelContext,
    ) -> NDArray[np.void]:
        """Key peak, slope, duration and complex motion class to fix resolution."""
        base = band_keys(table, rows, context.work_plan)
        keys = np.empty(
            len(rows),
            dtype=base.dtype.descr
            + [("duration", "f8"), ("bound", "f8"), ("slope", "f8"), ("complex", "?")],
        )
        for name in base.dtype.names or ():
            keys[name] = base[name]
        bounds = envelope_bounds(table, context.work_plan)
        peaks, slopes = bounds.peaks, bounds.slopes
        keys["slope"] = (
            np.abs(table.tone_amplitude[rows, 0]) * slopes[table.tone_envelope[rows, 0]]
        )
        keys["bound"] = (
            np.abs(table.tone_amplitude[rows, 0]) * peaks[table.tone_envelope[rows, 0]]
        )
        keys["duration"] = table.duration_ns[rows]
        keys["complex"] = ~bounds.real[table.tone_envelope[rows, 0]]
        return keys

    def _eligible(
        self,
        table: SegmentTable,
        rows: NDArray[np.int64],
        context: KernelContext,
    ) -> NDArray[np.bool_]:
        """Check density-free bandwidth, motion and split bounds before building."""
        source = context.source
        subdivisions = ramp_subdivisions(table, rows, source, context.work_plan)
        bounded = subdivisions <= 32
        # Strong, unresolved ramps need no keys or remaining eligibility work.
        if not np.any(bounded):
            return bounded
        strength, _ = drive_bounds(table, rows, source)
        keys = self.prepare_keys(table, rows, context)
        frequency = np.abs(keys["frequency"])
        bounds = envelope_bounds(table, context.work_plan)
        peaks, slopes = bounds.peaks, bounds.slopes
        envelope = table.tone_envelope[rows, 0]
        resolved = (
            slopes[envelope]
            <= 2 * np.maximum(peaks[envelope], 1e-30) * frequency * subdivisions
        )
        return (
            (frequency >= 0.03)
            & (keys["bound"] > 0)
            & resolved
            & (keys["bound"] * strength <= 0.5 * frequency * subdivisions)
            & bounded
        )

    def guard(
        self,
        table: SegmentTable,
        rows: NDArray[np.int64],
        context: KernelContext,
    ) -> NDArray[np.bool_]:
        """Return the same model-only convergence mask as guard_prepared."""
        return self.guard_prepared(table, rows, context).accepted

    def _build(
        self,
        keys: NDArray[np.void],
        table: SegmentTable,
        rows: NDArray[np.int64],
        context: KernelContext,
    ) -> tuple[_Data, NDArray[np.float64]]:
        source, budget = context.source, context.budget
        entries, models, ports, errors = [], [], [], []
        planned = ramp_families_for_rows(table, rows, context)
        # Spectral products remain call-local; only the family inventory moved
        # to the planner. Accepted payloads retain immutable coefficient arrays.
        families: dict[RampFamilyPlan, _SpectralFamily] = {}
        for index, key in enumerate(keys):
            model = source.model(tuple(int(v) for v in key["level"]))
            port, f, bound = (
                int(key["port"]),
                float(key["frequency"]),
                float(key["bound"]),
            )
            family_plan = planned[index]
            m = family_plan.subdivisions
            step = 1 / (abs(f) * m)
            # Start with a cubic amplitude polynomial; refine its estimated
            # omitted tail before accepting a key.
            nodes = 4
            cycles = max(1, table.duration_ns[rows[index]] / step)
            modes = 2
            # Compare the limiting amplitude, not 128 carrier-phase ODEs.
            # The bandwidth guard and the retained polynomial's last degree
            # accompany this calibrated (not rigorous) convergence estimate.
            family = families.get(family_plan)
            if family is None:
                family = _SpectralFamily(
                    model, port, f, bound, step, context.work_plan.carrier_workspace()
                )
                families[family_plan] = family
            error = family.error(modes)
            if error * cycles > budget / 4:
                modes = 3
                error = family.error(modes)
            coefficients, interpolation, ratio = family.polynomial((nodes, modes))
            if ratio > 0.25 or interpolation * cycles > budget / 8:
                nodes = 8
                coefficients, interpolation, _ = family.polynomial((nodes, modes))
            errors.append((error + interpolation) * cycles)
            populations = np.asarray(
                expm(step / 2 * population_generator(model.rates_per_ns)), dtype=float
            )
            coherences = np.exp(-step / 2 * coherence_decay(model))
            for array in (coefficients, populations, coherences):
                array.setflags(write=False)
            direct = prepare_models([model])
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
        owned_keys = keys.copy()
        convergence = np.asarray(errors)
        owned_keys.setflags(write=False)
        convergence.setflags(write=False)
        return _Data(
            PeriodPrepared(tuple(entries)),
            tuple(models),
            tuple(ports),
            owned_keys,
            convergence,
        ), convergence

    def admit_prepared(
        self,
        prepared: Prepared,
        table: SegmentTable,
        rows: NDArray[np.int64],
        context: KernelContext,
    ) -> GuardResult | None:
        """Reuse key-covered convergence data and recheck current row eligibility.

        Keys include duration, bound, slope and complex motion class; stored errors
        cover the complete spectral domain. Missing keys return None; eligibility
        rejection returns a decision without rebuilding periods.
        """
        if not isinstance(prepared, _Data):
            raise TypeError("shaped_sambe requires its own prepared payload")
        keys = self.prepare_keys(table, rows, context)
        indices = prepared_key_indices(prepared.keys, keys)
        if indices is None:
            return None
        inverse = np.array([indices[key.tobytes()] for key in keys], dtype=np.int64)
        accepted = self._eligible(table, rows, context)
        accepted &= prepared.errors[inverse] <= context.budget / 2
        return GuardResult(accepted, (GuardPreparation(prepared.keys, prepared),))

    def guard_prepared(
        self,
        table: SegmentTable,
        rows: NDArray[np.int64],
        context: KernelContext,
    ) -> GuardResult:
        """Build eligible keys once, hand accepted model-only periods to Engine."""
        budget = context.budget
        accepted = self._eligible(table, rows, context)
        selected = rows[accepted]
        preparations = []
        if len(selected):
            keys, representatives, inverse = np.unique(
                self.prepare_keys(table, selected, context),
                return_index=True,
                return_inverse=True,
            )
            prepared, errors = self._build(
                keys, table, selected[representatives], context
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
        context: KernelContext,
    ) -> Prepared:
        """Build immutable spectral periods from aligned representative rows."""
        return self._build(keys, table, rows, context)[0]

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
        context: KernelContext,
    ) -> None:
        """Decline input binding; every prepared array is model-only."""
        budget = context.budget
        del prepared, key_index, table, rows, source_initial, budget

    def apply(
        self,
        prepared: Prepared,
        key_index: NDArray[np.int64],
        table: SegmentTable,
        rows: NDArray[np.int64],
        states: NDArray[np.complex128],
        context: ApplyContext,
    ) -> NDArray[np.bool_]:
        """Advance smooth envelopes; fractional steps use frozen spectral tails.

        Incompatible payloads raise ValueError; nonfinite output raises
        RuntimeError. Return an all-False accuracy-exceedance mask.
        """
        context.require_no_plan()
        budget = context.budget
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
                amplitude = envelopes(
                    table, np.array([row]), np.array([[start + tail / 2]])
                )[0, 0]
                theta = (
                    table.tone_theta_rad[row, 0] + 2 * np.pi * entry.frequency * start
                )
                if amplitude.imag == 0:
                    # Keep the signed-real spectral path numerically unchanged.
                    amp = float(amplitude.real)
                else:
                    amp = float(abs(amplitude))
                    theta += float(np.angle(amplitude))
                blocks = _spectral(model, port, entry.frequency, amp, 3, tail)
                harmonics = np.fft.fftfreq(16) * 16
                u = np.einsum("k,kij->ij", np.exp(1j * harmonics * theta), blocks)
                # Polar projection changes U, not density or populations.
                left, _, right = np.linalg.svd(u)
                u = left @ right
                ratio = tail / step
                population = np.asarray(
                    expm(tail / 2 * population_generator(model.rates_per_ns)),
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
