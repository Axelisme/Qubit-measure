"""Population-preserving adiabatic transport along continuous flux samples."""

from dataclasses import dataclass
from hashlib import sha256

import numpy as np
from numpy.typing import NDArray
from scipy.special import roots_legendre

from ..kernel import ApplyContext, KernelContext, Prepared, SegmentTable
from ..secular import FluxExpansion, ModelSource
from ._secular import coherence_decay


def _waveform(table: SegmentTable, row: int):
    waveform = table.flux_waveform
    if waveform is None:
        raise ValueError("adiabatic transport requires a flux waveform")
    point = table.timeline[row]
    if (
        table.flux_waveform_available is not None
        and not table.flux_waveform_available[point]
    ):
        raise ValueError("adiabatic transport requires samples for this timeline")
    begin, end = waveform.offsets[point : point + 2]
    return waveform.time_ns[begin:end], waveform.flux[begin:end]


def _hamiltonian(expansion: FluxExpansion, times, flux, time: float):
    h = expansion.static.copy()
    for term in expansion.terms:
        if term.loop >= flux.shape[1]:
            raise ValueError("flux expansion loop exceeds waveform width")
        value = np.interp(time, times, flux[:, term.loop])
        angle = np.pi * value * (2 if term.kind.endswith("2pi") else 1)
        coefficient = np.cos(angle) if term.kind.startswith("cos") else np.sin(angle)
        h += coefficient * term.matrix
    return h


def row_basis(
    table: SegmentTable, row: int, source: ModelSource, *, end: bool
) -> NDArray[np.complex128]:
    """Return canonical instantaneous eigenvectors at a continuous row boundary.

    table/row identify one global row with available absolute-Phi0 samples.
    source supplies the exact fixed-coordinate FluxExpansion. end=False selects
    start_ns; True selects start_ns+duration_ns. Samples interpolate linearly
    and hold their first/last values outside support. Returned unitary columns
    are ascending-energy numpy eigh vectors in fixed coordinates. The kernel
    uses this same gauge for its input/output densities; the scheduler converts
    only at the outer boundaries, not through intervening quantized baths.
    No input is mutated. Missing samples/expansion or bad loop widths raise
    ValueError; eigensolver failures propagate.
    """
    times, flux = _waveform(table, row)
    time = float(table.start_ns[row] + (table.duration_ns[row] if end else 0))
    return np.linalg.eigh(_hamiltonian(source.flux_expansion(), times, flux, time))[1]


def _transport(expansion, times, flux, knots, order):
    _, previous = np.linalg.eigh(_hamiltonian(expansion, times, flux, knots[0]))
    vectors = previous
    resolved = True
    for left, right in zip(knots[:-1], knots[1:], strict=True):
        for time in np.linspace(left, right, order + 1)[1:]:
            _, vectors = np.linalg.eigh(_hamiltonian(expansion, times, flux, time))
            overlap = np.sum(previous.conj() * vectors, axis=0)
            magnitude = np.abs(overlap)
            if np.any(magnitude < 0.25):
                resolved = False
            # A vanishing overlap requires refinement, never a fabricated phase.
            if np.any(magnitude == 0):
                return np.ones(previous.shape[1], dtype=complex), False
            previous = vectors * (overlap.conj() / magnitude)
    # Compare the transported endpoint with its canonical model eigenbasis.
    return np.sum(vectors.conj() * previous, axis=0), resolved


def _integrate(expansion, times, flux, start, stop, order):
    nodes, weights = roots_legendre(order)
    knots = np.r_[start, times[(times > start) & (times < stop)], stop]
    integral = np.zeros(len(expansion.static))
    for left, right in zip(knots[:-1], knots[1:], strict=True):
        half = (right - left) / 2
        sample_times = (right + left) / 2 + half * nodes
        for time, weight in zip(sample_times, weights, strict=True):
            energy, _ = np.linalg.eigh(_hamiltonian(expansion, times, flux, time))
            integral += half * weight * energy
    phase, resolved = _transport(expansion, times, flux, knots, order)
    return integral, phase, resolved


def _factor(table: SegmentTable, row: int, context: KernelContext):
    source = context.source
    times, flux = _waveform(table, row)
    level = tuple(int(v) for v in table.level[row])
    bath = source.shifted_model(level, table.shift(row))
    shift = bath.energies_ghz - source.model(level).energies_ghz
    duration = table.duration_ns[row]
    start = table.start_ns[row]
    expansion = source.flux_expansion()
    previous = None
    previous_phase = None
    for order in (4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096):
        integral, phase, resolved = _integrate(
            expansion, times, flux, start, start + duration, order
        )
        integral += shift * duration
        corrected = phase
        if resolved and previous_phase is not None:
            # Uniform overlap transport has second-order phase error. Richardson
            # cancellation avoids thousands of nodes for a complex Berry phase.
            corrected = phase * np.exp(1j * np.angle(phase * previous_phase.conj()) / 3)
        value = np.exp(-2j * np.pi * (integral[:, None] - integral[None, :]))
        value *= corrected[:, None] * corrected.conj()[None, :]
        if (
            resolved
            and previous is not None
            and np.max(np.abs(value - previous)) <= context.budget * 0.1
        ):
            value *= np.exp(-duration * coherence_decay(bath))
            np.fill_diagonal(value, 1)
            return value
        previous = value if resolved and previous_phase is not None else None
        previous_phase = phase if resolved else None
    raise RuntimeError(
        "adiabatic phase/gauge quadrature did not converge at 4096 nodes per knot interval"
    )


@dataclass(frozen=True)
class _Prepared:
    factors: NDArray[np.complex128]

    @property
    def nbytes(self) -> int:
        return self.factors.nbytes


class AdiabaticMapKernel:
    """No-tone adiabatic map in the continuous endpoint eigenbases.

    Input densities use row_basis(end=False); output uses row_basis(end=True).
    Ordered populations stay unchanged, including when the row bath has rates.
    Coherence mn acquires exp(-2j*pi*integral(f_m-f_n)dt-Gamma_mn(level)*dt).
    Energies/vectors come from the exact FluxExpansion at interpolated waveform
    values. Gamma includes the quantized row bath's relaxation and dephasing;
    registered SegmentShift energies/dephasing remain additive and constant.
    No interlevel transitions or nonadiabatic corrections are modeled. Callers
    own the slow-motion, noncrossing, nondegenerate-spectrum applicability.

    Gauss-Legendre quadrature splits at every waveform knot inside the row.
    Orders 4,8,...,4096 per interval double until successive complex phase
    factors (including geometric phase) differ by <= budget/10 in max norm.
    At each order, a separate uniform grid of that many intervals per knot
    transports vector phases by ordered overlaps. Richardson extrapolation of
    successive gauge phases cancels the uniform grid's second-order error.
    Overlaps below 0.25 force refinement. The final transported/canonical
    endpoint phase difference acts on both density indices. This retains Berry
    phase and is independent of intermediate eigensolver sign/phase choices.
    This convergence estimate is not an adiabatic-error certificate. Failure
    to converge raises RuntimeError; absent samples/expansion raises ValueError.
    Preparation never reads density. All owned prepared arrays are immutable.
    name is the stable dispatch/cache family identifier "adiabatic_map".
    guard_builds=False selects ordinary row-local guards without handoff builds.
    """

    name = "adiabatic_map"
    guard_builds = False

    def guard(
        self, table: SegmentTable, rows: NDArray[np.int64], context: KernelContext
    ) -> NDArray[np.bool_]:
        """Accept zero-tone rows with samples; dispatch owns motion detection.

        table/rows are global row indices; context owns source and positive
        budget. Return a row-aligned bool mask, without evolving any density.
        """
        del context
        available = np.full(len(rows), table.flux_waveform is not None)
        if table.flux_waveform_available is not None:
            available &= table.flux_waveform_available[table.timeline[rows]]
        return (table.tones[rows] == 0) & available

    def prepare_keys(
        self, table: SegmentTable, rows: NDArray[np.int64], context: KernelContext
    ) -> NDArray[np.void]:
        """Hash row waveform, time window, bath level and shift into fixed keys.

        Return one fixed-width key per global row. Engine namespaces by source
        and budget. Absolute waveform content/time enter identity; row/timeline
        indices do not. No input is changed or density inspected.
        """
        del context
        keys = np.empty(len(rows), dtype="V32")
        for index, row in enumerate(rows):
            times, flux = _waveform(table, int(row))
            digest = sha256()
            for value in (
                times,
                flux,
                table.level[row],
                np.array([table.start_ns[row], table.duration_ns[row]]),
            ):
                digest.update(value.tobytes())
            digest.update(table.shift(int(row)))
            keys[index] = digest.digest()
        return keys

    def cost(self, table: SegmentTable, rows: NDArray[np.int64]) -> NDArray[np.float64]:
        """Return positive row work estimates, aligned with global rows."""
        del table
        return np.full(len(rows), 16.0, dtype=np.float64)

    def prepare(
        self,
        keys: NDArray[np.void],
        table: SegmentTable,
        rows: NDArray[np.int64],
        context: KernelContext,
    ) -> Prepared:
        """Build immutable coherence factors for unique keys/representative rows.

        keys and rows are aligned; context supplies source and budget. No input
        changes. Return model-only owned storage, or propagate numerical failure
        and malformed waveform/expansion errors described by the class contract.
        """
        del keys
        factors = np.asarray(
            [_factor(table, int(row), context) for row in rows], dtype=np.complex128
        )
        factors.setflags(write=False)
        return _Prepared(factors)

    def admit_prepared(
        self,
        prepared: Prepared,
        table: SegmentTable,
        rows: NDArray[np.int64],
        context: KernelContext,
    ) -> None:
        """Decline cached-domain admission; ordinary row-local guard is cheap."""
        del prepared, table, rows, context

    def guard_prepared(
        self, table: SegmentTable, rows: NDArray[np.int64], context: KernelContext
    ) -> None:
        """Decline preparation handoff; guard_builds=False selects guard instead."""
        del table, rows, context

    def bind_input(
        self,
        prepared: Prepared,
        key_index: int,
        table: SegmentTable,
        rows: NDArray[np.int64],
        source_initial: NDArray[np.complex128],
        context: KernelContext,
    ) -> None:
        """Decline input contraction; model-only factors apply to any density."""
        del prepared, key_index, table, rows, source_initial, context

    def apply(
        self,
        prepared: Prepared,
        key_index: NDArray[np.int64],
        table: SegmentTable,
        rows: NDArray[np.int64],
        states: NDArray[np.complex128],
        context: ApplyContext,
    ) -> NDArray[np.bool_]:
        """Advance row-aligned endpoint-basis states in place; no plan allowed.

        key_index selects immutable prepared factors for each state. table/rows
        identify the same global rows used to prepare. Return all-False accuracy
        flags; no nonadiabatic certificate is available. Wrong payload or a
        non-None plan raises ValueError. All inputs except states stay unchanged.
        """
        context.require_no_plan()
        del table
        if not isinstance(prepared, _Prepared):
            raise ValueError("adiabatic_map requires its own prepared payload")
        states *= prepared.factors[key_index]
        return np.zeros(len(rows), dtype=np.bool_)
