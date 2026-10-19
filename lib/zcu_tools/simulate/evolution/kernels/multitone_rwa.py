"""Calibrated joint-energy-basis RWA for simultaneous smooth or dual constant drives."""

from dataclasses import dataclass
from hashlib import sha256
from typing import NamedTuple

import numpy as np
from numba import njit
from numpy.typing import NDArray

from .._envelope import sample_value
from ..kernel import ApplyContext, KernelContext, Prepared, SegmentTable
from ..secular import SecularModel
from ..workplan import FactKey
from ._dop853 import Tolerances, integrate, tolerances
from ._dop853_data import Models, Segments, pack_models
from ._workload_facts import envelope_bounds, fixed_flux, packed_segments

# Calibrated on 128--512 ns Gaussian simultaneous gates in the 16-level
# joint model: 50 MHz edges + 4 ns ceiling gave <= 1.76e-3 population change
# against lab DOP853 through depth 16. The calibrated half-flux pair has
# summed full-operator/carrier ratio 0.217, sampled normalized slope 0.0192/ns
# and curvature 0.00231/ns² (including DAC quantization). Admission allows
# margins up to 0.25, 0.025/ns and 0.003/ns². These are not rigorous bounds.
_REVISION = "joint-rwa-v3"
_EDGE_GHZ = 0.05
_STEP_NS = 4.0
_CONSTANT_STEP_NS = 16.0
# 25 strict-lab points around +84/-52 MHz: error>0.1 first occurs at
# P=0.0116496. Halve that empirical cutoff, rounded down for safety.
_MAX_COMBINATION_TRANSFER = 5.8e-3
_MIN_BUDGET = 1e-3
_MAX_RATIO = 0.25
_MAX_SLOPE = 0.025
_MAX_CURVATURE = 0.003
# Integer 256 ns gain sweep exceeded 1e-2 at gain 0.4184, proxy 0.063 rad.
_MAX_PHASE_RAD = 0.06
# Constant-only calibration and safety margin are documented on the kernel.
# Keep RWA_POLICY unchanged for sampled rows and covariant gate identities.
_MAX_CONSTANT_PHASE_RAD = 0.16
RWA_POLICY = (
    _REVISION,
    _EDGE_GHZ,
    _STEP_NS,
    _CONSTANT_STEP_NS,
    _MAX_COMBINATION_TRANSFER,
    _MIN_BUDGET,
    _MAX_RATIO,
    _MAX_SLOPE,
    _MAX_CURVATURE,
    _MAX_PHASE_RAD,
)


class RWACoefficients(NamedTuple):
    """Joint RWA coefficients supplied by the single Hamiltonian owner.

    positive/negative: (tones,M,M) GHz/gain Fourier operators, mutual adjoints.
    diagonal: (tones,M) GHz/gain² phase-independent virtual diagonal terms,
        multiplied by abs(gain*envelope)**2. Contains window-excluded single-tone
        second-order shifts; retained edges are not counted twice.
    All consumers, including the gate cache, assemble through
    interaction_hamiltonian. They must not implement a separate drive RHS.
    """

    positive: NDArray[np.complex128]
    negative: NDArray[np.complex128]
    diagonal: NDArray[np.float64]


class _Context(NamedTuple):
    models: Models
    segments: Segments
    positive: NDArray[np.complex128]
    negative: NDArray[np.complex128]
    diagonal: NDArray[np.float64]
    row: int
    model: int


@dataclass(frozen=True)
class RWAPrepared:
    """Model-only RWA payload, charged as one read-only allocation batch.

    models: packed joint energies, drive quadratures, bases and secular rates.
    positive, negative: (keys,tones,M,M) coefficients multiplying z and conj(z),
        respectively, after the 50 MHz Fourier-edge selection.
    diagonal: (keys,max_active_tones,M) GHz/gain² diagonal terms from
        RWACoefficients, independent of carrier phase.
    """

    models: Models
    positive: NDArray[np.complex128]
    negative: NDArray[np.complex128]
    diagonal: NDArray[np.float64]

    @property
    def nbytes(self) -> int:
        return (
            sum(a.nbytes for a in self.models)
            + self.positive.nbytes
            + self.negative.nbytes
            + self.diagonal.nbytes
        )


@njit(nogil=True, cache=True)
def _fourier_coefficient(x, y, sign):
    """Single owner of full Fourier coefficients before the retained window."""
    return (x - sign * 1j * y) / 2


@njit(nogil=True, cache=True)
def _combination_probability(energies, x, y, frequencies, amplitudes, duration):
    """Maximum omitted-path two-carrier transfer proxy, in GHz and ns."""
    maximum = 0.0
    size = len(energies)
    for sign_a in (1, -1):
        for sign_b in (1, -1):
            signs = (sign_a, sign_b)
            operators = np.empty_like(x)
            for tone in range(2):
                operators[tone] = _fourier_coefficient(x[tone], y[tone], signs[tone])
                operators[tone] *= amplitudes[tone]
            combination = sign_a * frequencies[0] + sign_b * frequencies[1]
            for i in range(size):
                for k in range(size):
                    if i == k:
                        continue
                    coupling = 0j
                    for first in range(2):
                        second = 1 - first
                        for m in range(size):
                            d_first = (
                                energies[i]
                                - energies[m]
                                + signs[first] * frequencies[first]
                            )
                            d_second = (
                                energies[m]
                                - energies[k]
                                + signs[second] * frequencies[second]
                            )
                            if abs(d_first) >= _EDGE_GHZ:
                                denominator = d_first
                            elif abs(d_second) >= _EDGE_GHZ:
                                # Reverse the second leg to the same endpoint
                                # convention as the first-leg denominator.
                                denominator = -d_second
                            else:
                                # The retained joint RHS already has this path.
                                continue
                            coupling += (
                                operators[first, i, m]
                                * operators[second, m, k]
                                / denominator
                            )
                    strength = (2 * abs(coupling)) ** 2
                    delta = energies[i] - energies[k] + combination
                    if strength > 0:
                        transfer = min(
                            strength / (strength + delta**2), strength * duration**2
                        )
                        maximum = max(maximum, transfer)
    return maximum


@njit(nogil=True, cache=True)
def _coefficients(models, ports, frequencies, tones):
    """Build retained edges and compact per-tone diagonal from the same window."""
    size = models.energies.shape[1]
    positive = np.zeros((len(tones), ports.shape[1], size, size), np.complex128)
    negative = np.zeros_like(positive)
    max_tones = 0
    for count in tones:
        max_tones = max(max_tones, count)
    diagonal = np.zeros((len(tones), max_tones, size), np.float64)
    for key in range(len(tones)):
        for tone in range(tones[key]):
            port, frequency = ports[key, tone], frequencies[key, tone]
            for i in range(size):
                for j in range(size):
                    gap = models.energies[key, i] - models.energies[key, j]
                    x, y = models.x[key, port, i, j], models.y[key, port, i, j]
                    for sign in (1, -1):
                        coefficient = _fourier_coefficient(x, y, sign)
                        detuning = gap + sign * frequency
                        if abs(detuning) < _EDGE_GHZ:
                            if sign == 1:
                                positive[key, tone, i, j] = coefficient
                            else:
                                negative[key, tone, i, j] = coefficient
                        else:
                            # Exact resonances evolve explicitly above; no
                            # denominator regularization or double counting.
                            diagonal[key, tone, i] += abs(coefficient) ** 2 / detuning
    return positive, negative, diagonal


@njit(nogil=True, cache=True)
def interaction_hamiltonian(time, energies, coefficients, amplitudes):
    """Assemble the joint RWA interaction Hamiltonian, in GHz.

    time is row-local ns. energies is the static interaction-picture diagonal,
    either the joint energies for ordinary RWA or E+q@f for a covariant gate.
    coefficients is RWACoefficients and owns EVERY driven term, including
    phase-independent diagonal terms. amplitudes contains gain*envelope*phase
    for every active tone. Diagonal terms come from the coefficient record;
    the gate cache must not assemble its own RHS or add a separate shift.
    Returns a new Hermitian matrix; input arrays remain unchanged. Restoring
    the static picture at endpoints remains the integrator's responsibility.
    """
    size = len(energies)
    h = np.zeros((size, size), dtype=np.complex128)
    phases = np.exp(2j * np.pi * energies * time)
    for tone in range(len(amplitudes)):
        z = amplitudes[tone]
        for i in range(size):
            for j in range(size):
                h[i, j] += (
                    (
                        coefficients.positive[tone, i, j] * z
                        + coefficients.negative[tone, i, j] * z.conjugate()
                    )
                    * phases[i]
                    * phases[j].conjugate()
                )
            h[i, i] += coefficients.diagonal[tone, i] * abs(z) ** 2
    return h


@njit(nogil=True, cache=True)
def _rhs(time, state, context):
    models, segments, positive, negative, diagonal, row, model = context
    size = models.energies.shape[1]
    rho = state.reshape(size, size)
    amplitudes = np.empty(segments.tones[row], dtype=np.complex128)
    for tone in range(segments.tones[row]):
        env = segments.envelope[row, tone]
        begin, end = segments.sample_offsets[env : env + 2]
        amplitude = sample_value(
            time + segments.offset[row, tone],
            segments.sample_time[begin:end],
            segments.samples[begin:end],
        )
        phase = (
            segments.theta[row, tone] + 2 * np.pi * segments.frequency[row, tone] * time
        )
        amplitudes[tone] = segments.gain[row, tone] * amplitude * np.exp(1j * phase)
    h = interaction_hamiltonian(
        time,
        models.energies[model],
        RWACoefficients(positive[model], negative[model], diagonal[model]),
        amplitudes,
    )
    product = h @ rho
    derivative = np.empty_like(rho)
    for i in range(size):
        for j in range(size):
            derivative[i, j] = (
                -2j * np.pi * (product[i, j] - product[j, i].conjugate())
                - models.decay[model, i, j] * rho[i, j]
            )
        for j in range(size):
            derivative[i, i] += models.rates[model, i, j] * rho[j, j]
    return derivative.ravel()


@njit(nogil=True, cache=True)
def _run(data, row_models, states, controls):
    models, segments, positive, negative, diagonal = data
    rows, indices = row_models
    for index in range(len(rows)):
        row, model = rows[index], indices[index]
        context = _Context(models, segments, positive, negative, diagonal, row, model)
        constant = segments.tones[row] == 2
        for tone in range(segments.tones[row]):
            env = segments.envelope[row, tone]
            constant = (
                constant
                and segments.sample_offsets[env] == segments.sample_offsets[env + 1]
            )
        ceiling = _CONSTANT_STEP_NS if constant else _STEP_NS
        row_controls = Tolerances(
            controls.rtol, controls.atol, min(controls.max_step, ceiling)
        )
        result = integrate(
            _rhs,
            context,
            states[index].copy().ravel(),
            float(segments.duration[row]),
            row_controls,
            0.1,
        ).reshape(states.shape[1:])
        phases = np.exp(-2j * np.pi * models.energies[model] * segments.duration[row])
        for i in range(len(phases)):
            for j in range(len(phases)):
                states[index, i, j] = result[i, j] * phases[i] * phases[j].conjugate()


def _smooth(table: SegmentTable, envelope: int, context: KernelContext) -> bool:
    spec = table.envelopes[envelope]

    def build() -> bool:
        if spec.kind == "constant":
            return True
        if len(spec.samples) < 3 or np.any(spec.samples.imag != 0):
            return False
        peak = float(np.max(np.abs(spec.samples)))
        if peak == 0:
            return True
        spacing = np.diff(spec.sample_ns)
        slopes = np.diff(spec.samples.real) / spacing
        curvature = np.diff(slopes) / ((spacing[1:] + spacing[:-1]) / 2)
        return bool(
            np.max(np.abs(slopes)) <= _MAX_SLOPE * peak
            and np.max(np.abs(curvature)) <= _MAX_CURVATURE * peak
        )

    return context.work_plan.fact(
        FactKey(("multitone-rwa-smooth", RWA_POLICY, spec.identity), bool), build
    )


def _retained_norm(
    model: SecularModel, port: int, frequency: float, context: KernelContext
) -> float:
    """Cache the near-resonant positive Fourier operator spectral norm."""
    key = FactKey(
        (
            "multitone-rwa-retained-norm",
            RWA_POLICY,
            model.identity,
            model.level,
            port,
            frequency,
        ),
        float,
    )

    def build() -> float:
        return float(np.linalg.norm(retained_edges(model, port, frequency), 2))

    return context.work_plan.fact(key, build)


def admit_rows(
    table: SegmentTable, rows: NDArray[np.int64], context: KernelContext
) -> NDArray[np.bool_]:
    """Admit simultaneous rows under the shared calibrated joint RWA policy.

    Requires at least two tones. Reads model-only facts, never density;
    splitting rows preserves decisions. Invalid budgets raise ValueError.
    """
    return _admit_rows(table, rows, context, single_tone=False)


def admit_single_rows(
    table: SegmentTable, rows: NDArray[np.int64], context: KernelContext
) -> NDArray[np.bool_]:
    """Admit exactly one shaped tone for covariant gates, not row RWA routing.

    Uses the same smoothness, fixed-flux, strength, phase-proxy and budget
    policy as simultaneous RWA without relaxing its public multi-tone domain.
    """
    return _admit_rows(table, rows, context, single_tone=True)


def _admit_rows(
    table: SegmentTable,
    rows: NDArray[np.int64],
    context: KernelContext,
    *,
    single_tone: bool,
) -> NDArray[np.bool_]:
    tolerances(context.budget)
    bounds = envelope_bounds(table, context.work_plan)
    accepted = np.zeros(len(rows), dtype=np.bool_)
    for index, row in enumerate(rows):
        count = table.tones[row]
        if (
            context.budget < _MIN_BUDGET
            or (count != 1 if single_tone else count < 2)
            or not fixed_flux(table, int(row))
        ):
            continue
        envs = table.tone_envelope[row, :count]
        constant = all(table.envelopes[e].kind == "constant" for e in envs)
        # Same-port Stark/probe spectra retain their Floquet/direct route:
        # small population errors can still shift a narrow resonance peak.
        if constant and (
            count != 2 or table.tone_port[row, 0] == table.tone_port[row, 1]
        ):
            continue
        if not all(_smooth(table, int(e), context) for e in envs):
            continue
        frequencies = np.abs(table.tone_frequency_ghz[row, :count])
        if np.min(frequencies) < _EDGE_GHZ or np.any(
            np.diff(np.sort(frequencies)) < 2 * _EDGE_GHZ
        ):
            continue
        model = context.source.model(tuple(int(v) for v in table.level[row]))
        if constant:
            ports = table.tone_port[row, :count]
            probability = _combination_probability(
                model.energies_ghz,
                np.array([model.drives[p].in_phase_ghz for p in ports]),
                np.array([model.drives[p].quadrature_ghz for p in ports]),
                table.tone_frequency_ghz[row, :count],
                np.abs(table.tone_amplitude[row, :count]) * bounds.peaks[envs],
                table.duration_ns[row],
            )
            if probability > _MAX_COMBINATION_TRANSFER:
                continue
        ratio = 0.0
        phase = 0.0
        for tone, env in enumerate(envs):
            port = int(table.tone_port[row, tone])
            drive = model.drives[port]
            retained = _retained_norm(
                model, port, float(table.tone_frequency_ghz[row, tone]), context
            )
            phase += (
                table.tone_amplitude[row, tone] * bounds.peaks[env] * retained
            ) ** 2 / frequencies[tone]
            ratio += (
                abs(table.tone_amplitude[row, tone])
                * bounds.peaks[env]
                * (
                    np.linalg.norm(drive.in_phase_ghz, 2)
                    + np.linalg.norm(drive.quadrature_ghz, 2)
                )
                / frequencies[tone]
            )
        phase_limit = _MAX_CONSTANT_PHASE_RAD if constant else _MAX_PHASE_RAD
        accepted[index] = (
            ratio <= _MAX_RATIO
            and 2 * np.pi * table.duration_ns[row] * phase <= phase_limit
        )
    return accepted


def retained_edges(
    model: SecularModel, port: int, frequency: float
) -> NDArray[np.complex128]:
    """Return the positive-carrier joint operator in GHz/gain.

    The negative-carrier operator is its adjoint. Both RWA integrators and
    covariant gate construction use this same 50 MHz retained-edge policy.
    The caller owns the returned mutable matrix.
    """
    drive = model.drives[port]
    gaps = model.energies_ghz[:, None] - model.energies_ghz[None, :]
    coupling = _fourier_coefficient(drive.in_phase_ghz, drive.quadrature_ghz, 1)
    return coupling * (np.abs(gaps + frequency) < _EDGE_GHZ)


def rwa_coefficients(
    model: SecularModel, ports: tuple[int, ...], frequencies: tuple[float, ...]
) -> RWACoefficients:
    """Build all joint drive coefficients for ordered signed carriers.

    This is the shared coefficient owner for ordinary RWA and covariant gates.
    Each tone contributes sum_j,s |C_s[i,j]|²/(Ei-Ej+s*f) to the diagonal
    in GHz/gain², restricted to abs(Ei-Ej+s*f)>=0.05 GHz. Retained edges
    evolve explicitly, with no zero division or double counting. The common
    compiled builder also serves batched ordinary RWA preparation.
    Graph pruning may replace the Fourier operators but retains the diagonal.
    Returned arrays are owned and mutable. Misaligned/empty carriers raise
    ValueError; ports must index the supplied model's validated drives.
    """
    if not ports or len(ports) != len(frequencies):
        raise ValueError("RWA coefficients require aligned nonempty carriers")
    positive, negative, diagonal = _coefficients(
        pack_models([model]),
        np.array([ports], dtype=np.int64),
        np.array([frequencies], dtype=np.float64),
        np.array([len(ports)], dtype=np.int64),
    )
    return RWACoefficients(positive[0], negative[0], diagonal[0])


class MultitoneRWAKernel:
    """Approximate C5 family, revision joint-rwa-v3; never a DOP853 frame flag.

    Admission requires >=2 tones with a sampled envelope, or exactly two
    constant envelopes on distinct ports, fixed flux,
    real envelopes with peak-normalized slope <=0.025/ns and curvature
    <=0.003/ns², each |f|>=0.05 GHz, pairwise ||f_i|-|f_j||>=0.1 GHz,
    sum of peak |gain|*(||X||₂+||Y||₂)/|f| <=0.25, and cumulative
    counter-rotating phase proxy 2π*duration*sum((peak*gain*||V_ret||₂)²/|f|)
    <=0.06 rad for sampled rows, or <=0.16 rad for distinct-port dual constant
    rows. The constant cap uses one 96-point scan of the 16-level joint model
    against strict lab DOP853 (budget 1e-7): flux offsets
    (0.48*f,0.2,0.49*f) Phi0 for f=0,0.4,0.8,1 between integer and near-half;
    phases (0,0) and (0.7,-1.2) rad; full-operator/carrier ratios 0.03, 0.07,
    0.14, 0.24; durations 128, 512, 1024 ns. Among all forced-RWA points,
    including combination-guard rejects, the smallest proxy with population
    error >0.1 was 0.338362529 rad. Halving and rounding down gives 0.16 rad
    (2.11x margin). All 44 points below this cap had error <=0.08641; the 36
    also passing the combination guard had error <=0.005062. This empirical
    cap is not an arbitrary-input or whole-sequence certificate. Sampled
    admission and covariant gate policies remain unchanged. Budgets below
    1e-3 decline: this calibrated approximation is not a tight-error solver.
    Dual constant rows also require an omitted-path combination transfer proxy
    <=5.8e-3 for every signed sum/difference and dressed level pair.
    Calibration uses a 5x5 grid, each carrier within 10 MHz of the integer
    +84/-52 MHz counterexample, against lab budget 1e-7. Among points with
    population error >0.1, the smallest P was 0.0116496. The threshold is
    half that value, rounded down. Original mux scans lose no extra rows. Full dressed
    gaps include conditional ZZ. For each two-step path, both tone orders use
    the shared Fourier coefficient builder with abs(gain)*envelope peak. Paths
    with both edges retained already evolve in the joint RHS and are excluded.
    Otherwise divide the coefficient product by the first omitted detuning
    (or the reversed second detuning when only that edge is omitted), and sum
    over intermediates and tone orders to obtain g2 in GHz. With dressed
    detuning delta, P=min((2*abs(g2))²/((2*abs(g2))²+delta²),
    (2*abs(g2)*duration_ns)²). The finite-time cap avoids spurious rejection
    at zero detuning for vanishing coupling. An empty retained graph is NOT
    evidence of safety. This calibrated estimate is not an error certificate.
    Constant rows use a 16 ns adaptive ceiling; sampled rows retain 4 ns.
    No density reduction, dissipator change, clipping or normalization occurs.
    """

    name = "multitone_rwa"
    revision = _REVISION
    guard_builds = False

    def guard(
        self, table: SegmentTable, rows: NDArray[np.int64], context: KernelContext
    ) -> NDArray[np.bool_]:
        """Return row-local admission under the explicit calibrated policy above."""
        return admit_rows(table, rows, context)

    def prepare_keys(
        self, table: SegmentTable, rows: NDArray[np.int64], context: KernelContext
    ) -> NDArray[np.void]:
        """Hash policy, budget, model level and complete active tone/envelope content."""
        keys = np.empty(len(rows), dtype="V32")
        for index, row in enumerate(rows):
            count = table.tones[row]
            digest = sha256(repr((RWA_POLICY, context.budget)).encode("ascii"))
            if all(
                table.envelopes[e].kind == "constant"
                for e in table.tone_envelope[row, :count]
            ):
                digest.update(
                    repr(("constant_phase_rad", _MAX_CONSTANT_PHASE_RAD)).encode(
                        "ascii"
                    )
                )
            digest.update(table.level[row].astype("<i8").tobytes())
            digest.update(
                np.array(
                    [table.start_ns[row], table.duration_ns[row]], dtype="<f8"
                ).tobytes()
            )
            for values in (
                table.tone_port,
                table.tone_frequency_ghz,
                table.tone_amplitude,
                table.tone_theta_rad,
                table.tone_envelope_offset_ns,
            ):
                digest.update(values[row, :count].astype("<f8").tobytes())
            for env in table.tone_envelope[row, :count]:
                digest.update(table.envelopes[env].identity.encode("ascii"))
            keys[index] = np.void(digest.digest())
        return keys

    def cost(self, table: SegmentTable, rows: NDArray[np.int64]) -> NDArray[np.float64]:
        """Estimate slow-envelope RHS work, without runtime timing probes."""
        return np.asarray(
            (1 + table.tones[rows])
            * np.maximum(1.0, table.duration_ns[rows] / _STEP_NS),
            dtype=np.float64,
        )

    def prepare(
        self,
        keys: NDArray[np.void],
        table: SegmentTable,
        rows: NDArray[np.int64],
        context: KernelContext,
    ) -> Prepared:
        """Build retained Fourier edges and omitted-edge shifts for key-aligned rows.

        Uses the same compiled coefficient owner as rwa_coefficients. Apply
        scales its diagonal by the actual squared gain/envelope amplitude.
        """
        tolerances(context.budget)
        if len(keys) != len(rows):
            raise ValueError("RWA keys and representative rows must align")
        models = pack_models(
            [
                context.source.model(tuple(int(v) for v in table.level[row]))
                for row in rows
            ]
        )
        positive, negative, diagonal = _coefficients(
            models,
            table.tone_port[rows],
            table.tone_frequency_ghz[rows],
            table.tones[rows],
        )
        positive.setflags(write=False)
        negative.setflags(write=False)
        diagonal.setflags(write=False)
        return RWAPrepared(models, positive, negative, diagonal)

    def admit_prepared(
        self,
        prepared: Prepared,
        table: SegmentTable,
        rows: NDArray[np.int64],
        context: KernelContext,
    ) -> None:
        """Decline guard handoffs; admission is cheap and performs no preparation."""
        del prepared, table, rows, context

    def guard_prepared(
        self, table: SegmentTable, rows: NDArray[np.int64], context: KernelContext
    ) -> None:
        """Use the ordinary row-local guard, without building payloads."""
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
        """Retain arbitrary joint input densities; no input-bound shortcut is used."""
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
        """Advance independent rows in place and return an all-False exceedance mask.

        Every row starts in its joint eigenbasis, integrates all retained tones
        and envelope-squared omitted-edge shifts through interaction_hamiltonian
        in a row-local interaction picture, then restores Schrödinger coordinates.
        theta is the phase at row start; offsets preserve pulse-local envelopes.
        Adaptive integration uses the ordinary budget tolerance and 0.1 ns seed,
        capped at 16 ns for dual constant envelopes and 4 ns otherwise,
        rather than sampled knots. This is not a certified bound
        on omitted edges or accumulated RB fidelity. Engine owns inter-row basis
        changes and chronology. Invalid payload, plan or shapes raise ValueError;
        numerical integration failures propagate without a retry.
        """
        if context.plan is not None or not isinstance(prepared, RWAPrepared):
            raise ValueError("RWA requires its own prepared data and no row plan")
        size = prepared.models.energies.shape[1]
        if key_index.shape != rows.shape or states.shape != (len(rows), size, size):
            raise ValueError("RWA state/index shape mismatch")
        if np.any(key_index < 0) or np.any(key_index >= len(prepared.models.ports)):
            raise ValueError("RWA prepared index out of range")
        controls = tolerances(context.budget)
        _run(
            (
                prepared.models,
                packed_segments(table, context.work_plan),
                prepared.positive,
                prepared.negative,
                prepared.diagonal,
            ),
            (rows, key_index),
            states,
            controls,
        )
        return np.zeros(len(rows), dtype=np.bool_)
