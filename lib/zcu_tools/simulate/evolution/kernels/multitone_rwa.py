"""Calibrated joint-energy-basis RWA for simultaneous smooth shaped drives."""

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
from ._workload_facts import envelope_bounds, packed_segments

# Calibrated on 128--512 ns Gaussian simultaneous gates in the 16-level
# joint model: 50 MHz edges + 4 ns ceiling gave <= 1.76e-3 population change
# against lab DOP853 through depth 16. The calibrated half-flux pair has
# summed full-operator/carrier ratio 0.217, sampled normalized slope 0.0192/ns
# and curvature 0.00231/ns² (including DAC quantization). Admission allows
# margins up to 0.25, 0.025/ns and 0.003/ns². These are not rigorous bounds.
_REVISION = "joint-rwa-v2"
_EDGE_GHZ = 0.05
_STEP_NS = 4.0
_MIN_BUDGET = 1e-3
_MAX_RATIO = 0.25
_MAX_SLOPE = 0.025
_MAX_CURVATURE = 0.003
# Integer 256 ns gain sweep exceeded 1e-2 at gain 0.4184, proxy 0.063 rad.
_MAX_PHASE_RAD = 0.06
RWA_POLICY = (
    _REVISION,
    _EDGE_GHZ,
    _STEP_NS,
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
                        coefficient = (x - sign * 1j * y) / 2
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
        result = integrate(
            _rhs,
            context,
            states[index].copy().ravel(),
            float(segments.duration[row]),
            controls,
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


def _fixed_flux(table: SegmentTable, row: int) -> bool:
    waveform = table.flux_waveform
    point = table.timeline[row]
    if waveform is None or (
        table.flux_waveform_available is not None
        and not table.flux_waveform_available[point]
    ):
        return True
    first, last = waveform.offsets[point : point + 2]
    times = waveform.time_ns[first:last]
    flux = waveform.flux[first:last]
    start = table.start_ns[row]
    end = start + table.duration_ns[row]
    active = np.r_[start, times[(times > start) & (times < end)], end]
    return all(
        np.ptp(np.interp(active, times, flux[:, loop])) == 0
        for loop in range(flux.shape[1])
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
    """Admit joint RWA rows under the single shared calibrated policy.

    Reads model-only facts, never density; splitting rows preserves decisions.
    Invalid budgets raise ValueError through the common solver controls.
    """
    tolerances(context.budget)
    bounds = envelope_bounds(table, context.work_plan)
    accepted = np.zeros(len(rows), dtype=np.bool_)
    for index, row in enumerate(rows):
        count = table.tones[row]
        if (
            context.budget < _MIN_BUDGET
            or count < 2
            or not _fixed_flux(table, int(row))
        ):
            continue
        envs = table.tone_envelope[row, :count]
        if all(table.envelopes[e].kind == "constant" for e in envs) or not all(
            _smooth(table, int(e), context) for e in envs
        ):
            continue
        frequencies = np.abs(table.tone_frequency_ghz[row, :count])
        if np.min(frequencies) < _EDGE_GHZ or np.any(
            np.diff(np.sort(frequencies)) < 2 * _EDGE_GHZ
        ):
            continue
        model = context.source.model(tuple(int(v) for v in table.level[row]))
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
        accepted[index] = (
            ratio <= _MAX_RATIO
            and 2 * np.pi * table.duration_ns[row] * phase <= _MAX_PHASE_RAD
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
    coupling = (drive.in_phase_ghz - 1j * drive.quadrature_ghz) / 2
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
    """Approximate C5 family, revision joint-rwa-v2; never a DOP853 frame flag.

    Admission requires >=2 tones, at least one sampled envelope, fixed flux,
    real envelopes with peak-normalized slope <=0.025/ns and curvature
    <=0.003/ns², each |f|>=0.05 GHz, pairwise ||f_i|-|f_j||>=0.1 GHz,
    sum of peak |gain|*(||X||₂+||Y||₂)/|f| <=0.25, and cumulative
    counter-rotating phase proxy 2π*duration*sum((peak*gain*||V_ret||₂)²/|f|)
    <=0.06 rad. Budgets below
    1e-3 decline: this calibrated approximation is not a tight-error solver.
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
        capped at 4 ns rather than sampled knots. This is not a certified bound
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
            Tolerances(controls.rtol, controls.atol, min(controls.max_step, _STEP_NS)),
        )
        return np.zeros(len(rows), dtype=np.bool_)
