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
from ..workplan import FactKey, WorkPlan
from ._dop853 import Tolerances, integrate, tolerances
from ._dop853_data import Models, Segments, pack_models
from ._workload_facts import (
    envelope_bounds,
    fixed_flux,
    packed_model,
    packed_segments,
    quadrature_norms,
)

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
# Half the strict-lab failing combination-channel boundary (0.0116496).
# The extended 96-point single-constant scan had no failures below 0.045813;
# the same-tone counterexample is 0.015629. Keep the stricter shared cap.
_MAX_OMISSION_TRANSFER = 5.8e-3
_MIN_BUDGET = 1e-3
_MAX_RATIO = 0.25
_MAX_SLOPE = 0.025
_MAX_CURVATURE = 0.003
# Integer 256 ns gain sweep exceeded 1e-2 at gain 0.4184, proxy 0.063 rad.
_MAX_PHASE_RAD = 0.06
# Distinct-port dual constants: strict-lab calibration, half the 0.338 rad boundary.
_MAX_CONSTANT_PHASE_RAD = 0.16
CONSTANT_RWA_POLICY = (
    "constant-omission-v1",
    _MAX_OMISSION_TRANSFER,
    _MAX_CONSTANT_PHASE_RAD,
)
RWA_POLICY = (
    _REVISION,
    _EDGE_GHZ,
    _STEP_NS,
    _CONSTANT_STEP_NS,
    _MAX_OMISSION_TRANSFER,
    _MIN_BUDGET,
    _MAX_RATIO,
    _MAX_SLOPE,
    _MAX_CURVATURE,
    _MAX_PHASE_RAD,
)


ADAPTIVE_RWA_POLICY = ("omission-retained-edges-v1", CONSTANT_RWA_POLICY, RWA_POLICY)


@dataclass(frozen=True)
class RWARetention:
    """Resolved additions to the initial window and remaining omission estimate.

    edges has one ordered tuple of added (i,j) coordinates per positive carrier;
    the final set is the 50 MHz window union these edges. Negative Fourier masks
    are the transposes. omission_transfer is the maximum
    remaining first/second-order transfer proxy, not a rigorous error bound.
    """

    edges: tuple[tuple[tuple[int, int], ...], ...]
    omission_transfer: float


@dataclass(frozen=True)
class ConstantRetentions:
    """Single-constant retained sets aligned with absolute SegmentTable rows.

    rows contains None outside the fixed-flux single constant carrier domain.
    The complete immutable inventory avoids repeated per-key fact lookups.
    """

    rows: tuple[RWARetention | None, ...]


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
def _transfer_probability(coupling, detuning, duration):
    strength = (2 * abs(coupling)) ** 2
    if strength == 0:
        return 0.0
    return min(strength / (strength + detuning**2), strength * duration**2)


@njit(nogil=True, cache=True)
def _two_step_coupling(data, first, second, i, k):
    # Sum intermediates for ONE ordered pair of signed Fourier carriers.
    energies, operators, frequencies, retained = data
    a, sa = first
    b, sb = second
    sign_a, sign_b = 1 - 2 * sa, 1 - 2 * sb
    coupling = 0j
    for m in range(len(energies)):
        d_first = energies[i] - energies[m] + sign_a * frequencies[a]
        d_second = energies[m] - energies[k] + sign_b * frequencies[b]
        if not retained[a, sa, i, m]:
            denominator = d_first
        elif not retained[b, sb, m, k]:
            denominator = -d_second
        else:
            continue
        coupling += operators[a, sa, i, m] * operators[b, sb, m, k] / denominator
    return coupling


@njit(nogil=True, cache=True)
def _pair_probability(data, first, second, duration, additions):
    energies, _, frequencies, _ = data
    a, sa = first
    b, sb = second
    combination = (1 - 2 * sa) * frequencies[a] + (1 - 2 * sb) * frequencies[b]
    maximum = 0.0
    for i in range(len(energies)):
        for k in range(len(energies)):
            if i == k and a == b and sa != sb:
                continue  # The static single-tone diagonal already retains it.
            coupling = _two_step_coupling(data, first, second, i, k)
            if first != second:
                coupling += _two_step_coupling(data, second, first, i, k)
            probability = _transfer_probability(
                coupling, energies[i] - energies[k] + combination, duration
            )
            maximum = max(maximum, probability)
            if probability > _MAX_OMISSION_TRANSFER:
                _restore_paths(data, first, second, i, k, additions)
                if first != second:
                    _restore_paths(data, second, first, i, k, additions)
    return maximum


@njit(nogil=True, cache=True)
def _omission_probability(energies, x, y, frequencies, amplitudes, duration):
    retained = _window_mask(energies, frequencies)
    probability, _ = _omission_analysis(
        energies, (x, y), frequencies, amplitudes, duration, retained
    )
    return probability


@njit(nogil=True, cache=True)
def _window_mask(energies, frequencies):
    size = len(energies)
    retained = np.empty((len(frequencies), 2, size, size), dtype=np.bool_)
    for a in range(len(frequencies)):
        for sa in range(2):
            for i in range(size):
                for j in range(size):
                    retained[a, sa, i, j] = (
                        abs(energies[i] - energies[j] + (1 - 2 * sa) * frequencies[a])
                        < _EDGE_GHZ
                    )
    return retained


@njit(nogil=True, cache=True)
def _restore_paths(data, first, second, i, k, additions):
    # Restore only nonzero paths contributing to this effective channel.
    energies, operators, _, retained = data
    a, sa = first
    b, sb = second
    for m in range(len(energies)):
        if operators[a, sa, i, m] == 0 or operators[b, sb, m, k] == 0:
            continue
        if not retained[a, sa, i, m]:
            additions[a, sa, i, m] = True
            additions[a, 1 - sa, m, i] = True
        if not retained[b, sb, m, k]:
            additions[b, sb, m, k] = True
            additions[b, 1 - sb, k, m] = True


@njit(nogil=True, cache=True)
def _omission_analysis(
    energies, quadratures, frequencies, amplitudes, duration, retained
):
    """One rule for every removed first/second-order Fourier coupling.

    C_s=(X-s*iY)/2, d=Ei-Ej+s*f. Removed one-photon edges use
    g=C_s*amplitude. Two-step paths use g=sum(C_a*C_b/d_removed)
    over intermediates and distinct time orders; paths with both edges retained
    are excluded. A same-tone/same-sign path has one order, NOT a duplicated
    carrier. Mixed signs and distinct carriers include both orders. Full dressed
    gaps include conditional ZZ. Only zero-frequency same-tone diagonal
    terms are excluded: they already live in RWACoefficients.diagonal.
    Oscillating diagonal terms remain charged like every other omission.
    Each effective edge uses P=min(4|g|²/(4|g|²+delta²),4|g|²*T²).
    GHz/ns units and the finite-time cap follow the calibrated combination
    proxy, not a rigorous Dyson error bound (no claim of a 2*pi certificate).
    The maximum P is shared across window-excluded one-photon, same-tone and
    cross-tone two-photon terms, including empty retained graphs. Only constant
    rows use this rule; sampled/gate admission is unchanged.

    Calibration: 96 once-only strict-lab points on the 16-level joint model,
    four flux fractions 0,0.4,0.8,1 between integer and near-half, theta 0/0.7,
    full-operator/carrier ratios 0.01/0.07/0.2, durations 128/1024 ns and carrier
    offsets 0/51 MHz. Minimum failing P (population error>0.1) was 0.045813;
    all 14 admitted points had error<=0.002947. The same-tone counterexample
    fails at P=0.015629. The existing 25-point cross-tone scan fails first at
    channel P=0.0116496. Its halved boundary, rounded down to 0.0058, is more
    conservative than either new boundary. A recheck of the dual-constant
    96-point calibration keeps 24/36 previously admitted points; twelve safe
    intermediate-flux points now fall back due to the new omissions.
    """
    x, y = quadratures
    maximum = 0.0
    size, count = len(energies), len(frequencies)
    additions = np.zeros_like(retained)
    operators = np.empty((count, 2, size, size), dtype=np.complex128)
    for a in range(count):
        for sa in range(2):
            sign_a = 1 if sa == 0 else -1
            operators[a, sa] = _fourier_coefficient(x[a], y[a], sign_a) * amplitudes[a]
            for i in range(size):
                for j in range(size):
                    d = energies[i] - energies[j] + sign_a * frequencies[a]
                    if not retained[a, sa, i, j]:
                        probability = _transfer_probability(
                            operators[a, sa, i, j], d, duration
                        )
                        maximum = max(maximum, probability)
                        if probability > _MAX_OMISSION_TRANSFER:
                            additions[a, sa, i, j] = True
                            additions[a, 1 - sa, j, i] = True
    for a in range(count):
        for b in range(a, count):
            for sa in range(2):
                for sb in range(2):
                    if a == b and sa > sb:
                        continue
                    data = (energies, operators, frequencies, retained)
                    first, second = (a, sa), (b, sb)
                    maximum = max(
                        maximum,
                        _pair_probability(data, first, second, duration, additions),
                    )
    return maximum, additions


@njit(nogil=True, cache=True)
def _coefficients(models, ports, frequencies, tones, retained=None):
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
                        keep = abs(detuning) < _EDGE_GHZ
                        if retained is not None:
                            keep = retained[tone, 0 if sign == 1 else 1, i, j]
                        if keep:
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
    model: SecularModel,
    port: int,
    frequency: float,
    context: KernelContext,
    retention: RWARetention | None = None,
) -> float:
    """Cache the near-resonant positive Fourier operator spectral norm."""
    key = FactKey(
        (
            "multitone-rwa-retained-norm",
            RWA_POLICY,
            ADAPTIVE_RWA_POLICY,
            model.identity,
            model.level,
            port,
            retained_mask(model, (frequency,), retention).tobytes(),
        ),
        float,
    )

    def build() -> float:
        return float(
            np.linalg.norm(retained_edges(model, port, frequency, retention), 2)
        )

    return context.work_plan.fact(key, build)


def _constant_retention(
    model: SecularModel,
    port: int,
    frequency: float,
    amplitude: float,
    duration_ns: float,
) -> RWARetention:
    """Resolve a single constant carrier's model-only retained set.

    Start at 50 MHz. Each round restores all over-threshold one-photon edges
    and all omitted nonzero first-order edges in an over-threshold two-photon
    channel's contributing paths, together with their adjoints. Recompute the
    complete shared omission estimate after each round. Monotone edge addition
    terminates on a finite graph. Consumers still enforce strength, phase,
    compatible-charge pruning and block-size limits. No multitone policy changes.
    Downstream graph/prepare identities include the final edge coordinates,
    not a guessed wider window.
    """
    drive = model.drives[port]
    frequencies = np.array([frequency])
    initial = _window_mask(model.energies_ghz, frequencies)
    retained = initial
    while True:
        probability, additions = _omission_analysis(
            model.energies_ghz,
            (drive.in_phase_ghz[None], drive.quadrature_ghz[None]),
            frequencies,
            np.array([abs(amplitude)]),
            duration_ns,
            retained,
        )
        if not np.any(additions & ~retained):
            break
        retained = retained | additions
    edges = ()
    if retained is not initial:
        edges = tuple(
            (int(i), int(j)) for i, j in np.argwhere(retained[0, 0] & ~initial[0, 0])
        )
    return RWARetention((edges,), float(probability))


def constant_retentions(
    table: SegmentTable, context: KernelContext
) -> ConstantRetentions:
    """Publish one retained-set inventory for a complete constant-row table.

    Each eligible row uses its own model, signed carrier, abs(gain) and duration;
    no edges are shared across rows. Ineligible rows have None. The table/source
    scoped fact is built once, then guard, keys and per-key builders reuse it.
    Graph facts and prepare keys additionally identify the resolved edge set.
    This avoids one resource lookup per row at every subsequent prepare step.
    """
    key = FactKey(
        ("constant-rwa-retentions", ADAPTIVE_RWA_POLICY, id(table), id(context.source)),
        ConstantRetentions,
    )

    def build() -> ConstantRetentions:
        results: list[RWARetention | None] = []
        for row in range(len(table.timeline)):
            if (
                table.tones[row] != 1
                or table.envelopes[table.tone_envelope[row, 0]].kind != "constant"
                or abs(table.tone_frequency_ghz[row, 0]) < _EDGE_GHZ
                or not fixed_flux(table, row)
            ):
                results.append(None)
                continue
            model = context.source.model(tuple(int(v) for v in table.level[row]))
            results.append(
                _constant_retention(
                    model,
                    int(table.tone_port[row, 0]),
                    float(table.tone_frequency_ghz[row, 0]),
                    float(table.tone_amplitude[row, 0]),
                    float(table.duration_ns[row]),
                )
            )
        return ConstantRetentions(tuple(results))

    return context.work_plan.fact(key, build)


def admit_rows(
    table: SegmentTable, rows: NDArray[np.int64], context: KernelContext
) -> NDArray[np.bool_]:
    """Admit simultaneous rows under the shared calibrated joint RWA policy.

    Requires at least two tones. The unified omission rule applies only to
    distinct-port dual constants; sampled/gate policies remain unchanged.
    Reads model-only facts, never density;
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


def admit_constant_rows(
    table: SegmentTable, rows: NDArray[np.int64], context: KernelContext
) -> NDArray[np.bool_]:
    """Admit single constant rows under the same omission/strength/phase policy.

    Fixed flux, |f|>=50 MHz and budget>=1e-3 are required. Graph pruning is a
    separate consumer admission; this function never reads input density.
    """
    return _admit_rows(table, rows, context, single_tone=True, constant_single=True)


def _constant_domain(
    table: SegmentTable, row: int, *, constant: bool, single: bool
) -> bool:
    """Keep single constants separate from distinct-port dual constants.

    Same-port Stark/probe spectra retain their Floquet/direct route: small
    population errors can still reorder a narrow spectrum's local maxima.
    Shaped rows use the existing simultaneous or single-gate domain.
    """
    if single:
        return constant
    return not constant or (
        table.tones[row] == 2 and table.tone_port[row, 0] != table.tone_port[row, 1]
    )


def _admit_rows(
    table: SegmentTable,
    rows: NDArray[np.int64],
    context: KernelContext,
    *,
    single_tone: bool,
    constant_single: bool = False,
) -> NDArray[np.bool_]:
    tolerances(context.budget)
    bounds = envelope_bounds(table, context.work_plan)
    accepted = np.zeros(len(rows), dtype=np.bool_)
    retentions = None
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
        if not _constant_domain(
            table, int(row), constant=constant, single=constant_single
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
        retention = None
        if constant_single:
            if retentions is None:
                retentions = constant_retentions(table, context)
            retention = retentions.rows[row]
            assert retention is not None  # The row passed the same domain checks above.
            if retention.omission_transfer > _MAX_OMISSION_TRANSFER:
                continue
        elif constant:
            ports = table.tone_port[row, :count]
            probability = _omission_probability(
                model.energies_ghz,
                np.array([model.drives[p].in_phase_ghz for p in ports]),
                np.array([model.drives[p].quadrature_ghz for p in ports]),
                table.tone_frequency_ghz[row, :count],
                np.abs(table.tone_amplitude[row, :count]) * bounds.peaks[envs],
                table.duration_ns[row],
            )
            if probability > _MAX_OMISSION_TRANSFER:
                continue
        ratio = 0.0
        phase = 0.0
        for tone, env in enumerate(envs):
            port = int(table.tone_port[row, tone])
            norms = quadrature_norms(model, port, context.work_plan)
            retained = _retained_norm(
                model,
                port,
                float(table.tone_frequency_ghz[row, tone]),
                context,
                retention,
            )
            phase += (
                table.tone_amplitude[row, tone] * bounds.peaks[env] * retained
            ) ** 2 / frequencies[tone]
            ratio += (
                abs(table.tone_amplitude[row, tone])
                * bounds.peaks[env]
                * (norms.in_phase + norms.quadrature)
                / frequencies[tone]
            )
        phase_limit = (
            _MAX_CONSTANT_PHASE_RAD if constant and count == 2 else _MAX_PHASE_RAD
        )
        accepted[index] = (
            ratio <= _MAX_RATIO
            and 2 * np.pi * table.duration_ns[row] * phase <= phase_limit
        )
    return accepted


def retained_mask(
    model: SecularModel,
    frequencies: tuple[float, ...],
    retention: RWARetention | None = None,
) -> NDArray[np.bool_]:
    """Return the exact final (tones,signs,M,M) Fourier-edge mask.

    Start with each signed 50 MHz window and union all adaptively restored
    positive edges and their adjoints. Equal masks permit graph sharing, but
    never sharing of frequency-dependent virtual diagonals or row admission.
    Misaligned retained sets raise ValueError. The caller owns the result.
    """
    mask = _window_mask(model.energies_ghz, np.array(frequencies))
    if retention is not None:
        if len(retention.edges) != len(frequencies):
            raise ValueError("RWA retention must align with carriers")
        for tone, edges in enumerate(retention.edges):
            for i, j in edges:
                mask[tone, 0, i, j] = True
                mask[tone, 1, j, i] = True
    return mask


def retained_edges(
    model: SecularModel,
    port: int,
    frequency: float,
    retention: RWARetention | None = None,
    tone: int = 0,
) -> NDArray[np.complex128]:
    """Return the positive-carrier joint operator in GHz/gain.

    The negative-carrier operator is its adjoint. Both RWA integrators and
    covariant gate construction default to the same 50 MHz retained-edge policy.
    retention overrides it with the resolved edges for the ordered tone index.
    The caller owns the returned mutable matrix.
    """
    drive = model.drives[port]
    gaps = model.energies_ghz[:, None] - model.energies_ghz[None, :]
    coupling = _fourier_coefficient(drive.in_phase_ghz, drive.quadrature_ghz, 1)
    mask = np.abs(gaps + frequency) < _EDGE_GHZ
    if retention is not None:
        for i, j in retention.edges[tone]:
            mask[i, j] = True
    return coupling * mask


def rwa_coefficients(
    model: SecularModel,
    ports: tuple[int, ...],
    frequencies: tuple[float, ...],
    retention: RWARetention | None = None,
    *,
    work_plan: WorkPlan | None = None,
) -> RWACoefficients:
    """Build all joint drive coefficients for ordered signed carriers.

    This is the shared coefficient owner for ordinary RWA and covariant gates.
    Each tone contributes sum_j,s |C_s[i,j]|²/(Ei-Ej+s*f) to the diagonal
    in GHz/gain², restricted to abs(Ei-Ej+s*f)>=0.05 GHz. Retained edges
    evolve explicitly, with no zero division or double counting. The common
    compiled builder also serves batched ordinary RWA preparation.
    retention overrides the initial window in both Fourier terms and virtual
    diagonal, so restored edges are never counted twice. Graph pruning may
    replace the Fourier operators but retains the diagonal.
    work_plan shares only model packing, never frequency-dependent diagonals.
    Returned arrays are owned and mutable. Misaligned/empty carriers raise
    ValueError; ports must index the supplied model's validated drives.
    """
    if not ports or len(ports) != len(frequencies):
        raise ValueError("RWA coefficients require aligned nonempty carriers")
    retained = None
    if retention is not None and any(retention.edges):
        retained = retained_mask(model, frequencies, retention)
    positive, negative, diagonal = _coefficients(
        pack_models([model]) if work_plan is None else packed_model(model, work_plan),
        np.array([ports], dtype=np.int64),
        np.array([frequencies], dtype=np.float64),
        np.array([len(ports)], dtype=np.int64),
        retained,
    )
    return RWACoefficients(positive[0], negative[0], diagonal[0])


class MultitoneRWAKernel:
    """Joint RWA C5 family with unchanged sampled and covariant-gate admission.

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

    Constant rows also use the unified omitted-edge transfer estimate documented
    in _omission_probability: window-excluded one-photon edges, same-tone and
    cross-tone two-step paths all share a 5.8e-3 cap. This extends the existing
    combination calibration rather than layering separate resonance guards.
    Sampled rows and covariant gates do NOT use this constant-only estimate.
    The threshold uses half the smallest failing proxy in strict-lab calibration;
    it is empirical, not a whole-sequence or arbitrary-input error certificate.
    Constant rows use a 16 ns ceiling; sampled rows retain 4 ns. No density
    reduction, dissipator change, clipping or normalization occurs.
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
                digest.update(repr(CONSTANT_RWA_POLICY).encode("ascii"))
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
