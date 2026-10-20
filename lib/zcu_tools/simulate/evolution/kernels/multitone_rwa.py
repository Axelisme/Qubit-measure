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
from . import _rwa_fourier
from ._dop853 import Tolerances, integrate, tolerances
from ._dop853_data import Models, Segments, pack_models
from ._workload_facts import (
    envelope_bounds,
    fixed_flux,
    packed_model,
    packed_segments,
    quadrature_norms,
)

_REVISION = "joint-rwa-v3"
_STEP_NS = 4.0
_CONSTANT_STEP_NS = 16.0
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
    _rwa_fourier.MAX_OMISSION_TRANSFER,
    _MAX_CONSTANT_PHASE_RAD,
)
RWA_POLICY = (
    _REVISION,
    _rwa_fourier.EDGE_GHZ,
    _STEP_NS,
    _CONSTANT_STEP_NS,
    _rwa_fourier.MAX_OMISSION_TRANSFER,
    _MIN_BUDGET,
    _MAX_RATIO,
    _MAX_SLOPE,
    _MAX_CURVATURE,
    _MAX_PHASE_RAD,
)


ADAPTIVE_RWA_POLICY = ("omission-strongest-parent-v2", CONSTANT_RWA_POLICY, RWA_POLICY)
_MAX_OMISSION_PHASE_RAD = 0.0045
_MAX_RETAINED_DIMENSION = 256
MULTITONE_RETENTION_POLICY = (
    "multitone-omission-retention-v1",
    ADAPTIVE_RWA_POLICY,
    _MAX_OMISSION_PHASE_RAD,
    _MAX_RETAINED_DIMENSION,
)


@dataclass(frozen=True)
class RWARetention:
    """Resolved additions to the initial window and remaining omission estimate.

    edges has one ordered tuple of added (i,j) coordinates per positive carrier;
    the final set is the 50 MHz window union these edges. Negative Fourier masks
    are the transposes. omission_transfer and omission_phase are maximum
    remaining first/second-order transfer and phase proxies, not rigorous bounds.
    dimension is sum(component_size**2), the real population/coherence dimension
    of the full Fourier union graph, including isolated populations. Phase and
    dimension are None for single constant rows, whose phase-graph owner keeps
    its original phase/pruned-dimension policy. Exceeding a requested dimension
    limit returns infinite omission proxies without further analysis.
    """

    edges: tuple[tuple[tuple[int, int], ...], ...]
    omission_transfer: float
    omission_phase: float | None = None
    dimension: int | None = None


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
        respectively, after the omission-resolved Fourier-edge selection.
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


def _resolve_retention(
    model: SecularModel,
    ports: tuple[int, ...],
    carriers: tuple[float, ...],
    amplitudes: NDArray[np.float64],
    duration_ns: float,
    *,
    limits: tuple[float | None, int | None] = (None, None),
) -> RWARetention:
    """Resolve model-only retained edges for single or simultaneous carriers.

    Start at 50 MHz. Restore over-threshold one-photon edges and the strongest
    missing path per over-threshold two-photon channel/time order, with adjoints.
    Recompute after each round. Monotone addition terminates on a finite graph.
    A finite phase_limit also restores channels with excessive residual phase;
    dimension_limit stops expansion when the real graph dimension is too large.
    limits is (phase_limit in rad or None, dimension_limit or None).
    None skips the corresponding analysis, including its compiled helpers.
    Single constant callers keep their original phase/strength/pruning policy.
    """
    phase_limit, dimension_limit = limits
    quadratures = (
        np.array([model.drives[p].in_phase_ghz for p in ports]),
        np.array([model.drives[p].quadrature_ghz for p in ports]),
    )
    # Match immutable model quadratures used by coefficient construction, so
    # admission does not compile an otherwise identical writeable specialization.
    for quadrature in quadratures:
        quadrature.setflags(write=False)
    frequencies = np.array(carriers)
    initial = _rwa_fourier.window_mask(model.energies_ghz, frequencies)
    retained = initial
    probability = phase = np.inf
    dimension: int | None = None
    while True:
        if dimension_limit is not None:
            dimension = int(_rwa_fourier.retained_dimension(retained, quadratures))
            if dimension > dimension_limit:
                break
        probability, phase, additions = _rwa_fourier.omission_analysis(
            model.energies_ghz,
            quadratures,
            frequencies,
            (amplitudes, duration_ns),
            retained,
            phase_limit,
        )
        if not np.any(additions & ~retained):
            break
        retained = retained | additions
    edges = tuple(
        tuple((int(i), int(j)) for i, j in np.argwhere(retained[a, 0] & ~initial[a, 0]))
        for a in range(len(ports))
    )
    return RWARetention(
        edges,
        float(probability),
        None if phase_limit is None else float(phase),
        dimension,
    )


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
                or abs(table.tone_frequency_ghz[row, 0]) < _rwa_fourier.EDGE_GHZ
                or not fixed_flux(table, row)
            ):
                results.append(None)
                continue
            model = context.source.model(tuple(int(v) for v in table.level[row]))
            results.append(
                _resolve_retention(
                    model,
                    (int(table.tone_port[row, 0]),),
                    (float(table.tone_frequency_ghz[row, 0]),),
                    np.abs(table.tone_amplitude[row, :1]),
                    float(table.duration_ns[row]),
                )
            )
        return ConstantRetentions(tuple(results))

    return context.work_plan.fact(key, build)


def multitone_retention(
    table: SegmentTable, row: int, context: KernelContext
) -> RWARetention:
    """Resolve a row's omitted channels using peak-weighted envelope bounds.

    Caller establishes the shaped-single or multitone fixed-flux/smooth domain. The
    row-local fact includes signed carriers, gains, peaks and duration; it never
    depends on other scan rows or input densities. Keys and preparation reuse
    the final mask. Phase and dimension limits are part of its policy identity.
    """
    count = int(table.tones[row])
    model = context.source.model(tuple(int(v) for v in table.level[row]))
    ports = tuple(int(v) for v in table.tone_port[row, :count])
    frequencies = tuple(float(v) for v in table.tone_frequency_ghz[row, :count])
    bounds = envelope_bounds(table, context.work_plan)
    amplitudes = (
        np.abs(table.tone_amplitude[row, :count])
        * bounds.peaks[table.tone_envelope[row, :count]]
    )
    duration = float(table.duration_ns[row])
    key = FactKey(
        (
            "multitone-retention",
            MULTITONE_RETENTION_POLICY,
            model.identity,
            model.level,
            ports,
            frequencies,
            tuple(amplitudes),
            duration,
        ),
        RWARetention,
    )
    return context.work_plan.fact(
        key,
        lambda: _resolve_retention(
            model,
            ports,
            frequencies,
            amplitudes,
            duration,
            limits=(_MAX_OMISSION_PHASE_RAD, _MAX_RETAINED_DIMENSION),
        ),
    )


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

    Uses shared omission-driven retention with calibrated transfer, residual
    phase and block limits. Smoothness, fixed-flux and budget still apply.
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
    adaptive_multitone: bool = False,
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
        if np.min(frequencies) < _rwa_fourier.EDGE_GHZ or (
            not adaptive_multitone
            and np.any(np.diff(np.sort(frequencies)) < 2 * _rwa_fourier.EDGE_GHZ)
        ):
            continue
        model = context.source.model(tuple(int(v) for v in table.level[row]))
        if adaptive_multitone or (single_tone and not constant_single):
            resolved = multitone_retention(table, int(row), context)
            # Shaped single gates and ordinary multitone request both limits.
            assert (
                resolved.dimension is not None and resolved.omission_phase is not None
            )
            accepted[index] = (
                resolved.omission_transfer <= _rwa_fourier.MAX_OMISSION_TRANSFER
                and resolved.omission_phase <= _MAX_OMISSION_PHASE_RAD
                and resolved.dimension <= _MAX_RETAINED_DIMENSION
            )
            continue
        retention = None
        if constant_single:
            if retentions is None:
                retentions = constant_retentions(table, context)
            retention = retentions.rows[row]
            assert retention is not None  # The row passed the same domain checks above.
        elif constant:
            ports = table.tone_port[row, :count]
            probability = _rwa_fourier.omission_probability(
                model.energies_ghz,
                np.array([model.drives[p].in_phase_ghz for p in ports]),
                np.array([model.drives[p].quadrature_ghz for p in ports]),
                table.tone_frequency_ghz[row, :count],
                np.abs(table.tone_amplitude[row, :count]) * bounds.peaks[envs],
                float(table.duration_ns[row]),
            )
            if probability > _rwa_fourier.MAX_OMISSION_TRANSFER:
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
            and (
                retention is None
                or retention.omission_transfer <= _rwa_fourier.MAX_OMISSION_TRANSFER
            )
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
    mask = _rwa_fourier.window_mask(model.energies_ghz, np.array(frequencies))
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
    coupling = _rwa_fourier.fourier_coefficient(
        drive.in_phase_ghz, drive.quadrature_ghz, 1
    )
    mask = np.abs(gaps + frequency) < _rwa_fourier.EDGE_GHZ
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
    in GHz/gain², summed over the final omitted Fourier mask. Without adaptive
    retention this is abs(Ei-Ej+s*f)>=0.05 GHz. Retained edges evolve explicitly,
    with no zero division or double counting. The common
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
    positive, negative, diagonal = _rwa_fourier.coefficients(
        pack_models([model]) if work_plan is None else packed_model(model, work_plan),
        np.array([ports], dtype=np.int64),
        np.array([frequencies], dtype=np.float64),
        np.array([len(ports)], dtype=np.int64),
        retained,
    )
    return RWACoefficients(positive[0], negative[0], diagonal[0])


class MultitoneRWAKernel:
    """Joint RWA C5 family with omission-driven Fourier retention.

    Admission requires >=2 tones with a sampled envelope, or exactly two
    constant envelopes on distinct ports, fixed flux,
    real envelopes with peak-normalized slope <=0.025/ns and curvature
    <=0.003/ns², each |f|>=0.05 GHz and budget >=1e-3. Carrier spacing
    does not veto admission: close or equal carriers use the same joint
    omission solver and retained-dimension limit. Full-operator/carrier and
    retained-norm phase caps do not apply here. Single-constant RWA and
    covariant gates keep their policies.

    The shared omission solver starts from 50 MHz edges and estimates only
    omitted one-photon and same/cross-tone two-photon terms with peak envelope
    weights. It restores over-threshold terms and their contributing paths,
    recomputing until remaining transfer <=0.0058 and residual phase <=0.0045 rad.
    Each over-threshold combination channel/time order restores its strongest
    missing parent path per round, not all nonzero virtual paths. No amplitude
    cutoff is used; all remaining paths are re-estimated after every round.
    Static single-tone second-order diagonals are already explicit, so their
    one-photon phase estimate charges the fourth-order remainder instead.
    The final Fourier union must fit real population/coherence dimension
    sum(component_size**2) <=256, the same measure as single constant RWA.
    Resolved masks enter both coefficient builders and prepare/fact identities.

    Transfer retains half the measured 0.0116496 combination-channel boundary.
    Phase calibration uses a 24-point two-tone, 16-level joint-model grid:
    integer/quarter/half flux, constant/Gaussian (sigma=T/4), full-operator/carrier
    ratios 0.1/0.5, durations 128/1024 ns and phases 0.7/-1.2 rad. Candidate
    budget is 1e-3; lab DOP853 reference is 1e-7. Both transfer-only masks and
    final masks after phase restoration are checked: restoration can change
    the proxy/error relation. The smallest failing remaining phase is
    0.00934095 rad. The 0.0045 cap leaves a 2.0758 safety factor; final-grid
    maximum population error is 0.07511. Tightening apply alone does not fix
    the failing omitted-channel approximations.

    A 24-point chain/block scan covers real dimensions 4,16,64,128,256,512,
    constant/Gaussian envelopes and 30/200 ns (short Gaussians are forced solver
    probes, not smoothness admissions). Maximum population error is 0.001281;
    256 halves the validated 512 ceiling. At dimension 256, build plus apply
    is faster than same-budget DOP853 in all four envelope/duration probes.
    These are empirical admission rules, not arbitrary-input or sequence error
    certificates. Constant rows use a 16 ns ceiling; sampled rows retain 4 ns.
    Dispatch may prefer full-drive DOP853 even after numerical admission when
    restoring fast edges makes cold prepare/guard/apply more expensive. That
    choice does not change these omission bounds or the retained-dimension cap.
    No density reduction, dissipator change, clipping or normalization occurs.
    """

    name = "multitone_rwa"
    revision = MULTITONE_RETENTION_POLICY
    guard_builds = False

    def guard(
        self, table: SegmentTable, rows: NDArray[np.int64], context: KernelContext
    ) -> NDArray[np.bool_]:
        """Return row-local admission under the explicit calibrated policy above."""
        return _admit_rows(
            table, rows, context, single_tone=False, adaptive_multitone=True
        )

    def prepare_keys(
        self, table: SegmentTable, rows: NDArray[np.int64], context: KernelContext
    ) -> NDArray[np.void]:
        """Hash policy, budget, model level and complete active tone/envelope content."""
        keys = np.empty(len(rows), dtype="V32")
        for index, row in enumerate(rows):
            count = table.tones[row]
            retention = multitone_retention(table, int(row), context)
            digest = sha256(
                repr(
                    (MULTITONE_RETENTION_POLICY, retention.edges, context.budget)
                ).encode("ascii")
            )
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
        positive = np.zeros(
            (
                len(rows),
                table.tone_port.shape[1],
                models.energies.shape[1],
                models.energies.shape[1],
            ),
            dtype=np.complex128,
        )
        negative = np.zeros_like(positive)
        diagonal = np.zeros(
            (len(rows), int(np.max(table.tones[rows])), models.energies.shape[1])
        )
        for index, row in enumerate(rows):
            count = int(table.tones[row])
            model = context.source.model(tuple(int(v) for v in table.level[row]))
            coefficients = rwa_coefficients(
                model,
                tuple(int(v) for v in table.tone_port[row, :count]),
                tuple(float(v) for v in table.tone_frequency_ghz[row, :count]),
                multitone_retention(table, int(row), context),
                work_plan=context.work_plan,
            )
            positive[index, :count] = coefficients.positive
            negative[index, :count] = coefficients.negative
            diagonal[index, :count] = coefficients.diagonal
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
