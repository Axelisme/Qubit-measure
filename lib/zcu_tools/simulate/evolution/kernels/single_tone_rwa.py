"""Endpoint-dressed constant rotating-frame Lindbladians by retained graph."""

from contextlib import closing
from dataclasses import dataclass, replace
from hashlib import sha256

import numpy as np
from numba import njit
from numpy.typing import NDArray
from scipy.linalg import expm

from ..kernel import ApplyContext, KernelContext, Prepared, SegmentTable
from ..secular import SecularModel
from ..workplan import WorkPlan
from ._dop853 import tolerances
from ._rwa_fourier import (
    EDGE_GHZ,
    RWACoefficients,
    endpoint_kick,
    interaction_hamiltonian,
)
from ._rwa_graph import (
    PRUNE_LIMIT,
    RWA_GRAPH_POLICY,
    PhaseGraph,
    phase_graph,
    pruning_bound,
)
from ._rwa_numeric import (
    MIN_RWA_BUDGET,
    SINGLE_CONSTANT_RWA_POLICY,
    ConstantRWAAdmission,
    RWAGuardMetric,
    RWARetention,
    RWAUncomputed,
    admit_constant_rows,
    constant_admission,
    constant_retentions,
    rwa_coefficients,
    rwa_kick_coefficients,
)
from ._workload_facts import rwa_topology, secular_bath

_MAX_BLOCK_DIMENSION = 64
# Kick-created cross-component coherences activate costly pair generators.
# Dropping <1e-4 accumulated 7.3e-4 error over 200 random-phase gates; use
# ten times less to leave margin under the 2e-3 final-density allowance.
_CROSS_COMPONENT_CUTOFF = 1e-5
_POLICY = (
    "constant-rwa-endpoint-kick-kappa2-v4",
    _CROSS_COMPONENT_CUTOFF,
    RWA_GRAPH_POLICY,
    SINGLE_CONSTANT_RWA_POLICY,
    _MAX_BLOCK_DIMENSION,
)


@dataclass(frozen=True)
class SingleToneRWADiagnostic:
    """Read-only admission metrics for one fixed-flux constant tone.

    Each RWAGuardMetric supplies value, inclusive threshold and passed status.
    budget is the dimensionless per-segment allowance, required >=1e-3.
    carrier_ghz is the absolute carrier in GHz, required >=0.05.
    kappa2 is ||sum_sign abs(c/d)||_2, the dimensionless spectral norm over
    omitted Fourier edges after adaptive restoration, including diagonal +/-f
    edges. The coefficients include gain; d is signed detuning in GHz. This
    uses the endpoint kick's resolved mask, not the pruned phase graph. Its
    0.08 cap has full-density calibration with margin on the fixed candidate
    acceptance set. This does not establish a maximum usable threshold or
    arbitrary-input accuracy. strength_ratio is dimensionless,
    abs(gain)*(||X||+||Y||)/abs(carrier), required <=0.25.
    phase_rad is 2*pi*duration*(gain*||retained positive Fourier operator||)^2 /
    abs(carrier), in rad, required <=0.06. pruning is the dimensionless phase-graph
    population omission bound, required <=min(budget,1e-3). real_dimension counts
    populations and real coordinates of within-component coherences, required
    <=64. accepted is true only when every comparison passes. These admission
    proxies do not bound final density error or sequence accuracy.

    RWAUncomputed marks skipped numeric work and has no value or passed status.
    budget always has a value. carrier_ghz is uncomputed after budget rejection.
    Kappa2 and strength are uncomputed after budget/carrier rejection. Phase
    is also uncomputed after strength rejection. Pruning and dimension are
    uncomputed after any base rejection (budget/carrier/kappa2/strength/phase).
    Dimension is still computed if pruning fails. Kappa2 is computed after
    retention, before strength/phase checks, and remains visible if either rejects.
    This is a numeric table, not a decision trace: a value and passed status do
    not mean the guard used that comparison in this particular rejection.
    """

    budget: RWAGuardMetric
    carrier_ghz: RWAGuardMetric | RWAUncomputed = RWAUncomputed()
    kappa2: RWAGuardMetric | RWAUncomputed = RWAUncomputed()
    strength_ratio: RWAGuardMetric | RWAUncomputed = RWAUncomputed()
    phase_rad: RWAGuardMetric | RWAUncomputed = RWAUncomputed()
    pruning: RWAGuardMetric | RWAUncomputed = RWAUncomputed()
    real_dimension: RWAGuardMetric | RWAUncomputed = RWAUncomputed()

    @property
    def accepted(self) -> bool:
        """Whether this constant row passes every single_tone_rwa guard."""
        return all(
            isinstance(metric, RWAGuardMetric) and metric.passed
            for metric in (
                self.budget,
                self.carrier_ghz,
                self.kappa2,
                self.strength_ratio,
                self.phase_rad,
                self.pruning,
                self.real_dimension,
            )
        )


def _diagnostic(
    admission: ConstantRWAAdmission,
    graph: PhaseGraph | None,
    amplitude: float,
    duration_ns: float,
    budget: float,
    work_plan: WorkPlan,
) -> SingleToneRWADiagnostic:
    diagnostic = SingleToneRWADiagnostic(
        RWAGuardMetric(budget, MIN_RWA_BUDGET, ">="),
        admission.carrier_ghz,
        admission.kappa2,
        admission.strength_ratio,
        admission.phase_rad,
    )
    if graph is None:
        return diagnostic
    return replace(
        diagnostic,
        pruning=RWAGuardMetric(
            pruning_bound(graph, np.array([abs(amplitude)]), duration_ns),
            min(budget, PRUNE_LIMIT),
        ),
        real_dimension=RWAGuardMetric(
            float(len(rwa_topology(graph, work_plan).a)), _MAX_BLOCK_DIMENSION
        ),
    )


def single_tone_rwa_diagnostic(
    model: SecularModel,
    port: str,
    frequency_ghz: float,
    amplitude: float,
    duration_ns: float,
    *,
    budget: float = 1e-3,
) -> SingleToneRWADiagnostic:
    """Read every single_tone_rwa numeric guard for one constant model row.

    model is the fixed-flux, energy-basis model used by that row, including any
    segment shifts. port names one of model.drives. frequency_ghz is any finite
    signed GHz, including zero; amplitude is finite signed dimensionless gain;
    duration_ns is finite and nonnegative. Carrier phase and input density do
    not affect these guards. budget is a finite positive per-segment allowance.

    Returns frozen metrics for computed values, and RWAUncomputed for work
    skipped by guard early returns. Low/zero carrier and positive budget below
    1e-3 return accepted=False. See SingleToneRWADiagnostic for each metric's
    availability and the 0.08 kappa2 cap, calibrated with full-density margin
    on the fixed candidate acceptance set. This does not establish a maximum
    usable threshold or arbitrary-input accuracy. Raises ValueError for an
    unknown port, nonfinite inputs, negative duration or invalid budget.
    Numerical construction failures propagate.
    No evolution, routing, history or preparation is changed;
    temporary model-only facts are released before return. This query does not
    diagnose shaped, multitone or moving-flux rows and does not select a family.
    """
    tolerances(budget)
    if (
        not np.isfinite(frequency_ghz)
        or not np.isfinite(amplitude)
        or not np.isfinite(duration_ns)
        or duration_ns < 0
    ):
        raise ValueError(
            "constant RWA requires finite carrier/gain and nonnegative finite duration"
        )
    ports = [drive.name for drive in model.drives]
    if port not in ports:
        raise ValueError(f"unknown constant RWA drive port: {port}")
    diagnostic = SingleToneRWADiagnostic(RWAGuardMetric(budget, MIN_RWA_BUDGET, ">="))
    if not diagnostic.budget.passed:
        return diagnostic
    carrier = RWAGuardMetric(abs(frequency_ghz), EDGE_GHZ, ">=")
    diagnostic = replace(diagnostic, carrier_ghz=carrier)
    if not carrier.passed:
        return diagnostic
    index = ports.index(port)
    with closing(WorkPlan()) as plan:
        admission = constant_admission(
            model, index, frequency_ghz, amplitude, duration_ns, plan
        )
        graph = None
        if admission.accepted:
            graph = phase_graph(
                model, (index,), (frequency_ghz,), plan, admission.retention
            )
        return _diagnostic(admission, graph, amplitude, duration_ns, budget, plan)


@dataclass(frozen=True)
class _Block:
    # Canonical Hamiltonian/decay, active Hermitian coordinates, graph components.
    h: NDArray[np.complex128]
    decay: NDArray[np.float64]
    generator: NDArray[np.float64]
    a: NDArray[np.int64]
    b: NDArray[np.int64]
    kind: NDArray[np.int64]
    components: tuple[NDArray[np.int64], ...]
    labels: NDArray[np.int64]
    charges: NDArray[np.int64]
    kick: NDArray[np.complex128]

    @property
    def nbytes(self) -> int:
        return sum(
            a.nbytes
            for a in (
                self.h,
                self.decay,
                self.generator,
                self.a,
                self.b,
                self.kind,
                self.labels,
                self.charges,
                self.kick,
                *self.components,
            )
        )


@dataclass(frozen=True)
class _Prepared:
    entries: tuple[_Block, ...]

    @property
    def nbytes(self) -> int:
        return sum(e.nbytes for e in self.entries)


@njit(nogil=True, cache=True)
def _generator(h, population, decay, a, b, kind):
    # Real matrix in a Hermitian basis, applied complex-linearly. This preserves
    # non-Hermitian residuals too; it is NOT a projection onto physical states.
    dimension = len(a)
    result = np.empty((dimension, dimension))
    for r in range(dimension):
        i, j = a[r], b[r]
        for c in range(dimension):
            x, y = a[c], b[c]
            v = 1j if kind[c] == 2 else 1 + 0j
            value = (
                -2j
                * np.pi
                * v
                * ((h[i, x] if j == y else 0j) - (h[y, j] if i == x else 0j))
            )
            if kind[c] != 0:
                value += (
                    -2j
                    * np.pi
                    * v.conjugate()
                    * ((h[i, y] if j == x else 0j) - (h[x, j] if i == y else 0j))
                )
            if i == j:
                if kind[c] == 0:
                    value += population[i, x]
            elif x == i and y == j:
                value -= decay[i, j] * v
            result[r, c] = value.imag if kind[r] == 2 else value.real
    return result


def _build(
    table: SegmentTable,
    row: int,
    context: KernelContext,
    retention: RWARetention | None,
) -> _Block:
    if retention is None:
        raise ValueError("constant RWA requires a single fixed-flux constant carrier")
    model = table.model(context.source, row)
    port, frequency = (
        int(table.tone_port[row, 0]),
        float(table.tone_frequency_ghz[row, 0]),
    )
    graph = phase_graph(model, (port,), (frequency,), context.work_plan, retention)
    original = rwa_coefficients(
        model, (port,), (frequency,), retention, work_plan=context.work_plan
    )
    coefficients = RWACoefficients(
        graph.positive,
        graph.positive.conj().transpose(0, 2, 1).copy(),
        original.diagonal,
    )
    energies = model.energies_ghz + graph.charges[:, 0] * frequency
    energies = energies - energies[0]
    h = np.diag(energies).astype(np.complex128) + interaction_hamiltonian(
        0.0, energies, coefficients, np.array([complex(table.tone_amplitude[row, 0])])
    )
    topology = rwa_topology(graph, context.work_plan)
    # Resident payloads own and charge their arrays even when facts are shared.
    components = tuple(component.copy() for component in topology.components)
    a, b, kind = (v.copy() for v in (topology.a, topology.b, topology.kind))
    bath = secular_bath(model, context.work_plan)
    decay = bath.decay.copy()
    generator = _generator(h, bath.population, decay, a, b, kind)
    charges = graph.charges[:, 0].copy()
    labels = topology.labels.copy()
    kick = rwa_kick_coefficients(
        model, port, frequency, float(table.tone_amplitude[row, 0]), retention
    )
    for array in (h, decay, generator, a, b, kind, labels, charges, kick, *components):
        array.setflags(write=False)
    return _Block(h, decay, generator, a, b, kind, components, labels, charges, kick)


@njit(nogil=True, cache=True)
def _initial_coordinates(operators, coordinates, topology, rho, theta, duration):
    h, decay = operators
    a, b, kind = coordinates
    charges, labels = topology
    size = len(charges)
    phase = np.exp(1j * charges * theta)
    rotated = np.empty_like(rho)
    result = np.empty_like(rho)
    active = np.zeros((size, size), dtype=np.bool_)
    for i in range(size):
        for j in range(size):
            value = rho[i, j] * phase[i].conjugate() * phase[j]
            if labels[i] != labels[j] and abs(value) < _CROSS_COMPONENT_CUTOFF:
                value = 0j
            rotated[i, j] = value
            result[i, j] = value * np.exp(
                (-2j * np.pi * (h[i, i] - h[j, j]) - decay[i, j]) * duration
            )
            if labels[i] != labels[j] and value != 0:
                # Either direction activates the pair, including non-Hermitian
                # residuals, after the cross-component small-entry cutoff.
                active[min(labels[i], labels[j]), max(labels[i], labels[j])] = True
    state = np.empty(len(a), dtype=np.complex128)
    for k in range(len(a)):
        i, j = a[k], b[k]
        if kind[k] == 0:
            state[k] = rotated[i, j]
        elif kind[k] == 1:
            state[k] = (rotated[i, j] + rotated[j, i]) / 2
        else:
            state[k] = (rotated[i, j] - rotated[j, i]) / (2j)
    return rotated, result, state, active


@njit(nogil=True, cache=True)
def _finish_coordinates(result, out, a, b, kind):
    # Preparation emits each real/imaginary coordinate pair adjacently.
    # Both coordinates can be complex: this is not Hermitian projection.
    for k in range(len(a)):
        i, j = a[k], b[k]
        if kind[k] == 0:
            result[i, j] = out[k]
        elif kind[k] == 1:
            result[i, j] = out[k] + 1j * out[k + 1]
            result[j, i] = out[k] - 1j * out[k + 1]


@njit(nogil=True, cache=True)
def _finish_frame(result, charges, theta, frequency, duration):
    end = np.exp(1j * charges * (theta + 2 * np.pi * frequency * duration))
    for i in range(len(charges)):
        for j in range(len(charges)):
            result[i, j] *= end[i] * end[j].conjugate()
    return result


def _play(
    entry: _Block,
    rho: NDArray[np.complex128],
    frequency: float,
    theta: float,
    duration: float,
) -> NDArray[np.complex128]:
    initial = expm(1j * endpoint_kick(entry.kick, theta))
    rho = initial @ rho @ initial.conj().T
    rho, result, state, active = _initial_coordinates(
        (entry.h, entry.decay),
        (entry.a, entry.b, entry.kind),
        (entry.charges, entry.labels),
        rho,
        theta,
        duration,
    )
    out = expm(entry.generator * duration) @ state
    _finish_coordinates(result, out, entry.a, entry.b, entry.kind)
    for index, ca in enumerate(entry.components):
        for other in range(index + 1, len(entry.components)):
            cb = entry.components[other]
            if len(ca) == len(cb) == 1 or not active[index, other]:
                continue  # Scalar closed form, or an exact-zero RHS.
            forward, reverse = np.ix_(ca, cb), np.ix_(cb, ca)
            ha, hb = entry.h[np.ix_(ca, ca)], entry.h[np.ix_(cb, cb)]
            generator = (
                -2j
                * np.pi
                * (np.kron(ha, np.eye(len(cb))) - np.kron(np.eye(len(ca)), hb.T))
            )
            generator[np.diag_indices(len(generator))] -= entry.decay[forward].ravel()
            center = np.trace(generator) / len(generator)
            channel = np.exp(center * duration) * expm(
                (generator - center * np.eye(len(generator))) * duration
            )
            result[forward] = (channel @ rho[forward].ravel()).reshape(len(ca), len(cb))
            result[reverse] = (
                (channel @ rho[reverse].conj().T.ravel())
                .reshape(len(ca), len(cb))
                .conj()
                .T
            )
    result = _finish_frame(result, entry.charges, theta, frequency, duration)
    terminal = expm(
        -1j * endpoint_kick(entry.kick, theta + 2 * np.pi * frequency * duration)
    )
    return terminal @ result @ terminal.conj().T


class SingleToneRWAKernel:
    """Single constant-tone C5 family with arbitrary complex density playback.

    Uses final kappa2 <=0.08 admission, the shared strength comparison and
    0.06 rad phase admission, budget >=1e-3 and fixed flux. The 0.08 cap has
    full-density calibration with margin on the fixed candidate acceptance set.
    This does not establish a maximum usable threshold or arbitrary-input
    accuracy. The retained graph must also pass the explicit
    min(budget,1e-3) pruning bound. Preparation never sees density. Each key
    stores a constant rotating-frame Hamiltonian and full secular bath, with
    all populations and within-component coherences in one real generator
    of dimension at most 64 (larger graphs retain the original fallback).
    A 168-point synthetic sweep of chains and two disjoint eight-level blocks
    covered real-generator dimensions through 128, detunings 51/70/110 MHz,
    gains 0.5/1, durations 30/200 ns and phases 0/0.7. Maximum population
    difference against Floquet period playback was 0.010484 (<0.1); halving
    the largest verified dimension gives the 64 cap (2x safety factor).
    The 50 MHz window is only the initial edge set. Each iteration restores
    every over-threshold one-photon edge and the strongest missing first-order
    path per over-threshold two-photon channel and signed time order. The complete
    omission rule is recomputed each round, with no amplitude cutoff.
    The shared omission estimate, Fourier terms and virtual diagonal all use
    that same final set; incompatible phase cycles still require pruning.
    First-order endpoint kicks restore omitted-edge micromotion before and
    after each row. Other component pairs use small complex generators;
    cross-component input entries below 1e-5 are dropped at apply.
    No clipping, normalization, dense full-space channel or Strang splitting
    is used.
    This is a calibrated RWA approximation, not an error certificate.
    """

    name = "single_tone_rwa"
    revision = _POLICY
    guard_builds = False

    def guard(
        self, table: SegmentTable, rows: NDArray[np.int64], context: KernelContext
    ) -> NDArray[np.bool_]:
        """Return row-local single constant RWA and retained-graph admissions."""
        accepted = admit_constant_rows(table, rows, context)
        if not np.any(accepted):
            return accepted
        retentions = constant_retentions(table, context)
        for index, row in enumerate(rows):
            if not accepted[index]:
                continue
            model = table.model(context.source, row)
            port = int(table.tone_port[row, 0])
            frequency = float(table.tone_frequency_ghz[row, 0])
            retention = retentions.rows[row]
            graph = phase_graph(
                model, (port,), (frequency,), context.work_plan, retention
            )
            admission = constant_admission(
                model,
                port,
                frequency,
                float(table.tone_amplitude[row, 0]),
                float(table.duration_ns[row]),
                context.work_plan,
            )
            accepted[index] = _diagnostic(
                admission,
                graph,
                float(table.tone_amplitude[row, 0]),
                float(table.duration_ns[row]),
                context.budget,
                context.work_plan,
            ).accepted
        return accepted

    def prepare_keys(
        self, table: SegmentTable, rows: NDArray[np.int64], context: KernelContext
    ) -> NDArray[np.void]:
        """Hash model, policy, final retained set, budget, port, carrier and gain.

        Duration affects the retained set, not identity within a resolved set.
        Start/phase affect playback only; no input density enters this model-only
        key. Rows without a retention receive inventory-only keys, disjoint
        from retained-set keys. Only guard-accepted rows may prepare or apply.
        """
        keys = np.empty(len(rows), dtype="V32")
        retentions = constant_retentions(table, context)
        for index, row in enumerate(rows):
            model = table.model(context.source, row)
            retention = retentions.rows[row]
            identity = (
                _POLICY,
                retention.edges if retention is not None else None,
                model.identity,
                model.level,
                context.budget,
                int(table.tone_port[row, 0]),
                float(table.tone_frequency_ghz[row, 0]),
                float(table.tone_amplitude[row, 0]),
            )
            keys[index] = np.void(sha256(repr(identity).encode()).digest())
        return keys

    def cost(self, table: SegmentTable, rows: NDArray[np.int64]) -> NDArray[np.float64]:
        """Use the existing constant-block price pending workload calibration."""
        del table
        return np.full(len(rows), 4.0, dtype=np.float64)

    def prepare(
        self,
        keys: NDArray[np.void],
        table: SegmentTable,
        rows: NDArray[np.int64],
        context: KernelContext,
    ) -> Prepared:
        """Build immutable block generators aligned with representative keys.

        Raises ValueError for misaligned keys/rows or rows without a retention.
        Arrays are fully charged to nbytes; no density-dependent data enters
        resident preparation.
        """
        if len(keys) != len(rows):
            raise ValueError("constant RWA keys and rows must align")
        retentions = constant_retentions(table, context)
        return _Prepared(
            tuple(
                _build(table, int(row), context, retentions.rows[row]) for row in rows
            )
        )

    def guard_prepared(
        self, table: SegmentTable, rows: NDArray[np.int64], context: KernelContext
    ) -> None:
        """Use the ordinary guard without preparing any numeric generator."""
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
        """No input binding; apply owns the cross-component small-entry cutoff."""
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
        """Evolve in place, restoring lab endpoints; return all-False exceedances.

        states is (rows,M,M) in the model eigenbasis. Reject invalid payloads,
        row plans, indices and shapes with ValueError. Numerical failures
        propagate, without retry or rerouting. Endpoint kicks surround rotating-
        frame playback. Complex residuals are not symmetrized; cross-component
        input entries below 1e-5 after the initial kick are discarded.
        """
        if context.plan is not None or not isinstance(prepared, _Prepared):
            raise ValueError("constant RWA requires its own payload and no row plan")
        if (
            key_index.shape != rows.shape
            or states.ndim != 3
            or len(states) != len(rows)
        ):
            raise ValueError("constant RWA state/index shape mismatch")
        for index, row in enumerate(rows):
            key = int(key_index[index])
            if not 0 <= key < len(prepared.entries):
                raise ValueError("constant RWA prepared index out of range")
            entry = prepared.entries[key]
            if states.shape[1:] != entry.h.shape:
                raise ValueError("constant RWA state dimension mismatch")
            states[index] = _play(
                entry,
                states[index],
                float(table.tone_frequency_ghz[row, 0]),
                float(table.tone_theta_rad[row, 0]),
                float(table.duration_ns[row]),
            )
        return np.zeros(len(rows), dtype=np.bool_)
