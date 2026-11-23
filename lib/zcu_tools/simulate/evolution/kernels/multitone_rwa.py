"""Joint-energy-basis RWA for simultaneous sampled or constant pairs."""

from contextlib import closing
from dataclasses import dataclass
from hashlib import sha256
from typing import NamedTuple

import numpy as np
from numba import njit
from numpy.typing import NDArray
from scipy.linalg import expm

from .._envelope import sample_value
from .._probe_memo import ProbeCertificate
from ..kernel import ApplyContext, KernelContext, Prepared, SegmentTable
from ..secular import SecularModel
from ..workplan import FactKey, WorkPlan
from ._dop853 import Tolerances, integrate, tolerances
from ._dop853_data import Models, Segments, pack_models
from ._multitone_spot import (
    SPOT_POLICY,
    SampledGroup,
    probe_density,
    probe_key,
    sampled_groups,
)
from ._rwa_fourier import (
    RWACoefficients,
    endpoint_kick,
    interaction_hamiltonian,
    kick_coefficients,
)
from ._rwa_numeric import (
    CONSTANT_RWA_POLICY,
    CONSTANT_STEP_NS,
    MULTITONE_GUARD_POLICY,
    STEP_NS,
    RWAGuardMetric,
    RWARetention,
    RWAUncomputed,
    admit_multitone_rows,
    multitone_constant_admission,
    multitone_retention,
    retained_mask,
    rwa_coefficients,
)
from ._workload_facts import TableIdentity, packed_segments
from .dop853 import DOP853Plan, EndpointData, advance_endpoints


@dataclass(frozen=True)
class MultitoneRWADiagnostic:
    """Read-only admission metrics for two fixed-flux constant tones.

    Computed fields are RWAGuardMetric values with inclusive thresholds and a
    passed status. budget is the dimensionless per-segment allowance, required
    >=1e-3. carrier_ghz is the smaller absolute carrier in GHz, required >=0.05.
    omission_transfer is the final retained mask's remaining first/second-order
    transfer proxy, dimensionless, required <=0.0058. omission_phase_rad is its
    remaining omission phase in radians, required <=0.0045. real_dimension is
    the sum of squared retained component sizes, including isolated populations,
    required <=256. secular_accumulation is epsilon_dom**2 * Omega_ret * duration
    in cycles, required <=1.476e-4. epsilon_dom is the largest omitted coupling
    over absolute detuning; Omega_ret is twice the largest retained coupling in
    GHz. These proxies do not bound final density error or sequence accuracy.

    RWAUncomputed has neither value nor passed status and denotes skipped work,
    not a failed comparison. Only final-mask metrics are reportable, even for a
    rejected row; prior-mask values after restoration are not final metrics.
    accepted is a bool reporting numerical admission, not dispatch selection.

    Budget is always computed. Budget or same-port rejection skips all later
    metrics; carrier rejection skips retention and secular metrics. Dimension
    remains visible after a retention rejection. If restoration exceeds the
    dimension limit before final-mask omission analysis, transfer, phase and
    secular metrics are uncomputed rather than reporting prior-mask values.
    """

    budget: RWAGuardMetric | RWAUncomputed = RWAUncomputed()
    carrier_ghz: RWAGuardMetric | RWAUncomputed = RWAUncomputed()
    omission_transfer: RWAGuardMetric | RWAUncomputed = RWAUncomputed()
    omission_phase_rad: RWAGuardMetric | RWAUncomputed = RWAUncomputed()
    real_dimension: RWAGuardMetric | RWAUncomputed = RWAUncomputed()
    secular_accumulation: RWAGuardMetric | RWAUncomputed = RWAUncomputed()
    accepted: bool = False


def multitone_rwa_diagnostic(
    model: SecularModel,
    ports: tuple[str, str],
    frequencies_ghz: tuple[float, float],
    amplitudes: tuple[float, float],
    duration_ns: float,
    *,
    budget: float = 1e-3,
) -> MultitoneRWADiagnostic:
    """Read multitone_rwa admission for two fixed-flux constant tones.

    model is the row's SecularModel, including any segment shifts. The aligned
    length-two tuples name known ports, finite signed GHz carriers and finite
    signed dimensionless gains. duration_ns is finite nonnegative row length in
    ns; budget is a finite positive dimensionless per-segment allowance. Phase
    and initial density do not enter the query.

    Returns the guard's computed final-mask comparisons, including failed
    metrics on rejected rows, and RWAUncomputed for skipped work. Same-port,
    low/zero carrier and positive budget below 1e-3 return accepted=False.
    Raises ValueError for explicit collapse operators, unknown ports, wrong
    tuple lengths, nonfinite inputs, negative duration or invalid budget. Numerical construction failures
    propagate. The query does not evolve density, select a family, mutate inputs
    or history, or retain facts across calls. Temporary facts close before return.
    """
    if model.collapse_operators:
        raise ValueError("multitone_rwa does not support explicit collapse operators")
    tolerances(budget)
    if len(ports) != 2 or len(frequencies_ghz) != 2 or len(amplitudes) != 2:
        raise ValueError("constant multitone RWA requires two aligned tones")
    if (
        not all(
            np.isfinite(value) for value in (*frequencies_ghz, *amplitudes, duration_ns)
        )
        or duration_ns < 0
    ):
        raise ValueError(
            "constant RWA requires finite carrier/gain and nonnegative finite duration"
        )
    names = [drive.name for drive in model.drives]
    for port in ports:
        if port not in names:
            raise ValueError(f"unknown constant RWA drive port: {port}")
    with closing(WorkPlan()) as plan:
        admission = multitone_constant_admission(
            model,
            (names.index(ports[0]), names.index(ports[1])),
            frequencies_ghz,
            amplitudes,
            (duration_ns, budget),
            plan,
        )
        return MultitoneRWADiagnostic(
            budget=admission.budget,
            carrier_ghz=admission.carrier_ghz,
            omission_transfer=admission.omission_transfer,
            omission_phase_rad=admission.omission_phase_rad,
            real_dimension=admission.real_dimension,
            secular_accumulation=admission.secular_accumulation,
            accepted=admission.accepted,
        )


class _DensitySupport(NamedTuple):
    # Row-major density coordinates, inverse lookup and Hamiltonian CSR graph.
    # Every population is present. Cross-component rectangles are present iff
    # either input rectangle contains an exactly nonzero entry.
    coordinates: NDArray[np.int64]
    lookup: NDArray[np.int64]
    offsets: NDArray[np.int64]
    neighbors: NDArray[np.int64]
    edges: NDArray[np.int64]


class _Context(NamedTuple):
    models: Models
    segments: Segments
    positive: NDArray[np.complex128]
    negative: NDArray[np.complex128]
    diagonal: NDArray[np.float64]
    row: int
    model: int
    support: _DensitySupport


@dataclass(frozen=True)
class RWAPrepared:
    """Model-only RWA payload, charged as one read-only allocation batch.

    models: packed joint energies, drive quadratures, bases and secular rates.
    positive, negative: (keys,tones,M,M) coefficients multiplying z and conj(z),
        respectively, after the omission-resolved Fourier-edge selection.
    diagonal: (keys,max_active_tones,M) GHz/gain² diagonal terms from
        RWACoefficients, independent of carrier phase.
    endpoint_maps: (keys,2,M,M) initial/final kick unitaries for constant
        dual-tone rows; identity maps for sampled rows, which do not use them.
    """

    models: Models
    positive: NDArray[np.complex128]
    negative: NDArray[np.complex128]
    diagonal: NDArray[np.float64]
    endpoint_maps: NDArray[np.complex128]

    @property
    def nbytes(self) -> int:
        return (
            sum(a.nbytes for a in self.models)
            + self.positive.nbytes
            + self.negative.nbytes
            + self.diagonal.nbytes
            + self.endpoint_maps.nbytes
        )


def _constant_endpoint_maps(
    model: SecularModel,
    table: SegmentTable,
    row: int,
    retention: RWARetention,
) -> NDArray[np.complex128]:
    """Sum all omitted-tone K at each row-local endpoint before exponentiating."""
    count = int(table.tones[row])
    frequencies = tuple(float(f) for f in table.tone_frequency_ghz[row, :count])
    mask = retained_mask(model, frequencies, retention)
    size = len(model.energies_ghz)
    kicks = np.zeros((2, size, size), dtype=np.complex128)
    for tone, frequency in enumerate(frequencies):
        drive = model.drives[table.tone_port[row, tone]]
        coefficients = kick_coefficients(
            model.energies_ghz,
            drive.in_phase_ghz,
            drive.quadrature_ghz,
            frequency,
            float(table.tone_amplitude[row, tone]),
            mask[tone],
        )
        theta = float(table.tone_theta_rad[row, tone])
        kicks[0] += endpoint_kick(coefficients, theta)
        # theta already includes the absolute row start. Idle after drive ends
        # belongs to another row, so kick and integration use the same duration.
        phase = theta + 2 * np.pi * frequency * table.duration_ns[row]
        kicks[1] += endpoint_kick(coefficients, phase)
    return np.stack((expm(1j * kicks[0]), expm(-1j * kicks[1])), dtype=np.complex128)


@njit(nogil=True, cache=True)
def _retained_graph(positive, negative):
    """Return exact Fourier adjacency (with diagonals) and its components."""
    size = positive.shape[1]
    adjacency = np.eye(size, dtype=np.bool_)
    for tone in range(len(positive)):
        for i in range(size):
            for j in range(size):
                if positive[tone, i, j] != 0 or negative[tone, i, j] != 0:
                    adjacency[i, j] = adjacency[j, i] = True
    connected = adjacency.copy()
    for k in range(size):
        for i in range(size):
            for j in range(size):
                connected[i, j] |= connected[i, k] and connected[k, j]
    return adjacency, connected


@njit(nogil=True, cache=True)
def _density_support(positive, negative, rho):
    """Close exact input support under the retained graph, never under the bath.

    The secular bath transfers only populations, all of which are retained.
    Hamiltonian components can create every within-component coherence. A
    nonzero cross-component rectangle evolves homogeneously and must be kept
    in full, including its reverse for the Hermitian commutator calculation.
    No amplitude threshold removes small coefficients or input coherences.
    """
    size = len(rho)
    adjacency, connected = _retained_graph(positive, negative)
    active = connected.copy()
    for i in range(size):
        for j in range(size):
            if rho[i, j] != 0:
                for a in range(size):
                    for b in range(size):
                        if connected[a, i] and connected[b, j]:
                            active[a, b] = active[b, a] = True
    coordinates = np.empty((int(np.sum(active)), 2), dtype=np.int64)
    edges = np.empty((int(np.sum(adjacency)), 2), dtype=np.int64)
    lookup = np.full((size, size), -1, dtype=np.int64)
    offsets = np.zeros(size + 1, dtype=np.int64)
    neighbors = np.empty(len(edges), dtype=np.int64)
    count, edge = 0, 0
    for i in range(size):
        for j in range(size):
            if active[i, j]:
                coordinates[count, 0], coordinates[count, 1] = i, j
                lookup[i, j] = count
                count += 1
            if adjacency[i, j]:
                edges[edge, 0], edges[edge, 1] = i, j
                neighbors[edge] = j
                edge += 1
        offsets[i + 1] = edge
    return _DensitySupport(coordinates, lookup, offsets, neighbors, edges)


@njit(nogil=True, cache=True)
def _rhs(time, state, context):
    models, segments, positive, negative, diagonal, row, model, support = context
    size = models.energies.shape[1]
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
        support.edges,
    )
    product = np.zeros_like(state)
    for entry in range(len(state)):
        i, j = support.coordinates[entry]
        for edge in range(support.offsets[i], support.offsets[i + 1]):
            k = support.neighbors[edge]
            product[entry] += h[i, k] * state[support.lookup[k, j]]
    derivative = np.empty_like(state)
    for entry in range(len(state)):
        i, j = support.coordinates[entry]
        reverse = support.lookup[j, i]
        derivative[entry] = (
            -2j * np.pi * (product[entry] - product[reverse].conjugate())
            - models.decay[model, i, j] * state[entry]
        )
    # Bath edges need not belong to the Hamiltonian graph. In particular an
    # isolated excited population must still feed a driven lower component.
    for i in range(size):
        for j in range(size):
            derivative[support.lookup[i, i]] += (
                models.rates[model, i, j] * state[support.lookup[j, j]]
            )
    return derivative


@njit(nogil=True, cache=True)
def _run(data, row_models, states, controls):
    models, segments, positive, negative, diagonal, endpoint_maps = data
    rows, indices = row_models
    for index in range(len(rows)):
        row, model = rows[index], indices[index]
        constant = segments.tones[row] == 2
        for tone in range(segments.tones[row]):
            env = segments.envelope[row, tone]
            constant = (
                constant
                and segments.sample_offsets[env] == segments.sample_offsets[env + 1]
            )
        if constant:
            initial_kick = endpoint_maps[model, 0]
            states[index] = initial_kick @ states[index] @ initial_kick.conj().T
        support = _density_support(positive[model], negative[model], states[index])
        context = _Context(
            models, segments, positive, negative, diagonal, row, model, support
        )
        initial = np.empty(len(support.coordinates), dtype=np.complex128)
        for entry in range(len(initial)):
            i, j = support.coordinates[entry]
            initial[entry] = states[index, i, j]
        ceiling = CONSTANT_STEP_NS if constant else STEP_NS
        row_controls = Tolerances(
            controls.rtol,
            controls.atol,
            min(controls.max_step, ceiling),
            controls.phase_budget,
            states.shape[1] ** 2,
        )
        result = integrate(
            _rhs,
            context,
            initial,
            float(segments.duration[row]),
            row_controls,
            0.1,
        )
        phases = np.exp(-2j * np.pi * models.energies[model] * segments.duration[row])
        states[index] = 0
        for entry in range(len(result)):
            i, j = support.coordinates[entry]
            states[index, i, j] = result[entry] * phases[i] * phases[j].conjugate()
        if constant:
            final_kick = endpoint_maps[model, 1]
            states[index] = final_kick @ states[index] @ final_kick.conj().T


class MultitoneRWAKernel:
    """Joint RWA C5 family with omission-driven Fourier retention.

    Admission requires two envelopes, fixed flux, each |f|>=0.05 GHz and
    budget >=1e-3. Constant pairs require distinct ports. Sampled candidates
    share an evolve-local spot check against full-drive DOP853 on a fixed
    coherent probe. A preparation memo can supply exact probe certificates.
    Within each same-shape group, select the row with the largest abs(gain)
    times envelope peak for each tone and probe each selected
    row once. No additional slope or curvature probes are selected. All probes
    must agree within 0.1 times the budget in maximum absolute density difference.
    Carrier spacing does not veto admission: close or equal carriers use the same joint
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
    Constant pairs also require epsilon_dom**2 * Omega_ret * row duration
    <=1.476e-4 cycle. epsilon_dom is max(abs(C/d)) over omitted nonzero Fourier
    coefficients, including diagonals; Omega_ret is twice max(abs(C)) over all
    retained tones/signs in cycles/ns. Both use the final resolved mask and
    gain-weighted coefficients. This endpoint-kick remainder proxy adds no
    2*pi factor and does not apply to sampled rows.

    Transfer retains half the measured 0.0116496 combination-channel boundary.
    Historical phase calibration used a 24-point two-tone, 16-level joint-model grid:
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
    certificates. Admitted constant rows use a 16 ns ceiling. Sampled rows
    use a 4 ns ceiling and no endpoint kick.
    Dispatch may prefer full-drive DOP853 even after numerical admission when
    restoring fast edges makes cold prepare/guard/apply more expensive. That
    choice does not change these omission bounds or the retained-dimension cap.
    No density reduction, dissipator change, clipping or normalization occurs.
    """

    name = "multitone_rwa"
    revision = (MULTITONE_GUARD_POLICY, "constant-dual-endpoint-kick-v1", SPOT_POLICY)
    guard_builds = False

    def guard(
        self, table: SegmentTable, rows: NDArray[np.int64], context: KernelContext
    ) -> NDArray[np.bool_]:
        """Check constant proxies and borrow each sampled group's own decision.

        rows index the supplied table. Sampled groups that intersect these
        rows in the complete geometry compare their probes once per work plan.
        Returns a row-aligned acceptance mask; numerical failures propagate.
        """
        constant = table.constant_envelopes(rows)
        accepted = np.zeros(len(rows), dtype=np.bool_)
        accepted[constant] = admit_multitone_rows(table, rows[constant], context)
        sampled = ~constant & (table.tones[rows] == 2)
        if not np.any(sampled):
            return accepted
        complete = context.work_plan.complete_table
        geometry = table if complete is None else complete
        admission = sampled_groups(geometry, context)
        membership = admission.by_row[rows + context.complete_row_offset]
        pending = sampled & (membership >= 0)
        needed = membership[pending]
        passed = np.zeros(len(admission.groups), dtype=np.bool_)
        for index in np.unique(needed):
            passed[index] = self._spot_check(geometry, admission.groups[index], context)
        accepted[pending] = passed[needed]
        return accepted

    def _spot_check(
        self, table: SegmentTable, group: SampledGroup, context: KernelContext
    ) -> bool:
        key = FactKey(
            (SPOT_POLICY, TableIdentity(table), context.budget, group.rows[0]), bool
        )

        def build() -> bool:
            rows = np.asarray(group.spot_rows, dtype=np.int64)
            certificates = self._certify_spots(table, rows, context)
            accepted = all(certificate.accepted for certificate in certificates)
            if context.capture is not None:
                context.capture.sampled_group(
                    group,
                    "accepted" if accepted else "rejected",
                    np.asarray([c.difference for c in certificates]),
                )
            return accepted

        return context.work_plan.fact(key, build)

    def _certify_spots(
        self, table: SegmentTable, rows: NDArray[np.int64], context: KernelContext
    ) -> tuple[ProbeCertificate, ...]:
        keys = self.prepare_keys(table, rows, context)
        memo = context.probe_memo
        if memo is None:
            return self._integrate_spots(keys, table, rows, context)

        probe_keys = [
            probe_key(table, int(row), context, key.tobytes())
            for row, key in zip(rows, keys, strict=True)
        ]
        certificates = {}
        misses = []
        for index, key in enumerate(probe_keys):
            certificate = memo.get(key)
            if certificate is None:
                misses.append(index)
            else:
                certificates[index] = certificate
        context.work_plan.routing.hit_probe_memo(self.name, len(certificates))
        if misses:
            integrated = self._integrate_spots(
                keys[misses], table, rows[misses], context
            )
            for index, certificate in zip(misses, integrated, strict=True):
                memo.put(probe_keys[index], certificate)
                certificates[index] = certificate
        return tuple(certificates[index] for index in range(len(rows)))

    def _integrate_spots(
        self,
        keys: NDArray[np.void],
        table: SegmentTable,
        rows: NDArray[np.int64],
        context: KernelContext,
    ) -> tuple[ProbeCertificate, ...]:
        prepared = self.prepare(keys, table, rows, context)
        size = prepared.models.energies.shape[1]
        direct = np.repeat(probe_density(size)[None], len(rows), axis=0)
        rwa = direct.copy()
        indices = np.arange(len(rows), dtype=np.int64)
        # These probes are independent endpoints, never a production chain.
        plan = DOP853Plan(
            np.full(len(table.timeline), "interaction"),
            np.full(len(table.timeline), -1, dtype=np.int64),
        )
        data = EndpointData(prepared.models, packed_segments(table, context.work_plan))
        routing = context.work_plan.routing
        with routing.measure(self.name, "probe_integration"):
            self.apply(
                prepared,
                indices,
                table,
                rows,
                rwa,
                ApplyContext(context.budget, work_plan=context.work_plan),
            )
            advance_endpoints(data, (rows, indices), direct, context.budget, plan)
        routing.admitted(self.name, len(rows))
        threshold = SPOT_POLICY[1] * context.budget
        return tuple(
            ProbeCertificate(float(difference), bool(difference <= threshold))
            for difference in np.max(np.abs(rwa - direct), axis=(1, 2))
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
                repr((self.revision, retention.edges, context.budget)).encode("ascii")
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
            * np.maximum(1.0, table.duration_ns[rows] / STEP_NS),
            dtype=np.float64,
        )

    def prepare(
        self,
        keys: NDArray[np.void],
        table: SegmentTable,
        rows: NDArray[np.int64],
        context: KernelContext,
    ) -> RWAPrepared:
        """Build retained Fourier edges and omitted-edge shifts for key-aligned rows.

        Uses the same compiled coefficient owner as rwa_coefficients. Apply
        scales its diagonal by the actual squared gain/envelope amplitude.
        """
        tolerances(context.budget)
        if len(keys) != len(rows):
            raise ValueError("RWA keys and representative rows must align")
        models = pack_models([table.model(context.source, row) for row in rows])
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
        size = models.energies.shape[1]
        endpoint_maps = np.broadcast_to(
            np.eye(size, dtype=np.complex128), (len(rows), 2, size, size)
        ).copy()
        for index, row in enumerate(rows):
            count = int(table.tones[row])
            model = table.model(context.source, row)
            retention = multitone_retention(table, int(row), context)
            coefficients = rwa_coefficients(
                model,
                tuple(int(v) for v in table.tone_port[row, :count]),
                tuple(float(v) for v in table.tone_frequency_ghz[row, :count]),
                retention,
                work_plan=context.work_plan,
            )
            positive[index, :count] = coefficients.positive
            negative[index, :count] = coefficients.negative
            diagonal[index, :count] = coefficients.diagonal
            if count == 2 and all(
                table.envelopes[e].kind == "constant"
                for e in table.tone_envelope[row, :count]
            ):
                endpoint_maps[index] = _constant_endpoint_maps(
                    model, table, int(row), retention
                )
        endpoint_maps.setflags(write=False)
        positive.setflags(write=False)
        negative.setflags(write=False)
        diagonal.setflags(write=False)
        return RWAPrepared(models, positive, negative, diagonal, endpoint_maps)

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

        Every row starts in its joint eigenbasis. Exact Fourier union components
        retain all populations and within-component coherences. Cross-component
        rectangles are skipped only when both input rectangles are exactly zero;
        arbitrary nonzero residuals retain the complete rectangle. The full bath
        transfers populations across components. The compressed integrator keeps
        the dense M² error-norm denominator, so removing zeros does not change
        adaptive controls. It integrates all retained tones and envelope-squared
        omitted-edge shifts through interaction_hamiltonian
        in a row-local interaction picture, then restores Schrödinger coordinates.
        Constant dual-tone rows receive the summed first-order omitted-edge
        kick before support selection and after returning to the lab frame.
        The kick uses the complement of resolved retention, including diagonal
        +/-carrier terms, not phase-graph pruning. Sampled rows receive no kick.
        theta is the phase at row start; offsets preserve pulse-local envelopes.
        The final kick and propagation share the exact row duration.
        Adaptive integration uses the ordinary budget tolerance and 0.1 ns seed,
        capped at 16 ns for the admitted dual constant envelopes.
        This is not a certified bound
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
                prepared.endpoint_maps,
            ),
            (rows, key_index),
            states,
            controls,
        )
        return np.zeros(len(rows), dtype=np.bool_)
