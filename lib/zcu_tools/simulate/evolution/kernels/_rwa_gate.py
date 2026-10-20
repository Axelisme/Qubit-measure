"""Phase-covariant, adaptively split gates over the shared joint RWA graph."""

from dataclasses import dataclass
from hashlib import sha256
from typing import NamedTuple

import numpy as np
from numba import njit
from numpy.typing import NDArray
from scipy.linalg import expm

from .._envelope import sample_value
from ..kernel import ApplyContext, KernelContext, Prepared, SegmentTable
from ..workplan import FactKey
from ._dop853 import Tolerances, integrate, tolerances
from ._rwa_graph import RWA_GRAPH_POLICY, PhaseGraph, phase_graph, pruning_admitted
from ._secular import coherence_decay, population_generator, split_step
from ._workload_facts import TableIdentity, envelope_bounds, fixed_flux
from .multitone_rwa import (
    MULTITONE_RETENTION_POLICY,
    RWA_POLICY,
    RWACoefficients,
    RWARetention,
    admit_rows,
    admit_single_rows,
    interaction_hamiltonian,
    multitone_retention,
    retained_mask,
    rwa_coefficients,
)

_MIN_SUBSEGMENTS = 4
_SPLIT_ALLOWANCE = 0.01
_MAX_STEP_NS = 4.0
_RTOL = 1e-9
_ATOL = 1e-11

GATE_POLICY = (
    "joint-covariant-adaptive-v2",
    RWA_GRAPH_POLICY,
    _MIN_SUBSEGMENTS,
    _SPLIT_ALLOWANCE,
    _MAX_STEP_NS,
    _RTOL,
    _ATOL,
)

SINGLE_GATE_POLICY = (
    "single-gate-omission-v1",
    GATE_POLICY,
    MULTITONE_RETENTION_POLICY,
)


@dataclass(frozen=True)
class GatePrepared:
    """Owned, read-only joint products aligned with representative keys.

    unitaries: (keys,max(segments),M,M) canonical Hilbert propagators, zero padded.
    populations/coherences: (keys,M,M) half-subsegment secular maps.
    charges: (keys,M,tones) integer dressed phase charges, not bare-qubit Z.
    """

    unitaries: NDArray[np.complex128]
    populations: NDArray[np.float64]
    coherences: NDArray[np.float64]
    charges: NDArray[np.int64]
    segments: NDArray[np.int64]
    """Number of active Strang subsegments for each representative key."""

    @property
    def nbytes(self) -> int:
        return sum(
            a.nbytes
            for a in (
                self.unitaries,
                self.populations,
                self.coherences,
                self.charges,
                self.segments,
            )
        )


def canonical_key(table: SegmentTable, row: int, context: KernelContext) -> bytes:
    """Hash canonical gate physics, omitting only covariant start and theta.

    Model identity, level, bath, ordered carriers, gains, waveform, offsets,
    duration, final retained mask and graph/integration policy distinguish products.
    Guard admission is required before using this equivalence for playback.
    """
    count = table.tones[row]
    model = context.source.model(tuple(int(v) for v in table.level[row]))
    retention = _row_retention(table, row, context)
    policy = SINGLE_GATE_POLICY if retention is not None else GATE_POLICY
    digest = sha256(repr((policy, model.identity, model.level)).encode())
    if retention is not None:
        digest.update(
            retained_mask(
                model, tuple(table.tone_frequency_ghz[row, :count]), retention
            ).tobytes()
        )
    # Identity must describe the complete source; explicit bath bytes also make
    # the numerical product's dependence visible in its canonical key.
    for a in (model.energies_ghz, model.rates_per_ns, model.dephasing_per_ns):
        digest.update(a.tobytes())
    digest.update(np.array([count, table.duration_ns[row]], dtype="<f8").tobytes())
    for a in (
        table.tone_port,
        table.tone_frequency_ghz,
        table.tone_amplitude,
        table.tone_envelope_offset_ns,
    ):
        digest.update(a[row, :count].astype("<f8").tobytes())
    for env in table.tone_envelope[row, :count]:
        digest.update(table.envelopes[env].identity.encode())
    return digest.digest()


def canonical_keys(table: SegmentTable, context: KernelContext) -> tuple[bytes, ...]:
    """Return table-aligned physics hashes, with b'' for undriven rows.

    Planner, guards and preparation share these unchanged canonical hashes.
    Exact row bytes group equal physical inputs before hashing representatives.
    Start/theta remain apply-only, while flux motion is checked separately on
    every row. Table and source ownership scope the fact to this evolve.
    """
    key = FactKey(("covariant-gate-keys", TableIdentity(table), context.source), tuple)

    def build() -> tuple[bytes, ...]:
        rows = np.flatnonzero(table.tones > 0)
        result = [b""] * len(table.timeline)
        if not len(rows):
            return tuple(result)
        # Compare bytes rather than floating values to preserve the original
        # hash distinctions, including signed zeros and integer field widths.
        physical = np.concatenate(
            [
                np.ascontiguousarray(array[rows]).reshape(len(rows), -1).view(np.uint8)
                for array in (
                    table.level,
                    table.tones,
                    table.duration_ns,
                    table.tone_port,
                    table.tone_frequency_ghz,
                    table.tone_amplitude,
                    table.tone_envelope,
                    table.tone_envelope_offset_ns,
                )
            ],
            axis=1,
        )
        records = physical.view(np.dtype((np.void, physical.shape[1]))).ravel()
        _, representatives, inverse = np.unique(
            records, return_index=True, return_inverse=True
        )
        hashes = [canonical_key(table, int(rows[i]), context) for i in representatives]
        for row, index in zip(rows, inverse, strict=True):
            result[row] = hashes[index]
        return tuple(result)

    return context.work_plan.fact(key, build)


def _row_retention(
    table: SegmentTable, row: int, context: KernelContext
) -> RWARetention | None:
    """Keep simultaneous gate policy unchanged; resolve only single shaped rows."""
    if (
        table.tones[row] == 1
        and table.envelopes[table.tone_envelope[row, 0]].kind == "sampled"
    ):
        return multitone_retention(table, row, context)
    return None


def _row_graph(table: SegmentTable, row: int, context: KernelContext) -> PhaseGraph:
    count = table.tones[row]
    return phase_graph(
        context.source.model(tuple(int(v) for v in table.level[row])),
        tuple(int(v) for v in table.tone_port[row, :count]),
        tuple(float(v) for v in table.tone_frequency_ghz[row, :count]),
        context.work_plan,
        _row_retention(table, row, context),
    )


def _admitted(
    table: SegmentTable,
    rows: NDArray[np.int64],
    context: KernelContext,
    *,
    single_tone: bool,
) -> NDArray[np.bool_]:
    tolerances(context.budget)
    accepted = np.zeros(len(rows), dtype=np.bool_)
    physics = canonical_keys(table, context)
    for index, row in enumerate(rows):
        # Waveform support is not covariant: equal playback keys can have
        # different flux motion, so this check must precede every fact lookup.
        if not fixed_flux(table, int(row)):
            continue
        key = FactKey(
            (
                "covariant-gate-admission",
                GATE_POLICY,
                RWA_POLICY,
                single_tone,
                physics[row],
                context.budget,
            ),
            bool,
        )

        def build(row=int(row)) -> bool:
            return bool(
                _admit_physics(
                    table,
                    np.array([row], dtype=np.int64),
                    context,
                    single_tone=single_tone,
                )[0]
            )

        accepted[index] = context.work_plan.fact(key, build)
    return accepted


def _admit_physics(
    table: SegmentTable,
    rows: NDArray[np.int64],
    context: KernelContext,
    *,
    single_tone: bool,
) -> NDArray[np.bool_]:
    """Evaluate RWA and phase-graph validity once per physics key."""
    accepted = (admit_single_rows if single_tone else admit_rows)(table, rows, context)
    peaks = envelope_bounds(table, context.work_plan).peaks
    for index in np.flatnonzero(accepted):
        row = int(rows[index])
        # SegmentTable owns finite playback support. As for single-tone gates,
        # sample grids may extend beyond a row or end before its final hold.
        # The builder samples only the row interval with the same offsets and
        # interpolation as direct integration; no waveform rewrite is needed.
        graph = _row_graph(table, row, context)
        count = table.tones[row]
        amplitudes = (
            np.abs(table.tone_amplitude[row, :count])
            * peaks[table.tone_envelope[row, :count]]
        )
        accepted[index] = pruning_admitted(
            graph, amplitudes, float(table.duration_ns[row]), context.budget
        )
    return accepted


def gate_segments(table: SegmentTable, row: int, context: KernelContext) -> int:
    """Return the least n >= 4 with duration**3*Gamma*V**2/n**2 <= 0.01.

    This density-free, calibrated Strang proxy uses retained off-diagonal drive
    only: diagonal shifts commute with the secular dissipator. Numerical RWA
    and pruning admission remain separate. The result is shared per physics key.
    """
    key = FactKey(("covariant-gate-segments", canonical_keys(table, context)[row]), int)

    def build() -> int:
        graph = _row_graph(table, row, context)
        model = context.source.model(tuple(int(v) for v in table.level[row]))
        peaks = envelope_bounds(table, context.work_plan).peaks
        amplitudes = (
            np.abs(table.tone_amplitude[row, : table.tones[row]])
            * peaks[table.tone_envelope[row, : table.tones[row]]]
        )
        gamma = max(
            float(np.max(coherence_decay(model))),
            float(np.max(model.rates_per_ns.sum(axis=0))),
        )
        strength = sum(
            amplitudes[k] * 2 * np.linalg.norm(graph.positive[k], 2)
            for k in range(table.tones[row])
        )
        return max(
            _MIN_SUBSEGMENTS,
            int(
                np.ceil(
                    np.sqrt(
                        table.duration_ns[row] ** 3
                        * gamma
                        * strength**2
                        / _SPLIT_ALLOWANCE
                    )
                )
            ),
        )

    return context.work_plan.fact(key, build)


class _SingleContext(NamedTuple):
    """Joint interaction data; start_ns shifts solver time to row-local ns."""

    energy: NDArray[np.float64]
    coefficients: RWACoefficients
    times: NDArray[np.float64]
    samples: NDArray[np.complex128]
    gain: float
    offset_ns: float
    start_ns: float


@njit(nogil=True, cache=True)
def _single_rhs(time, state, context):
    # The solver restarts at each subsegment; the Hamiltonian and envelope use
    # row-local time, not solver-local time or absolute carrier start.
    row_time = time + context.start_ns
    amplitudes = np.empty(1, dtype=np.complex128)
    amplitudes[0] = context.gain * sample_value(
        row_time + context.offset_ns, context.times, context.samples
    )
    h = interaction_hamiltonian(
        row_time, context.energy, context.coefficients, amplitudes
    )
    size = len(context.energy)
    return (-2j * np.pi * h @ state.reshape(size, size)).ravel()


@njit(nogil=True, cache=True)
def _single_unitaries(context, duration_ns):
    size = len(context.energy)
    result = np.empty((_MIN_SUBSEGMENTS, size, size), dtype=np.complex128)
    boundaries = np.linspace(0.0, duration_ns, _MIN_SUBSEGMENTS + 1)
    for segment in range(_MIN_SUBSEGMENTS):
        start, end = boundaries[segment], boundaries[segment + 1]
        interval = _SingleContext(
            context.energy,
            context.coefficients,
            context.times,
            context.samples,
            context.gain,
            context.offset_ns,
            start,
        )
        state = integrate(
            _single_rhs,
            interval,
            np.eye(size, dtype=np.complex128).ravel(),
            end - start,
            Tolerances(_RTOL, _ATOL, _MAX_STEP_NS),
            1.0,
        )
        endpoint = state.reshape(size, size)
        endpoint *= np.exp(-2j * np.pi * context.energy * end)[:, None]
        endpoint *= np.exp(2j * np.pi * context.energy * start)[None, :]
        result[segment] = endpoint
    return result


@njit(nogil=True, cache=True)
def _adaptive_unitaries(context, duration_ns, segments):
    size = len(context.energy)
    result = np.empty((segments, size, size), dtype=np.complex128)
    boundaries = np.linspace(0.0, duration_ns, segments + 1)
    for segment in range(segments):
        start, end = boundaries[segment], boundaries[segment + 1]
        interval = _SingleContext(
            context.energy,
            context.coefficients,
            context.times,
            context.samples,
            context.gain,
            context.offset_ns,
            start,
        )
        state = integrate(
            _single_rhs,
            interval,
            np.eye(size, dtype=np.complex128).ravel(),
            end - start,
            Tolerances(_RTOL, _ATOL, _MAX_STEP_NS),
            1.0,
        )
        endpoint = state.reshape(size, size)
        endpoint *= np.exp(-2j * np.pi * context.energy * end)[:, None]
        endpoint *= np.exp(2j * np.pi * context.energy * start)[None, :]
        result[segment] = endpoint
    return result


class _MultiContext(NamedTuple):
    """Tone-aligned samples; lengths exclude padding, start_ns is row-local."""

    energy: NDArray[np.float64]
    coefficients: RWACoefficients
    times: NDArray[np.float64]
    samples: NDArray[np.complex128]
    lengths: NDArray[np.int64]
    gains: NDArray[np.float64]
    offsets: NDArray[np.float64]
    start_ns: float


@njit(nogil=True, cache=True)
def _multi_rhs(time, state, context):
    row_time = time + context.start_ns
    amplitudes = np.empty(len(context.gains), dtype=np.complex128)
    for tone in range(len(context.gains)):
        length = context.lengths[tone]
        amplitudes[tone] = context.gains[tone] * sample_value(
            row_time + context.offsets[tone],
            context.times[tone, :length],
            context.samples[tone, :length],
        )
    h = interaction_hamiltonian(
        row_time, context.energy, context.coefficients, amplitudes
    )
    size = len(context.energy)
    return (-2j * np.pi * h @ state.reshape(size, size)).ravel()


@njit(nogil=True, cache=True)
def _multi_unitaries(context, duration_ns, segments):
    size = len(context.energy)
    result = np.empty((segments, size, size), dtype=np.complex128)
    boundaries = np.linspace(0.0, duration_ns, segments + 1)
    for segment in range(segments):
        start, end = boundaries[segment], boundaries[segment + 1]
        interval = _MultiContext(
            context.energy,
            context.coefficients,
            context.times,
            context.samples,
            context.lengths,
            context.gains,
            context.offsets,
            start,
        )
        state = integrate(
            _multi_rhs,
            interval,
            np.eye(size, dtype=np.complex128).ravel(),
            end - start,
            Tolerances(_RTOL, _ATOL, _MAX_STEP_NS),
            1.0,
        )
        endpoint = state.reshape(size, size)
        endpoint *= np.exp(-2j * np.pi * context.energy * end)[:, None]
        endpoint *= np.exp(2j * np.pi * context.energy * start)[None, :]
        result[segment] = endpoint
    return result


def _build_unitaries(
    table: SegmentTable,
    row: int,
    context: KernelContext,
    graph: PhaseGraph,
    segments: int,
) -> NDArray[np.complex128]:
    model = context.source.model(tuple(int(v) for v in table.level[row]))
    count = table.tones[row]
    energy = model.energies_ghz + graph.charges @ table.tone_frequency_ghz[row, :count]
    energy -= energy[0]

    original = rwa_coefficients(
        model,
        tuple(int(v) for v in table.tone_port[row, :count]),
        tuple(float(v) for v in table.tone_frequency_ghz[row, :count]),
        _row_retention(table, row, context),
    )
    coefficients = RWACoefficients(
        graph.positive,
        np.ascontiguousarray(graph.positive.conj().transpose(0, 2, 1)),
        original.diagonal,
    )

    if count == 1:
        env = table.envelopes[table.tone_envelope[row, 0]]
        single_context = _SingleContext(
            energy,
            coefficients,
            env.sample_ns,
            env.samples,
            float(table.tone_amplitude[row, 0]),
            float(table.tone_envelope_offset_ns[row, 0]),
            0.0,
        )
        duration = float(table.duration_ns[row])
        if segments == _MIN_SUBSEGMENTS:
            return _single_unitaries(single_context, duration)
        return _adaptive_unitaries(single_context, duration, segments)

    envelopes = [table.envelopes[index] for index in table.tone_envelope[row, :count]]
    lengths = np.array([len(env.samples) for env in envelopes], dtype=np.int64)
    width = max(1, int(lengths.max()))
    times = np.zeros((count, width), dtype=np.float64)
    samples = np.zeros((count, width), dtype=np.complex128)
    for tone, env in enumerate(envelopes):
        times[tone, : lengths[tone]] = env.sample_ns
        samples[tone, : lengths[tone]] = env.samples
    multi_context = _MultiContext(
        energy,
        coefficients,
        times,
        samples,
        lengths,
        table.tone_amplitude[row, :count].copy(),
        table.tone_envelope_offset_ns[row, :count].copy(),
        0.0,
    )
    return _multi_unitaries(multi_context, float(table.duration_ns[row]), segments)


@njit(nogil=True, cache=True)
def _play(
    data,
    keys,
    playback,
    states,
):
    unitaries, populations, coherences, charges = data
    theta, frequency, duration, tones = playback
    for row in range(len(states)):
        key = keys[row]
        count = tones[row]
        size = states.shape[-1]
        start = np.empty(size, dtype=np.complex128)
        end = np.empty(size, dtype=np.complex128)
        for i in range(size):
            initial, final = 0.0, 0.0
            for k in range(count):
                initial += charges[key, i, k] * theta[row, k]
                final += charges[key, i, k] * (
                    theta[row, k] + 2 * np.pi * frequency[row, k] * duration[row]
                )
            start[i] = np.exp(1j * initial)
            end[i] = np.exp(1j * final)
        rho = states[row] * start.conj()[:, None] * start[None, :]
        for segment in range(4):
            rho = split_step(
                rho, unitaries[key, segment], populations[key], coherences[key]
            )
        states[row] = rho * end[:, None] * end.conj()[None, :]


@njit(nogil=True, cache=True)
def _adaptive_play(
    data,
    keys,
    playback,
    states,
):
    unitaries, populations, coherences, charges, segments = data
    theta, frequency, duration, tones = playback
    for row in range(len(states)):
        key = keys[row]
        count = tones[row]
        size = states.shape[-1]
        start = np.empty(size, dtype=np.complex128)
        end = np.empty(size, dtype=np.complex128)
        for i in range(size):
            initial, final = 0.0, 0.0
            for k in range(count):
                initial += charges[key, i, k] * theta[row, k]
                final += charges[key, i, k] * (
                    theta[row, k] + 2 * np.pi * frequency[row, k] * duration[row]
                )
            start[i] = np.exp(1j * initial)
            end[i] = np.exp(1j * final)
        rho = states[row] * start.conj()[:, None] * start[None, :]
        for segment in range(segments[key]):
            rho = split_step(
                rho, unitaries[key, segment], populations[key], coherences[key]
            )
        states[row] = rho * end[:, None] * end.conj()[None, :]


class CovariantGateKernel:
    """Shared adaptive U builder and playback for single/multi-tone families.

    Numerical guard checks shared calibrated RWA and bounded phase-graph pruning.
    Preparation refines secular splitting without relaxing its proxy threshold.
    Both families allow sampled slices and final sample hold within row support.
    All gates compile Hilbert integration through the shared DOP853 solver.
    Dispatch/family wrappers own acquire-wide cost and reuse decisions.
    """

    revision = GATE_POLICY[0]
    guard_builds = False
    single_tone = False

    def guard(
        self, table: SegmentTable, rows: NDArray[np.int64], context: KernelContext
    ) -> NDArray[np.bool_]:
        """Return numerical admissions without reading density or cache warmth."""
        return _admitted(table, rows, context, single_tone=self.single_tone)

    def prepare_keys(
        self, table: SegmentTable, rows: NDArray[np.int64], context: KernelContext
    ) -> NDArray[np.void]:
        """Hash full canonical physics, policy and budget; omit only start/theta.

        These are candidate equivalence keys. Only guard-admitted rows can build
        or consume a phase-reduced product. Source identity covers quadratures,
        dressed basis and all model parameters; bath bytes are also explicit.
        """
        by_physics: dict[bytes, bytes] = {}
        keys = []
        canonical = canonical_keys(table, context)
        for row in rows:
            physics = canonical[row]
            if physics not in by_physics:
                segments = gate_segments(table, int(row), context)
                by_physics[physics] = sha256(
                    physics + repr((context.budget, segments)).encode()
                ).digest()
            keys.append(np.void(by_physics[physics]))
        return np.array(keys, dtype="V32")

    def cost(self, table: SegmentTable, rows: NDArray[np.int64]) -> NDArray[np.float64]:
        """Four-step baseline work; the cold planner also charges adaptive n."""
        del table
        return np.full(len(rows), float(_MIN_SUBSEGMENTS))

    def prepare(
        self,
        keys: NDArray[np.void],
        table: SegmentTable,
        rows: NDArray[np.int64],
        context: KernelContext,
    ) -> Prepared:
        """Build n U and half-step D per representative; failures propagate."""
        if (
            len(keys) != len(rows)
            or not _admitted(table, rows, context, single_tone=self.single_tone).all()
        ):
            raise ValueError(
                "joint gate preparation requires admitted key-aligned rows"
            )
        segments = np.array(
            [gate_segments(table, int(row), context) for row in rows], dtype=np.int64
        )
        unitaries, populations, coherences, charges = [], [], [], []
        width = int(np.max(table.tones[rows], initial=0))
        for row, count in zip(rows, segments, strict=True):
            graph = _row_graph(table, int(row), context)
            model = context.source.model(tuple(int(v) for v in table.level[row]))
            u = _build_unitaries(table, int(row), context, graph, int(count))
            padding = int(segments.max() - count)
            unitaries.append(
                np.pad(u, ((0, padding), (0, 0), (0, 0))) if padding else u
            )
            half = table.duration_ns[row] / (2 * count)
            populations.append(expm(population_generator(model.rates_per_ns) * half))
            coherences.append(np.exp(-coherence_decay(model) * half))
            charges.append(
                np.pad(graph.charges, ((0, 0), (0, width - graph.charges.shape[1])))
            )
        prepared = GatePrepared(
            np.array(unitaries, dtype=np.complex128),
            np.array(populations, dtype=np.float64),
            np.array(coherences, dtype=np.float64),
            np.array(charges, dtype=np.int64),
            segments,
        )
        for a in (
            prepared.unitaries,
            prepared.populations,
            prepared.coherences,
            prepared.charges,
            prepared.segments,
        ):
            a.setflags(write=False)
        return prepared

    def admit_prepared(
        self,
        prepared: Prepared,
        table: SegmentTable,
        rows: NDArray[np.int64],
        context: KernelContext,
    ) -> None:
        """No guard-built payload; ordinary guards always recheck current rows."""

    def guard_prepared(
        self, table: SegmentTable, rows: NDArray[np.int64], context: KernelContext
    ) -> None:
        """Admission builds graph facts, not resident numerical propagators."""

    def bind_input(
        self,
        prepared: Prepared,
        key_index: int,
        table: SegmentTable,
        rows: NDArray[np.int64],
        source_initial: NDArray[np.complex128],
        context: KernelContext,
    ) -> None:
        """No density-dependent product; arbitrary joint input is supported."""

    def apply(
        self,
        prepared: Prepared,
        key_index: NDArray[np.int64],
        table: SegmentTable,
        rows: NDArray[np.int64],
        states: NDArray[np.complex128],
        context: ApplyContext,
    ) -> NDArray[np.bool_]:
        """Play n D-U-D steps and restore lab/DC frame at every endpoint.

        theta is the actual row-start phase, including absolute carrier origin.
        No clipping, trace correction, independent-qubit product or retry occurs.
        Invalid payload, row plan, state shape or key index raises ValueError.
        """
        if not isinstance(prepared, GatePrepared) or context.plan is not None:
            raise ValueError("joint gate requires its own payload and no row plan")
        size = prepared.unitaries.shape[-1]
        if (
            key_index.shape != rows.shape
            or states.shape != (len(rows), size, size)
            or np.any(key_index < 0)
            or np.any(key_index >= len(prepared.unitaries))
        ):
            raise ValueError("joint gate state/index shape mismatch")
        data = (
            prepared.unitaries,
            prepared.populations,
            prepared.coherences,
            prepared.charges,
        )
        if prepared.unitaries.shape[1] == 4:
            _play(
                data,
                key_index,
                (
                    table.tone_theta_rad[rows],
                    table.tone_frequency_ghz[rows],
                    table.duration_ns[rows],
                    table.tones[rows],
                ),
                states,
            )
            return np.zeros(len(rows), dtype=bool)
        _adaptive_play(
            (*data, prepared.segments),
            key_index,
            (
                table.tone_theta_rad[rows],
                table.tone_frequency_ghz[rows],
                table.duration_ns[rows],
                table.tones[rows],
            ),
            states,
        )
        return np.zeros(len(rows), dtype=bool)
