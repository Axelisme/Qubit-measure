"""Phase-covariant four-subsegment gates over the shared joint RWA graph."""

from dataclasses import dataclass
from hashlib import sha256

import numpy as np
import scipy.integrate
from numba import njit
from numpy.typing import NDArray
from scipy.linalg import expm

from .._envelope import sample_value
from ..kernel import ApplyContext, KernelContext, Prepared, SegmentTable
from ..secular import ModelSource
from ._rwa_graph import RWA_GRAPH_POLICY, PhaseGraph, phase_graph, pruning_admitted
from ._secular import coherence_decay, population_generator, split_step
from ._workload_facts import envelope_bounds
from .multitone_rwa import (
    RWACoefficients,
    admit_rows,
    admit_single_rows,
    interaction_hamiltonian,
    rwa_coefficients,
)

GATE_POLICY = ("joint-covariant-pruned-v1", RWA_GRAPH_POLICY, 4, 4.0, 1e-9, 1e-11)


@dataclass(frozen=True)
class GatePrepared:
    """Owned, read-only joint products aligned with representative keys.

    unitaries: (keys,4,M,M) canonical rotating-frame Hilbert propagators.
    populations/coherences: (keys,M,M) half-subsegment secular maps.
    charges: (keys,M,tones) integer dressed phase charges, not bare-qubit Z.
    """

    unitaries: NDArray[np.complex128]
    populations: NDArray[np.float64]
    coherences: NDArray[np.float64]
    charges: NDArray[np.int64]

    @property
    def nbytes(self) -> int:
        return sum(
            a.nbytes
            for a in (self.unitaries, self.populations, self.coherences, self.charges)
        )


def canonical_key(table: SegmentTable, row: int, source: ModelSource) -> bytes:
    """Hash canonical gate physics, omitting only covariant start and theta.

    Model identity, level, bath, ordered carriers, gains, waveform, offsets,
    duration and graph/integration policy distinguish numerical products.
    Guard admission is required before using this equivalence for playback.
    """
    count = table.tones[row]
    model = source.model(tuple(int(v) for v in table.level[row]))
    digest = sha256(repr((GATE_POLICY, model.identity, model.level)).encode())
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


def _common_support(table: SegmentTable, row: int) -> bool:
    # First release intentionally accepts complete, coextensive shaped pulses.
    # Partly overlapping, sliced or constant companions retain ordinary RWA.
    for tone in range(table.tones[row]):
        env = table.envelopes[table.tone_envelope[row, tone]]
        offset = table.tone_envelope_offset_ns[row, tone]
        if (
            env.kind != "sampled"
            or env.sample_ns[0] != offset
            or env.sample_ns[-1] != offset + table.duration_ns[row]
        ):
            return False
    return True


def _row_graph(table: SegmentTable, row: int, context: KernelContext) -> PhaseGraph:
    count = table.tones[row]
    return phase_graph(
        context.source.model(tuple(int(v) for v in table.level[row])),
        tuple(int(v) for v in table.tone_port[row, :count]),
        tuple(float(v) for v in table.tone_frequency_ghz[row, :count]),
        context.work_plan,
    )


def _admitted(
    table: SegmentTable,
    rows: NDArray[np.int64],
    context: KernelContext,
    *,
    single_tone: bool,
) -> NDArray[np.bool_]:
    accepted = (admit_single_rows if single_tone else admit_rows)(table, rows, context)
    peaks = envelope_bounds(table, context.work_plan).peaks
    for index in np.flatnonzero(accepted):
        row = int(rows[index])
        if not single_tone and not _common_support(table, row):
            accepted[index] = False
            continue
        graph = _row_graph(table, row, context)
        count = table.tones[row]
        amplitudes = (
            np.abs(table.tone_amplitude[row, :count])
            * peaks[table.tone_envelope[row, :count]]
        )
        model = context.source.model(tuple(int(v) for v in table.level[row]))
        gamma = max(
            float(np.max(coherence_decay(model))),
            float(np.max(model.rates_per_ns.sum(axis=0))),
        )
        # Diagonal shifts commute with secular D; only off-diagonal drive
        # contributes to this D-U-D splitting proxy.
        strength = sum(
            amplitudes[k] * 2 * np.linalg.norm(graph.positive[k], 2)
            for k in range(count)
        )
        split = table.duration_ns[row] ** 3 * gamma * strength**2 / 16
        accepted[index] = (
            pruning_admitted(
                graph, amplitudes, float(table.duration_ns[row]), context.budget
            )
            and split <= 0.01
        )
    return accepted


def _build_unitaries(
    table: SegmentTable, row: int, context: KernelContext, graph: PhaseGraph
) -> NDArray[np.complex128]:
    model = context.source.model(tuple(int(v) for v in table.level[row]))
    count = table.tones[row]
    size = len(model.energies_ghz)
    energy = model.energies_ghz + graph.charges @ table.tone_frequency_ghz[row, :count]
    energy -= energy[0]

    original = rwa_coefficients(
        model,
        tuple(int(v) for v in table.tone_port[row, :count]),
        tuple(float(v) for v in table.tone_frequency_ghz[row, :count]),
    )
    coefficients = RWACoefficients(
        graph.positive,
        np.ascontiguousarray(graph.positive.conj().transpose(0, 2, 1)),
        original.diagonal,
    )

    def rhs(time, state):
        amplitudes = np.empty(count, dtype=np.complex128)
        for k in range(count):
            env = table.envelopes[table.tone_envelope[row, k]]
            amplitude = sample_value(
                time + table.tone_envelope_offset_ns[row, k], env.sample_ns, env.samples
            )
            amplitudes[k] = table.tone_amplitude[row, k] * amplitude
        h = interaction_hamiltonian(time, energy, coefficients, amplitudes)
        return (-2j * np.pi * h @ state.reshape(size, size)).ravel()

    result = []
    for start, end in zip(
        np.linspace(0, table.duration_ns[row], 5)[:-1],
        np.linspace(0, table.duration_ns[row], 5)[1:],
        strict=True,
    ):
        solution = scipy.integrate.solve_ivp(
            rhs,
            (start, end),
            np.eye(size, dtype=complex).ravel(),
            method="DOP853",
            rtol=1e-9,
            atol=1e-11,
            max_step=4.0,
        )
        if not solution.success:
            raise RuntimeError(f"joint gate integration failed: {solution.message}")
        endpoint = solution.y[:, -1].reshape(size, size)
        endpoint *= np.exp(-2j * np.pi * energy * end)[:, None]
        endpoint *= np.exp(2j * np.pi * energy * start)[None, :]
        result.append(endpoint)
    return np.array(result)


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


class CovariantGateKernel:
    """Shared four-U builder and playback for named single/multi-tone families.

    Numerical guard checks shared calibrated RWA, bounded phase-graph pruning
    and secular splitting. Multi-tone gates additionally require coextensive
    sampled support. Single-tone gates allow slices and final sample hold.
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
        return np.array(
            [
                np.void(
                    sha256(
                        canonical_key(table, int(row), context.source)
                        + repr(context.budget).encode()
                    ).digest()
                )
                for row in rows
            ],
            dtype="V32",
        )

    def cost(self, table: SegmentTable, rows: NDArray[np.int64]) -> NDArray[np.float64]:
        """Four joint matrix actions per admitted row, independent of duration."""
        del table
        return np.full(len(rows), 4.0)

    def prepare(
        self,
        keys: NDArray[np.void],
        table: SegmentTable,
        rows: NDArray[np.int64],
        context: KernelContext,
    ) -> Prepared:
        """Build four U and half-step D per representative; failures propagate."""
        if (
            len(keys) != len(rows)
            or not _admitted(table, rows, context, single_tone=self.single_tone).all()
        ):
            raise ValueError(
                "joint gate preparation requires admitted key-aligned rows"
            )
        unitaries, populations, coherences, charges = [], [], [], []
        width = int(np.max(table.tones[rows], initial=0))
        for row in rows:
            graph = _row_graph(table, int(row), context)
            model = context.source.model(tuple(int(v) for v in table.level[row]))
            unitaries.append(_build_unitaries(table, int(row), context, graph))
            half = table.duration_ns[row] / 8
            populations.append(expm(population_generator(model.rates_per_ns) * half))
            coherences.append(np.exp(-coherence_decay(model) * half))
            charges.append(
                np.pad(graph.charges, ((0, 0), (0, width - graph.charges.shape[1])))
            )
        arrays = [np.array(a) for a in (unitaries, populations, coherences, charges)]
        for a in arrays:
            a.setflags(write=False)
        return GatePrepared(*arrays)

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
        """Play four D-U-D steps and restore lab/DC frame at every endpoint.

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
        _play(
            (
                prepared.unitaries,
                prepared.populations,
                prepared.coherences,
                prepared.charges,
            ),
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
