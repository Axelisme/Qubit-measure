"""Immutable numerical facts shared across one evolve's kernel calls."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numba import njit
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ._rwa_graph import PhaseGraph

from ..kernel import SegmentTable
from ..secular import SecularModel
from ..workplan import FactKey, WorkPlan
from ._dop853_data import Models, Segments, pack_models, pack_segments
from ._secular import secular_operators


@dataclass(frozen=True, eq=False)
class TableIdentity:
    """Object-identity fact key retaining its SegmentTable for the plan lifetime.

    table is the exact row-indexed input, not a structural copy. Retaining it
    prevents a later representative from reusing its object id.
    """

    table: SegmentTable

    def __hash__(self) -> int:
        return id(self.table)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TableIdentity) and self.table is other.table


@dataclass(frozen=True)
class EnvelopeBounds:
    """Peak magnitude, maximum linear slope per ns and real-sample flag."""

    peaks: NDArray[np.float64]
    slopes: NDArray[np.float64]
    real: NDArray[np.bool_]


def envelope_bounds(table: SegmentTable, work_plan: WorkPlan) -> EnvelopeBounds:
    """Resolve envelope bounds once per table, including prefix representatives."""
    key = FactKey(("envelope-bounds", TableIdentity(table)), EnvelopeBounds)

    def build() -> EnvelopeBounds:
        peaks = np.ones(len(table.envelopes))
        slopes = np.zeros(len(table.envelopes))
        real = np.array(
            [np.all(e.samples.imag == 0) for e in table.envelopes], dtype=np.bool_
        )
        for index, envelope in enumerate(table.envelopes):
            if envelope.kind == "sampled":
                peaks[index] = np.max(np.abs(envelope.samples))
                if len(envelope.samples) > 1:
                    slopes[index] = np.max(
                        np.abs(np.diff(envelope.samples)) / np.diff(envelope.sample_ns)
                    )
        return EnvelopeBounds(peaks, slopes, real)

    return work_plan.fact(key, build)


@dataclass(frozen=True)
class QuadratureNorms:
    """Separate spectral norms in GHz/gain; consumers choose sum or maximum."""

    in_phase: float
    quadrature: float


def quadrature_norms(
    model: SecularModel, port: int, work_plan: WorkPlan
) -> QuadratureNorms:
    """Resolve both port norms once per full model identity and level.

    RWA uses their sum; shaped admission uses their maximum. Keeping both
    scalars avoids making those distinct strength policies interchangeable.
    """
    key = FactKey(
        ("model-quadrature-norms", model.identity, model.level, port), QuadratureNorms
    )

    def build() -> QuadratureNorms:
        drive = model.drives[port]
        return QuadratureNorms(
            float(np.linalg.norm(drive.in_phase_ghz, 2)),
            float(np.linalg.norm(drive.quadrature_ghz, 2)),
        )

    return work_plan.fact(key, build)


def packed_model(model: SecularModel, work_plan: WorkPlan) -> Models:
    """Pack one immutable model once per complete identity and quantized level."""
    key = FactKey(("packed-model", model.identity, model.level), Models)
    return work_plan.fact(key, lambda: pack_models([model]))


@dataclass(frozen=True)
class SecularBath:
    """Full population generator and pairwise coherence decay, both in /ns."""

    population: NDArray[np.float64]
    decay: NDArray[np.float64]


def secular_bath(model: SecularModel, work_plan: WorkPlan) -> SecularBath:
    """Resolve the unmodified secular bath once per full model identity/level."""
    key = FactKey(("secular-bath", model.identity, model.level), SecularBath)
    return work_plan.fact(key, lambda: SecularBath(*secular_operators(model)))


@dataclass(frozen=True)
class RWATopology:
    """Connected components and closed Hermitian coordinates of a phase graph.

    labels assigns each dressed state to a component in minimum-index order.
    components lists those state indices. a/b/kind enumerate populations, then
    within-component real/imaginary coherence pairs (kind 0/1/2 respectively).
    identity carries the graph's model, levels, ports, policy and final mask.
    """

    identity: Hashable
    labels: NDArray[np.int64]
    components: tuple[NDArray[np.int64], ...]
    a: NDArray[np.int64]
    b: NDArray[np.int64]
    kind: NDArray[np.int64]


@njit(nogil=True, cache=True)
def _component_labels(adjacency):
    # Minimum-vertex traversal matches scipy's undirected component ordering.
    size = len(adjacency)
    labels = np.full(size, -1, dtype=np.int64)
    queue = np.empty(size, dtype=np.int64)
    component = 0
    for root in range(size):
        if labels[root] != -1:
            continue
        labels[root] = component
        queue[0] = root
        head, tail = 0, 1
        while head < tail:
            vertex = queue[head]
            head += 1
            for other in range(size):
                if labels[other] == -1 and adjacency[vertex, other] != 0:
                    labels[other] = component
                    queue[tail] = other
                    tail += 1
        component += 1
    return labels


@njit(nogil=True, cache=True)
def _hermitian_coordinates(labels):
    # Populations precede adjacent real/imaginary pairs in dressed index order.
    size = len(labels)
    dimension = size
    for i in range(size):
        for j in range(i + 1, size):
            if labels[i] == labels[j]:
                dimension += 2
    a = np.empty(dimension, dtype=np.int64)
    b = np.empty(dimension, dtype=np.int64)
    kind = np.empty(dimension, dtype=np.int64)
    for i in range(size):
        a[i], b[i], kind[i] = i, i, 0
    index = size
    for i in range(size):
        for j in range(i + 1, size):
            if labels[i] == labels[j]:
                a[index], b[index], kind[index] = i, j, 1
                a[index + 1], b[index + 1], kind[index + 1] = i, j, 2
                index += 2
    return a, b, kind


def rwa_topology(graph: PhaseGraph, work_plan: WorkPlan) -> RWATopology:
    """Resolve graph topology once, without admitting any row or building a bath.

    Graph pruning already fixes compatible Fourier edges. Callers still check
    pruning and the coordinate dimension against each row's own admission.
    """
    key = FactKey(("rwa-topology", graph.identity), RWATopology)

    def build() -> RWATopology:
        positive = np.abs(graph.positive).sum(axis=0)
        labels = _component_labels(positive + positive.T)
        a, b, kind = _hermitian_coordinates(labels)
        components = tuple(
            np.flatnonzero(labels == label) for label in sorted(set(labels))
        )
        return RWATopology(graph.identity, labels, components, a, b, kind)

    return work_plan.fact(key, build)


@dataclass(frozen=True)
class DriveBound:
    """Model/port scalars for shaped guards, independent of density and gain.

    strength is the maximum spectral norm of the two drive quadratures in GHz.
    rate is the larger of total outgoing rates and pure-dephasing rates, in /ns.
    """

    strength: float
    rate: float


def drive_bound(model: SecularModel, port: int, work_plan: WorkPlan) -> DriveBound:
    """Resolve shaped drive/rate bounds once per full model identity/level/port.

    port indexes model.drives. Use the original spectral-norm and rate-reduction
    formulas; only their immutable scalar results live until WorkPlan.close.
    """
    key = FactKey(("shaped-drive-bound", model.identity, model.level, port), DriveBound)

    def build() -> DriveBound:
        norms = quadrature_norms(model, port, work_plan)
        return DriveBound(
            max(norms.in_phase, norms.quadrature),
            float(
                max(
                    model.rates_per_ns.sum(axis=0).max(),
                    model.dephasing_per_ns.max(),
                )
            ),
        )

    return work_plan.fact(key, build)


def split_rate(
    model: SecularModel,
    port: int,
    frequency: float,
    gain: float,
    work_plan: WorkPlan,
) -> float:
    """Resolve the period split bound per model/level/port/signed carrier/gain."""
    key = FactKey(
        ("period-split-rate", model.identity, model.level, port, frequency, gain), float
    )

    def build() -> float:
        generator, decay = secular_operators(model)
        gamma = max(float(np.linalg.norm(generator, 2)), float(np.max(decay)))
        drive = model.drives[port]
        strength = (
            2
            * np.pi
            * abs(gain)
            * (
                np.linalg.norm(drive.in_phase_ghz, 2)
                + np.linalg.norm(drive.quadrature_ghz, 2)
            )
        )
        return float(gamma * (strength / frequency) ** 2)

    return work_plan.fact(key, build)


def packed_segments(
    table: SegmentTable, work_plan: WorkPlan, *, include_flux: bool = False
) -> Segments:
    """Pack read-only DOP samples once per table and flux-packing mode.

    Prefix representatives have their own table identity and row numbering.
    Solver history and density workspaces never enter these immutable facts.
    """
    key = FactKey(("dop853-segments", TableIdentity(table), include_flux), Segments)
    return work_plan.fact(key, lambda: pack_segments(table, include_flux=include_flux))


def fixed_flux(table: SegmentTable, row: int) -> bool:
    """Return whether this row's actual interpolated flux is constant on support.

    Check every row before using phase-covariant physics facts: equal model
    levels and playback keys do not imply equal continuous flux waveforms.
    Missing continuous samples retain quantized-model admission.
    """
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
