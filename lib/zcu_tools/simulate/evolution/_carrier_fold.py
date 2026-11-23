"""Fold proportional drive lines without rewriting the caller's tracks or models."""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from hashlib import sha256
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from .engine import EnvelopeSpec, TimelineBatch
from .secular import FluxLevel, ModelSource, SecularDrive, SecularModel


class ToneColumns(NamedTuple):
    """Mutable row/slot arrays owned by geometry construction."""

    port: NDArray[np.int64]
    frequency: NDArray[np.float64]
    amplitude: NDArray[np.float64]
    theta: NDArray[np.float64]
    envelope: NDArray[np.int64]
    offset: NDArray[np.float64]


@dataclass(frozen=True)
class CarrierGroup:
    """Repeated signed carrier in a row, with physical channels in input order.

    channels: original port names, allowing repeats for distinct drive slots.
    slots: indices on the effective tone axis; one slot if folded.
    folded_identity: stable drive recipe digest, empty if the slots are kept.
    The carrier itself is on the tone axis at any of these slots.
    """

    channels: tuple[str, ...]
    slots: tuple[int, ...]
    folded_identity: bytes = b""


class FoldedDrive(SecularDrive):
    """A model-local virtual port, not an independently playable physical port."""


type _DriveTerms = tuple[tuple[int, complex], ...]
type _DrivePort = tuple[int, bytes]


class FoldedDrives:
    """Intern virtual quadratures after the physical ports at each flux level.

    A single owner serves all geometry batches and their prefix subsets. Slot
    indices distinguish recipes inside this evolve. The digest uses port names
    and coefficients rather than a local index for keys that outlive the plan.
    """

    def __init__(self, source: ModelSource) -> None:
        self._source = source
        self._ports: dict[FluxLevel, dict[_DriveTerms, _DrivePort]] = {}
        self._ratios: dict[tuple[str, str], complex | None] = {}

    def ratio(self, first: EnvelopeSpec, other: EnvelopeSpec) -> complex | None:
        """Return the sample scale from the first shape to the other, if equal."""
        if first.identity == other.identity:
            return 1 + 0j
        key = (first.identity, other.identity)
        if key not in self._ratios:
            self._ratios[key] = _sample_ratio(first, other)
        return self._ratios[key]

    def port(self, level: FluxLevel, terms: _DriveTerms) -> _DrivePort:
        """Return the virtual model port and its stable, ordered recipe digest."""
        ports = self._ports.setdefault(level, {})
        if terms not in ports:
            model = self._source.model(level)
            recipe = tuple(
                (model.drives[port].name, scale.real.hex(), scale.imag.hex())
                for port, scale in terms
            )
            ports[terms] = (
                len(model.drives) + len(ports),
                sha256(repr(recipe).encode()).digest(),
            )
        return ports[terms]

    def source(self, batches: Sequence[TimelineBatch]) -> ModelSource:
        """Build an evolve-local source view, or return the original if unchanged."""
        physical = self._source
        if not self._ports:
            return physical
        try:
            expansion = physical.flux_expansion()
        except ValueError:
            expansion = None  # ModelSource reports an absent optional expansion.
        planned = ModelSource(
            physical.initial_density,
            self._build,
            expansion,
            slow_envelope_segments_per_period=physical.slow_envelope_segments_per_period,
            history=physical.history,
        )
        for batch in batches:
            for shifts in batch.segment_shifts:
                for shift in shifts:
                    planned.register_shift(shift)
        return planned

    def _build(self, level: FluxLevel) -> SecularModel:
        model = self._source.model(level)
        if level not in self._ports:
            return model
        drives = list(model.drives)
        names = {drive.name for drive in drives}
        for terms, (_, identity) in self._ports[level].items():
            name = "folded-carrier:" + identity.hex()
            while name in names:
                name += ":"
            names.add(name)
            drives.append(_combine_drive(model, name, terms))
        return replace(model, drives=tuple(drives))


def _combine_drive(model: SecularModel, name: str, terms: _DriveTerms) -> SecularDrive:
    x = np.zeros_like(model.eigenvectors)
    y = np.zeros_like(x)
    for port, scale in terms:
        drive = model.drives[port]
        x += scale.real * drive.in_phase_ghz + scale.imag * drive.quadrature_ghz
        y += -scale.imag * drive.in_phase_ghz + scale.real * drive.quadrature_ghz
    return FoldedDrive(name, x, y)


def repeated_carrier_rows(columns: ToneColumns) -> NDArray[np.int64]:
    """Return rows with at least two playing slots at exactly one signed carrier."""
    frequency, port = columns.frequency, columns.port
    repeated = np.zeros(len(port), dtype=bool)
    for slot in range(port.shape[1] - 1):
        repeated |= (port[:, slot] >= 0) & np.any(
            (port[:, slot + 1 :] >= 0)
            & (frequency[:, slot + 1 :] == frequency[:, slot, None]),
            axis=1,
        )
    return np.flatnonzero(repeated)


@dataclass(frozen=True)
class _Line:
    port: int
    channel: str
    start: float
    duration: float
    envelope: EnvelopeSpec
    scale: complex


def _line(batch: TimelineBatch, point: int, track_index: int, port: int) -> _Line:
    track = batch.tracks[track_index]
    return _Line(
        port,
        track.port,
        float(track.start_ns[point]),
        float(track.duration_ns[point]),
        track.envelopes[track.envelope_index[point]],
        complex(track.amplitude[point] * np.exp(1j * track.phase_rad[point])),
    )


def _sample_ratio(first: EnvelopeSpec, other: EnvelopeSpec) -> complex | None:
    """Admit only roundoff-scale differences on an identical sampled clock."""
    if (
        first.kind != "sampled"
        or other.kind != "sampled"
        or not np.array_equal(first.sample_ns, other.sample_ns)
    ):
        return None
    a, b = first.samples, other.samples
    # Use the largest component as the pivot. In particular, the pulse may
    # begin with zero padding, and comparing with an absolute tolerance would
    # silently discard real drive at those zeros.
    with np.errstate(over="ignore", invalid="ignore"):
        pivot = int(np.argmax(np.maximum(np.abs(a.real), np.abs(a.imag))))
        if a[pivot] == 0 or b[pivot] == 0:
            return None
        ratio = complex(b[pivot]) / complex(a[pivot])
        if np.isfinite(ratio) and np.allclose(
            a * ratio, b, rtol=8 * np.finfo(np.float64).eps, atol=0
        ):
            return ratio
    return None


def _terms(
    lines: tuple[_Line, ...], drives: FoldedDrives, start: float, *, flattened: bool
) -> _DriveTerms:
    """Return a fold recipe, or no terms when any line has a different shape."""
    first = lines[0]
    scale = 1 + 0j
    if flattened and first.envelope.kind == "sampled":
        # Use the same absolute clock as plateau event construction. Subtracting
        # the pulse start can round to just before the knot on that boundary.
        envelope = first.envelope
        index = np.searchsorted(first.start + envelope.sample_ns, start, side="right")
        scale = complex(envelope.samples[max(0, index - 1)])
    terms = []
    for line in lines:
        if (line.start, line.duration) != (first.start, first.duration):
            return ()
        ratio = drives.ratio(first.envelope, line.envelope)
        if ratio is None:
            return ()
        terms.append((line.port, scale * ratio * line.scale))
    return tuple(terms)


class CarrierRow(NamedTuple):
    """Track provenance and clocks for an unmodified geometry row."""

    point: int
    start: float
    level: FluxLevel
    tracks: NDArray[np.int64]


def fold_row(
    batch: TimelineBatch,
    columns: ToneColumns,
    row: int,
    original: CarrierRow,
    drives: FoldedDrives,
) -> tuple[int, tuple[CarrierGroup, ...]]:
    """Fold complete equal-shape groups in place; leave heterogeneous groups intact."""
    by_carrier: dict[float, list[int]] = {}
    for slot in range(len(original.tracks)):
        by_carrier.setdefault(float(columns.frequency[row, slot]), []).append(slot)
    removed: set[int] = set()
    groups = []
    for frequency, members in by_carrier.items():
        if len(members) < 2:
            continue
        lines = tuple(
            _line(
                batch,
                original.point,
                int(original.tracks[slot]),
                int(columns.port[row, slot]),
            )
            for slot in members
        )
        slot = members[0]
        terms = _terms(
            lines, drives, original.start, flattened=columns.envelope[row, slot] == 0
        )
        identity = b""
        if terms:
            port, identity = drives.port(original.level, terms)
            columns.port[row, slot] = port
            columns.amplitude[row, slot] = 1
            columns.theta[row, slot] = 2 * np.pi * frequency * original.start
            removed.update(members[1:])
            members = [slot]
        groups.append(
            CarrierGroup(
                tuple(line.channel for line in lines), tuple(members), identity
            )
        )
    if not removed:
        return len(original.tracks), tuple(groups)

    kept = [slot for slot in range(len(original.tracks)) if slot not in removed]
    for array in columns:
        array[row, : len(kept)] = array[row, kept]
        array[row, len(kept) :] = -1 if array.dtype.kind == "i" else 0
    slots = {old: new for new, old in enumerate(kept)}
    return len(kept), tuple(
        replace(group, slots=tuple(slots[slot] for slot in group.slots))
        for group in groups
    )
