"""Immutable numerical facts shared across one evolve's kernel calls."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..kernel import SegmentTable
from ..secular import SecularModel
from ..workplan import FactKey, WorkPlan
from ._dop853_data import Segments, pack_segments
from ._secular import secular_operators


@dataclass(frozen=True, eq=False)
class _TableIdentity:
    # Keep the table alive so a later representative cannot reuse its object id.
    table: SegmentTable

    def __hash__(self) -> int:
        return id(self.table)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _TableIdentity) and self.table is other.table


@dataclass(frozen=True)
class EnvelopeBounds:
    """Peak magnitude, maximum linear slope per ns and real-sample flag."""

    peaks: NDArray[np.float64]
    slopes: NDArray[np.float64]
    real: NDArray[np.bool_]


def envelope_bounds(table: SegmentTable, work_plan: WorkPlan) -> EnvelopeBounds:
    """Resolve envelope bounds once per table, including prefix representatives."""
    key = FactKey(("envelope-bounds", _TableIdentity(table)), EnvelopeBounds)

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
        drive = model.drives[port]
        return DriveBound(
            float(
                max(
                    np.linalg.norm(drive.in_phase_ghz, 2),
                    np.linalg.norm(drive.quadrature_ghz, 2),
                )
            ),
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
    key = FactKey(("dop853-segments", _TableIdentity(table), include_flux), Segments)
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
