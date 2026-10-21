"""Vectorized event geometry for one acquire, without point-wise Python loops."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from numba import njit
from numpy.typing import NDArray

from .engine import DriveTrack, EnvelopeSpec, PathDiagnostics, TimelineBatch
from .kernel import SegmentTable
from .secular import FluxWaveform, ModelSource, ShiftOrderingError


@dataclass(frozen=True)
class Geometry:
    """Segmented acquire supplied to dispatch and the engine scheduler.

    table contains global rows and optional engine-computed prefix certificates.
    points maps timeline indices to unique caller point IDs.
    flux_drive marks driven rows with continuous motion or quantized flux changes.
    prefix_paths maps computed prefix IDs to diagnostics adopted by timelines.
    plan(source, batches) constructs geometry without evolving a density.
    """

    table: SegmentTable
    points: NDArray[np.int64]
    flux_drive: NDArray[np.bool_]
    prefix_paths: Mapping[int, PathDiagnostics] = field(default_factory=dict)

    @property
    def resume_rows(self) -> NDArray[np.int64]:
        """Latest resolved prefix row per timeline, or -1 for source input."""
        resume = np.full(len(self.points), -1, dtype=np.int64)
        if self.table.prefix_group is not None:
            rows = np.array(
                [
                    i
                    for i, group in enumerate(self.table.prefix_group)
                    if group is not None and group in self.table.prefix_inputs
                ],
                dtype=np.int64,
            )
            np.maximum.at(resume, self.table.timeline[rows], rows)
        return resume

    @property
    def pending(self) -> NDArray[np.bool_]:
        """Rows after the adopted input, excluding already computed prefixes."""
        return (
            np.arange(len(self.table.timeline)) >= self.resume_rows[self.table.timeline]
        )


def _events(timeline, time):
    result = np.empty(np.shape(time), dtype=[("timeline", "i8"), ("time", "f8")])
    result["timeline"], result["time"] = timeline, time
    return result


type _Plateaus = tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128]]
type _PlateauCache = dict[tuple[str, float], _Plateaus]


def _plateaus(envelope: EnvelopeSpec, duration: float) -> _Plateaus:
    """Return maximal equal-sample runs, including final-sample hold."""
    times = np.r_[envelope.sample_ns, max(duration, envelope.sample_ns[-1])]
    same = np.r_[envelope.samples[1:] == envelope.samples[:-1], True]
    edges = np.diff(np.r_[False, same, False].astype(np.int64))
    first, last = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    valid = times[last] - times[first] >= 10
    return times[first[valid]], times[last[valid]], envelope.samples[first[valid]]


@dataclass(frozen=True)
class _Support:
    """Absolute half-open support shared by event emission and membership."""

    active: NDArray[np.bool_]
    start: NDArray[np.float64]
    end: NDArray[np.float64]

    def contains(self, point, start, end=None):
        """Select points inside support, optionally requiring a whole interval."""
        inside = (
            self.active[point]
            & (start >= self.start[point])
            & (start < self.end[point])
        )
        if end is not None:
            inside &= end <= self.end[point]
        return inside


@dataclass(frozen=True)
class _TrackGeometry:
    """Pulse and plateau endpoints, formed once in absolute lattice time."""

    pulse: _Support
    plateaus: tuple[tuple[_Support, complex], ...]


def _track_geometry(track: DriveTrack, plateau_cache: _PlateauCache) -> _TrackGeometry:
    active = track.active & (track.duration_ns > 0) & (track.amplitude != 0)
    pulse = _Support(active, track.start_ns, track.start_ns + track.duration_ns)
    plateaus = []
    for index, envelope in enumerate(track.envelopes):
        selected = active & (track.envelope_index == index)
        if envelope.kind != "sampled" or not selected.any():
            continue
        maximum = float(track.duration_ns[selected].max())
        # Final-sample hold makes run lengths depend on playback duration too.
        key = (envelope.identity, maximum)
        if key not in plateau_cache:
            plateau_cache[key] = _plateaus(envelope, maximum)
        starts, ends, values = plateau_cache[key]
        for a, b, value in zip(starts, ends, values, strict=True):
            clipped_end = np.minimum(b, track.duration_ns)
            support = _Support(
                selected & (clipped_end - a >= 10),
                track.start_ns + a,
                track.start_ns + clipped_end,
            )
            plateaus.append((support, complex(value)))
    return _TrackGeometry(pulse, tuple(plateaus))


def _boundaries(batch: TimelineBatch, tracks: Sequence[_TrackGeometry]):
    n = len(batch.points)
    point = np.arange(n)
    flux_point = np.repeat(point, np.diff(batch.flux.offsets))
    events = [
        _events(point, np.zeros(n)),
        _events(point, batch.end_ns),
        _events(flux_point, batch.flux.start_ns),
    ]
    for point_index, times in enumerate(batch.segment_boundaries):
        events.append(_events(np.full(len(times), point_index), np.asarray(times)))
    for track in tracks:
        for support in (track.pulse, *(p for p, _ in track.plateaus)):
            for boundary in (support.start, support.end):
                events.append(_events(point[support.active], boundary[support.active]))
    merged = np.concatenate(events)
    merged = np.unique(
        merged[
            (merged["time"] >= 0) & (merged["time"] <= batch.end_ns[merged["timeline"]])
        ]
    )
    valid = (merged["timeline"][1:] == merged["timeline"][:-1]) & (
        np.diff(merged["time"]) > 0
    )
    return merged[:-1][valid], merged[1:]["time"][valid], flux_point


def _active_level_ports(
    source: ModelSource, level: NDArray[np.int64], active: NDArray[np.bool_]
) -> tuple[NDArray[np.int64], list[dict[str, int]]]:
    """Index each driven row's level and its model-local port names once."""
    level_index = np.full(len(level), -1, dtype=np.int64)
    ports = []
    if active.any():
        levels, level_index[active] = np.unique(
            level[active], axis=0, return_inverse=True
        )
        for lev in levels:
            model = source.model(tuple(int(v) for v in lev))
            ports.append(
                {drive.name: index for index, drive in enumerate(model.drives)}
            )
    return level_index, ports


@njit(cache=True)
def _support_rows(active, support_start, support_end, row_offsets, start, end):
    """Index half-open support in sorted per-point rows, retaining whole rows.

    Pulse/plateau endpoints already split the table. Binary searches avoid a
    full row-array scan for every historical pulse. Empty point spans and
    support beyond readout contribute no rows.
    """
    first = np.zeros(len(active), dtype=np.int64)
    last = np.zeros(len(active), dtype=np.int64)
    count = 0
    for point in range(len(active)):
        if not active[point]:
            continue
        base, stop = row_offsets[point], row_offsets[point + 1]
        times = start[base:stop]
        a = base + np.searchsorted(times, support_start[point])
        b = base + np.searchsorted(times, support_end[point])
        if b > a and end[b - 1] > support_end[point]:
            b -= 1
        first[point], last[point] = a, b
        count += b - a
    chosen = np.empty(count, dtype=np.int64)
    offset = 0
    for point in range(len(active)):
        for row in range(first[point], last[point]):
            chosen[offset] = row
            offset += 1
    return chosen


def _tone_columns(batch, tracks, table, source, flux_point):
    """Write only playing tones, in track order, folding phase and flux flags."""
    point, start, end, level = table
    rows = len(point)
    row_offsets = np.r_[0, np.cumsum(np.bincount(point, minlength=len(batch.points)))]
    track_rows = tuple(
        _support_rows(
            geometry.pulse.active,
            geometry.pulse.start,
            geometry.pulse.end,
            row_offsets,
            start,
            end,
        )
        for geometry in tracks
    )
    counts = np.zeros(rows, dtype=np.int64)
    for chosen in track_rows:
        counts[chosen] += 1
    width = int(counts.max(initial=0))
    port = np.full((rows, width), -1, dtype=np.int64)
    frequency, amplitude, theta, offset = (np.zeros((rows, width)) for _ in range(4))
    env_index = np.full((rows, width), -1, dtype=np.int64)
    used = np.zeros(rows, dtype=np.int64)
    envelopes = [EnvelopeSpec("constant")]
    flux_drive = np.zeros(rows, dtype=bool)
    change = np.ones(len(flux_point), dtype=bool)
    change[1:] = (flux_point[1:] == flux_point[:-1]) & np.any(
        batch.flux.level[1:] != batch.flux.level[:-1], axis=1
    )
    change[batch.flux.offsets[:-1]] = False
    level_index, ports = _active_level_ports(source, level, counts > 0)
    for track, geometry, chosen in zip(batch.tracks, tracks, track_rows, strict=True):
        if not len(chosen):
            continue
        slot = used[chosen]
        used[chosen] += 1
        frequency[chosen, slot] = track.frequency_ghz[point[chosen]]
        amplitude[chosen, slot] = track.amplitude[point[chosen]]
        theta[chosen, slot] = (
            2 * np.pi * frequency[chosen, slot] * start[chosen]
            + track.phase_rad[point[chosen]]
        )
        offset[chosen, slot] = np.maximum(
            start[chosen] - track.start_ns[point[chosen]], 0
        )
        base = len(envelopes)
        envelopes.extend(track.envelopes)
        env_index[chosen, slot] = base + track.envelope_index[point[chosen]]
        for support, value in geometry.plateaus:
            flat = support.contains(point[chosen], start[chosen], end[chosen])
            rr, cc = chosen[flat], slot[flat]
            env_index[rr, cc] = 0
            amplitude[rr, cc] *= abs(value)
            theta[rr, cc] += np.angle(value)
            offset[rr, cc] = 0
        for index, lookup in enumerate(ports):
            same = level_index[chosen] == index
            if same.any():
                if track.port not in lookup:
                    raise ValueError(f"{track.port!r} is not in list")
                port[chosen[same], slot[same]] = lookup[track.port]
        changed_points = np.zeros(len(batch.points), dtype=bool)
        times = batch.flux.start_ns
        inside = (
            change
            & geometry.pulse.contains(flux_point, times)
            & (times > geometry.pulse.start[flux_point])
            & (times < batch.end_ns[flux_point])
        )
        np.logical_or.at(changed_points, flux_point, inside)
        flux_drive[chosen] |= changed_points[point[chosen]]
    return (
        counts,
        (port, frequency, amplitude, theta, env_index, offset),
        tuple(envelopes),
        flux_drive,
    )


def _waveform_varies(batch: TimelineBatch, point, start, end):
    """Detect nonconstant interpolated sample intervals overlapping each row.

    Terminal samples hold DC after the model owner's tail truncation. Looking
    at sample slopes, not lattice switches, retains tails inside one level.
    """
    waveform = batch.flux_waveform
    if waveform is None:
        return np.zeros(len(point), dtype=bool)
    sample_point = np.repeat(np.arange(len(batch.points)), np.diff(waveform.offsets))
    sample_events = _events(sample_point, waveform.time_ns)
    varying = np.r_[
        (sample_point[1:] == sample_point[:-1])
        & np.any(waveform.flux[1:] != waveform.flux[:-1], axis=1),
        False,
    ]
    # Count varying intervals touching the open row support. Clamp at ragged
    # point edges so samples outside their span use the held endpoint value.
    first = np.maximum(
        np.searchsorted(sample_events, _events(point, start), side="right") - 1,
        waveform.offsets[point],
    )
    last = np.searchsorted(sample_events, _events(point, end), side="left")
    last = np.maximum(last, first)
    count = np.r_[0, np.cumsum(varying)]
    return count[last] > count[first]


def batch_boundaries(batch: TimelineBatch, plateau_cache: _PlateauCache):
    """Single authority for public queries and evolve event geometry."""
    tracks = tuple(_track_geometry(track, plateau_cache) for track in batch.tracks)
    return _boundaries(batch, tracks)


def _batch_geometry(
    batch: TimelineBatch, source: ModelSource, plateau_cache: _PlateauCache
) -> Geometry:
    tracks = tuple(_track_geometry(track, plateau_cache) for track in batch.tracks)
    events, end, flux_point = batch_boundaries(batch, plateau_cache)
    point, start = events["timeline"], events["time"]
    duration = end - start
    flux_events = _events(flux_point, batch.flux.start_ns)
    flux_row = np.searchsorted(flux_events, events, side="right") - 1
    level = batch.flux.level[flux_row]
    tones, columns, envelopes, flux_drive = _tone_columns(
        batch, tracks, (point, start, end, level), source, flux_point
    )
    flux_drive |= (tones > 0) & _waveform_varies(batch, point, start, end)
    port, frequency, amplitude, theta, envelope_index, offset = columns
    table = SegmentTable(
        point,
        start,
        duration,
        level,
        tones,
        port,
        frequency,
        amplitude,
        theta,
        envelope_index,
        offset,
        envelopes,
        batch.flux_waveform,
        shift_content=_shift_contents(batch, point, start, level, source),
    )
    return Geometry(table, batch.points, flux_drive)


def _shift_contents(batch, point, start, levels, source):
    contents = np.zeros(len(point), dtype="S64")
    for index, shifts in enumerate(batch.segment_shifts):
        for shift in shifts:
            row = np.flatnonzero((point == index) & (start == shift.start_ns))
            contents[row] = source.register_shift(shift)
            for selected in row:
                # Shape/order validation precedes any kernel preparation.
                try:
                    source.shifted_model(
                        tuple(int(v) for v in levels[selected]),
                        bytes(contents[selected]),
                    )
                except ShiftOrderingError as error:
                    raise ValueError(
                        f"point {batch.points[index]} segment "
                        f"[{shift.start_ns}, {shift.end_ns}] ns: {error}"
                    ) from error
    return contents


def plan(source: ModelSource, batches: Sequence[TimelineBatch]) -> Geometry:
    """Segment templates with array event sorting; assign global timeline indices."""
    plateau_cache: _PlateauCache = {}
    geometries = [_batch_geometry(batch, source, plateau_cache) for batch in batches]
    points = np.concatenate([g.points for g in geometries])
    if len(np.unique(points)) != len(points):
        raise ValueError("point indices must be unique across batches")
    tables = [g.table for g in geometries]
    width = max(table.tone_port.shape[1] for table in tables)
    columns = {
        name: []
        for name in SegmentTable.__dataclass_fields__
        if name
        not in (
            "envelopes",
            "flux_waveform",
            "flux_waveform_available",
            "prefix_group",
            "prefix_no_drive",
            "prefix_inputs",
        )
    }
    envelopes: list[EnvelopeSpec] = []
    envelope_ids: dict[str, int] = {}
    timeline_base = 0
    for geometry in geometries:
        table = geometry.table
        remap = np.empty(len(table.envelopes), dtype=np.int64)
        for index, envelope in enumerate(table.envelopes):
            if envelope.identity not in envelope_ids:
                envelope_ids[envelope.identity] = len(envelopes)
                envelopes.append(envelope)
            remap[index] = envelope_ids[envelope.identity]
        for name in columns:
            column = getattr(table, name)
            if name == "timeline":
                column = column + timeline_base
            if name.startswith("tone_"):
                pad = -1 if name in ("tone_port", "tone_envelope") else 0
                if name == "tone_envelope":
                    column = np.where(column >= 0, remap[np.maximum(column, 0)], -1)
                column = np.pad(
                    column, ((0, 0), (0, width - column.shape[1])), constant_values=pad
                )
            columns[name].append(column)
        timeline_base += len(geometry.points)
    waveform = _combine_waveforms(batches)
    combined = SegmentTable(
        timeline=np.concatenate(columns["timeline"]),
        start_ns=np.concatenate(columns["start_ns"]),
        duration_ns=np.concatenate(columns["duration_ns"]),
        level=np.concatenate(columns["level"]),
        tones=np.concatenate(columns["tones"]),
        shift_content=np.concatenate(columns["shift_content"]),
        tone_port=np.concatenate(columns["tone_port"]),
        tone_frequency_ghz=np.concatenate(columns["tone_frequency_ghz"]),
        tone_amplitude=np.concatenate(columns["tone_amplitude"]),
        tone_theta_rad=np.concatenate(columns["tone_theta_rad"]),
        tone_envelope=np.concatenate(columns["tone_envelope"]),
        tone_envelope_offset_ns=np.concatenate(columns["tone_envelope_offset_ns"]),
        envelopes=tuple(envelopes),
        flux_waveform=waveform,
        flux_waveform_available=(
            np.concatenate(
                [
                    np.full(
                        len(batch.points), batch.flux_waveform is not None, dtype=bool
                    )
                    for batch in batches
                ]
            )
            if waveform is not None
            else None
        ),
    )
    from ._prefix import certify

    combined = certify(combined)
    return Geometry(
        combined,
        points,
        np.concatenate([g.flux_drive for g in geometries]),
    )


def _combine_waveforms(batches: Sequence[TimelineBatch]) -> FluxWaveform | None:
    """Concatenate ragged samples; absent batches get inert per-point samples.

    The availability mask keeps absent timelines distinguishable from physical
    zero flux so a continuous kernel can reject them explicitly.
    """
    if all(batch.flux_waveform is None for batch in batches):
        return None
    counts, times, values = [], [], []
    for batch in batches:
        waveform = batch.flux_waveform
        if waveform is None:
            counts.append(np.ones(len(batch.points), dtype=np.int64))
            times.append(np.zeros(len(batch.points)))
            values.append(np.zeros((len(batch.points), batch.flux.level.shape[1])))
        else:
            counts.append(np.diff(waveform.offsets))
            times.append(waveform.time_ns)
            values.append(waveform.flux)
    return FluxWaveform(
        np.r_[0, np.cumsum(np.concatenate(counts))],
        np.concatenate(times),
        np.concatenate(values),
    )
