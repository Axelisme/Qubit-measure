"""Vectorized event geometry for one acquire, without point-wise Python loops."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .engine import EnvelopeSpec, TimelineBatch
from .kernel import SegmentTable
from .secular import FluxWaveform, ModelSource


@dataclass(frozen=True)
class Geometry:
    table: SegmentTable
    points: NDArray[np.int64]
    flux_drive: NDArray[np.bool_]


def _events(timeline, time):
    result = np.empty(np.shape(time), dtype=[("timeline", "i8"), ("time", "f8")])
    result["timeline"], result["time"] = timeline, time
    return result


def _plateaus(envelope: EnvelopeSpec, duration: float):
    """Return maximal equal-sample runs, including final-sample hold."""
    times = np.r_[envelope.sample_ns, max(duration, envelope.sample_ns[-1])]
    same = np.r_[envelope.samples[1:] == envelope.samples[:-1], True]
    edges = np.diff(np.r_[False, same, False].astype(np.int64))
    first, last = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    valid = times[last] - times[first] >= 10
    return times[first[valid]], times[last[valid]], envelope.samples[first[valid]]


def _boundaries(batch: TimelineBatch):
    n = len(batch.points)
    point = np.arange(n)
    flux_point = np.repeat(point, np.diff(batch.flux.offsets))
    events = [
        _events(point, np.zeros(n)),
        _events(point, batch.end_ns),
        _events(flux_point, batch.flux.start_ns),
    ]
    for track in batch.tracks:
        active = track.active & (track.duration_ns > 0) & (track.amplitude != 0)
        events.extend(
            (
                _events(point[active], track.start_ns[active]),
                _events(point[active], (track.start_ns + track.duration_ns)[active]),
            )
        )
        for index, envelope in enumerate(track.envelopes):
            if envelope.kind != "sampled":
                continue
            selected = active & (track.envelope_index == index)
            if not selected.any():
                continue
            starts, ends, _ = _plateaus(
                envelope, float(track.duration_ns[selected].max())
            )
            clipped_ends = np.minimum(ends[None, :], track.duration_ns[:, None])
            mask = selected[:, None] & (clipped_ends - starts[None, :] >= 10)
            for boundary in (np.broadcast_to(starts, clipped_ends.shape), clipped_ends):
                times = track.start_ns[:, None] + boundary
                events.append(
                    _events(
                        np.broadcast_to(point[:, None], times.shape)[mask], times[mask]
                    )
                )
    merged = np.concatenate(events)
    merged = np.unique(
        merged[
            (merged["time"] >= 0) & (merged["time"] <= batch.end_ns[merged["timeline"]])
        ]
    )
    valid = (merged["timeline"][1:] == merged["timeline"][:-1]) & (
        np.diff(merged["time"]) > 0
    )
    return merged[:-1][valid], np.diff(merged["time"])[valid], flux_point


def _tone_columns(batch, table, source, flux_point):
    """Populate per-slot columns, fold plateau phase, and detect flux overlap."""
    point, start, duration, level = table
    width, rows = len(batch.tracks), len(point)
    playing = np.zeros((rows, width), dtype=bool)
    port = np.full((rows, width), -1, dtype=np.int64)
    frequency, amplitude, theta, offset = (np.zeros((rows, width)) for _ in range(4))
    env_index = np.full((rows, width), -1, dtype=np.int64)
    envelopes: list[EnvelopeSpec] = [EnvelopeSpec("constant")]
    flux_drive = np.zeros(rows, dtype=bool)
    # Lattice events also route varying drives when continuous samples are absent.
    change = np.ones(len(flux_point), dtype=bool)
    change[1:] = (flux_point[1:] == flux_point[:-1]) & np.any(
        batch.flux.level[1:] != batch.flux.level[:-1], axis=1
    )
    change[batch.flux.offsets[:-1]] = False
    for slot, track in enumerate(batch.tracks):
        local = start - track.start_ns[point]
        active = (
            track.active[point]
            & (local >= 0)
            & (local < track.duration_ns[point])
            & (track.amplitude[point] != 0)
        )
        playing[:, slot] = active
        frequency[:, slot] = track.frequency_ghz[point]
        amplitude[:, slot] = track.amplitude[point]
        theta[:, slot] = 2 * np.pi * frequency[:, slot] * start + track.phase_rad[point]
        offset[:, slot] = np.maximum(local, 0)
        base = len(envelopes)
        envelopes.extend(track.envelopes)
        env_index[active, slot] = base + track.envelope_index[point[active]]
        for index, envelope in enumerate(track.envelopes):
            selected = active & (track.envelope_index[point] == index)
            if envelope.kind == "sampled" and selected.any():
                starts, ends, values = _plateaus(
                    envelope, float(track.duration_ns[point[selected]].max())
                )
                # Plateau count is envelope structure, not sweep point count.
                for a, b, value in zip(starts, ends, values, strict=True):
                    clipped_end = np.minimum(b, track.duration_ns[point])
                    flat = (
                        selected
                        & (clipped_end - a >= 10)
                        & (local >= a)
                        & (local + duration <= clipped_end)
                    )
                    env_index[flat, slot] = 0
                    amplitude[flat, slot] *= abs(value)
                    theta[flat, slot] += np.angle(value)
                    offset[flat, slot] = 0
        for lev in np.unique(level[active], axis=0):
            mask = active & np.all(level == lev, axis=1)
            model = source.model(tuple(int(v) for v in lev))
            names = [drive.name for drive in model.drives]
            port[mask, slot] = names.index(track.port)
        changed_points = np.zeros(len(batch.points), dtype=bool)
        times = batch.flux.start_ns
        inside = (
            change
            & (times > track.start_ns[flux_point])
            & (times < (track.start_ns + track.duration_ns)[flux_point])
            & (times < batch.end_ns[flux_point])
        )
        np.logical_or.at(changed_points, flux_point, inside)
        flux_drive |= active & changed_points[point]
    # Pack active slots to the left; padding has finite ignored values.
    order = np.argsort(~playing, axis=1, kind="stable")
    packed = (
        np.take_along_axis(port, order, axis=1),
        np.take_along_axis(frequency, order, axis=1),
        np.take_along_axis(amplitude, order, axis=1),
        np.take_along_axis(theta, order, axis=1),
        np.take_along_axis(env_index, order, axis=1),
        np.take_along_axis(offset, order, axis=1),
    )
    return playing.sum(axis=1).astype(np.int64), packed, tuple(envelopes), flux_drive


def _waveform_varies(batch: TimelineBatch, point, start, duration):
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
    last = np.searchsorted(sample_events, _events(point, start + duration), side="left")
    last = np.maximum(last, first)
    count = np.r_[0, np.cumsum(varying)]
    return count[last] > count[first]


def _batch_geometry(batch: TimelineBatch, source: ModelSource) -> Geometry:
    events, duration, flux_point = _boundaries(batch)
    point, start = events["timeline"], events["time"]
    flux_events = _events(flux_point, batch.flux.start_ns)
    flux_row = np.searchsorted(flux_events, events, side="right") - 1
    level = batch.flux.level[flux_row]
    tones, columns, envelopes, flux_drive = _tone_columns(
        batch, (point, start, duration, level), source, flux_point
    )
    flux_drive |= (tones > 0) & _waveform_varies(batch, point, start, duration)
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
    )
    return Geometry(table, batch.points, flux_drive)


def plan(source: ModelSource, batches: Sequence[TimelineBatch]) -> Geometry:
    """Segment templates with array event sorting; assign global timeline indices."""
    geometries = [_batch_geometry(batch, source) for batch in batches]
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
    timeline_base = 0
    for geometry in geometries:
        table = geometry.table
        for name in columns:
            column = getattr(table, name)
            if name == "timeline":
                column = column + timeline_base
            if name.startswith("tone_"):
                pad = -1 if name in ("tone_port", "tone_envelope") else 0
                if name == "tone_envelope":
                    column = np.where(column >= 0, column + len(envelopes), -1)
                column = np.pad(
                    column, ((0, 0), (0, width - column.shape[1])), constant_values=pad
                )
            columns[name].append(column)
        timeline_base += len(geometry.points)
        envelopes.extend(table.envelopes)
    waveform = _combine_waveforms(batches)
    combined = SegmentTable(
        timeline=np.concatenate(columns["timeline"]),
        start_ns=np.concatenate(columns["start_ns"]),
        duration_ns=np.concatenate(columns["duration_ns"]),
        level=np.concatenate(columns["level"]),
        tones=np.concatenate(columns["tones"]),
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
