"""Select point-aligned fields for the TimelineBatch public operation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .secular import FluxSchedule, FluxWaveform

if TYPE_CHECKING:
    from .engine import DriveTrack, TimelineBatch


def select_points(batch: TimelineBatch, positions: Sequence[int]) -> TimelineBatch:
    rows = tuple(positions)
    if any(not _valid_point_position(row, len(batch.points)) for row in rows) or len(
        set(rows)
    ) != len(rows):
        raise ValueError("point positions must be unique integer rows in range")
    index = np.array(rows, dtype=np.int64)
    offsets, samples = _ragged_selection(batch.flux.offsets, index)
    waveform = batch.flux_waveform
    if waveform is not None:
        wave_offsets, wave_samples = _ragged_selection(waveform.offsets, index)
        waveform = FluxWaveform(
            wave_offsets,
            waveform.time_ns[wave_samples],
            waveform.flux[wave_samples],
        )
    return replace(
        batch,
        points=batch.points[index],
        tracks=tuple(_select_drive_rows(track, index) for track in batch.tracks),
        flux=FluxSchedule(
            offsets, batch.flux.start_ns[samples], batch.flux.level[samples]
        ),
        end_ns=batch.end_ns[index],
        flux_waveform=waveform,
        segment_boundaries=tuple(batch.segment_boundaries[i] for i in rows)
        if batch.segment_boundaries
        else (),
        segment_shifts=tuple(batch.segment_shifts[i] for i in rows)
        if batch.segment_shifts
        else (),
        start_ns=None if batch.start_ns is None else batch.start_ns[index],
        initial_densities=None
        if batch.initial_densities is None
        else batch.initial_densities[index],
    )


def _valid_point_position(position: object, count: int) -> bool:
    return (
        isinstance(position, (int, np.integer))
        and not isinstance(position, bool)
        and 0 <= int(position) < count
    )


def _select_drive_rows(track: DriveTrack, index: NDArray[np.int64]) -> DriveTrack:
    return replace(
        track,
        start_ns=track.start_ns[index],
        duration_ns=track.duration_ns[index],
        frequency_ghz=track.frequency_ghz[index],
        phase_rad=track.phase_rad[index],
        amplitude=track.amplitude[index],
        envelope_index=track.envelope_index[index],
        active=track.active[index],
    )


def _ragged_selection(
    offsets: NDArray[np.int64], index: NDArray[np.int64]
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    starts, stops = offsets[index], offsets[index + 1]
    selected_offsets = np.empty(len(index) + 1, dtype=np.int64)
    selected_offsets[0] = 0
    np.cumsum(stops - starts, out=selected_offsets[1:])
    spans = [
        np.arange(start, stop, dtype=np.int64)
        for start, stop in zip(starts, stops, strict=True)
    ]
    samples = np.concatenate(spans) if spans else np.empty(0, dtype=np.int64)
    return selected_offsets, samples
