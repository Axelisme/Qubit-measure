"""Certify identical histories and resolve their evolve-local known inputs."""

from collections.abc import Mapping
from dataclasses import replace
from threading import Event

import numpy as np

from ._dispatch import classify
from ._parallel import Executor
from ._planning import Geometry
from ._schedule import PrepareCache, Schedule
from .engine import Cancelled
from .kernel import Kernel, SegmentTable
from .secular import FluxWaveform, ModelSource


def certify(table: SegmentTable) -> SegmentTable:
    """Group repeated nonempty prefixes of single constant-tone candidates.

    Compare every active tone's physical envelope identity, not its table index.
    Keep the first waveform sample at/after the probe for interpolation closure.
    Source and initial density are common to this whole plan, never persisted.
    """
    groups: list[int | None] = [None] * len(table.timeline)
    no_drive: list[bool] = []
    if not len(groups) or not table.tone_port.shape[1]:
        return replace(table, prefix_group=tuple(groups))
    constant = np.array([e.kind == "constant" for e in table.envelopes])
    candidates = np.flatnonzero(
        (table.start_ns > 0)
        & (table.tones == 1)
        & constant[np.maximum(table.tone_envelope[:, 0], 0)]
    )
    if len(candidates):
        # Identical complete prefixes must end at the same probe start time.
        _, inverse, counts = np.unique(
            table.start_ns[candidates], return_inverse=True, return_counts=True
        )
        candidates = candidates[counts[inverse] >= 2]
    if not len(candidates):
        return replace(table, prefix_group=tuple(groups))
    # Build each segment signature once. Padding does not describe physics.
    signatures = []
    for row, count in enumerate(table.tones):
        key = (
            table.start_ns[row].tobytes(),
            table.duration_ns[row].tobytes(),
            table.level[row].tobytes(),
            count.tobytes(),
        )
        if count:
            key += tuple(
                column[row, :count].tobytes()
                for column in (
                    table.tone_port,
                    table.tone_frequency_ghz,
                    table.tone_amplitude,
                    table.tone_theta_rad,
                    table.tone_envelope_offset_ns,
                )
            )
            key += tuple(
                table.envelopes[i].identity.encode()
                for i in table.tone_envelope[row, :count]
            )
        signatures.append(b"".join(key))
    matches: dict[tuple[bytes, ...], list[int]] = {}
    starts = np.searchsorted(table.timeline, table.timeline[candidates], side="left")
    for row, first in zip(candidates, starts, strict=True):
        key = (*signatures[first:row], table.level[row].tobytes())
        waveform, timeline = table.flux_waveform, table.timeline[row]
        available = table.flux_waveform_available
        if waveform is not None and (available is None or available[timeline]):
            a, b = waveform.offsets[timeline : timeline + 2]
            endpoint = np.searchsorted(
                waveform.time_ns[a:b], table.start_ns[row], side="left"
            )
            b = min(b, a + endpoint + 1)
            key += (waveform.time_ns[a:b].tobytes(), waveform.flux[a:b].tobytes())
        matches.setdefault(key, []).append(int(row))
    for rows in matches.values():
        if len(rows) < 2:
            continue
        first = np.searchsorted(table.timeline, table.timeline[rows[0]], side="left")
        group = len(no_drive)
        no_drive.append(not np.any(table.tones[first : rows[0]]))
        for row in rows:
            groups[row] = group
    return replace(table, prefix_group=tuple(groups), prefix_no_drive=tuple(no_drive))


def _representative(geometry: Geometry, first: int, stop: int) -> Geometry:
    """Slice one physical timeline prefix and remap its timeline to zero."""
    table = geometry.table
    point = table.timeline[stop]
    columns = {
        name: getattr(table, name)[first:stop]
        for name in (
            "start_ns",
            "duration_ns",
            "level",
            "tones",
            "tone_port",
            "tone_frequency_ghz",
            "tone_amplitude",
            "tone_theta_rad",
            "tone_envelope",
            "tone_envelope_offset_ns",
        )
    }
    waveform = table.flux_waveform
    if waveform is not None:
        a, b = waveform.offsets[point : point + 2]
        waveform = FluxWaveform(
            np.array([0, b - a]), waveform.time_ns[a:b], waveform.flux[a:b]
        )
    subset = replace(
        table,
        **columns,
        timeline=np.zeros(stop - first, dtype=np.int64),
        prefix_group=table.prefix_group[first:stop]
        if table.prefix_group is not None
        else None,
        flux_waveform=waveform,
        flux_waveform_available=table.flux_waveform_available[point : point + 1]
        if table.flux_waveform_available is not None
        else None,
    )
    return Geometry(
        subset, geometry.points[point : point + 1], geometry.flux_drive[first:stop]
    )


class PrefixResolver:
    """One evolve's chronological prefix evaluation with ordinary scheduling."""

    def __init__(
        self,
        geometry: Geometry,
        kernels: Mapping[str, Kernel],
        source: ModelSource,
        budget: float,
        cancel: Event,
    ):
        self.geometry = geometry
        self.kernels = kernels
        self.source = source
        self.budget = budget
        self.cancel = cancel

    def resolve(self, cache: PrepareCache, pool: Executor, chunks: int) -> Geometry:
        """Resolve nested groups in time order, never storing inputs in the LRU."""
        geometry, kernels, source, budget, cancel = (
            self.geometry,
            self.kernels,
            self.source,
            self.budget,
            self.cancel,
        )
        table = geometry.table
        representatives: dict[int, int] = {}
        for row, group in enumerate(table.prefix_group or ()):
            if group is not None:
                representatives.setdefault(group, row)
        if not representatives:
            return geometry
        inputs = {}
        for group, row in sorted(
            representatives.items(), key=lambda item: table.start_ns[item[1]]
        ):
            if cancel.is_set():
                raise Cancelled()
            first = int(
                np.searchsorted(table.timeline, table.timeline[row], side="left")
            )
            initial = source.initial_density
            for earlier in range(first, row):
                previous = (
                    table.prefix_group[earlier]
                    if table.prefix_group is not None
                    else None
                )
                if previous in inputs:
                    first, initial = earlier, inputs[previous]
            table = replace(table, prefix_inputs=inputs)
            geometry = replace(geometry, table=table)
            prefix = _representative(geometry, first, row)
            dispatch = classify(prefix, kernels, source, budget)
            schedule = Schedule(prefix, dispatch, source, budget, cancel, initial)
            result = schedule.run(kernels, cache, pool, chunks)
            inputs[group] = result.densities[0]
        return replace(geometry, table=replace(table, prefix_inputs=inputs))
