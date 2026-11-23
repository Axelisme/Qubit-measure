"""Certify identical histories and resolve their evolve-local known inputs."""

from collections.abc import Mapping
from dataclasses import replace

import numpy as np

from zcu_tools.simulate.boundary import CancellationQuery, Cancelled

from ._capture import CaptureRun
from ._dispatch import classify
from ._parallel import Executor
from ._planning import Geometry
from ._schedule import Schedule
from ._validation import owned
from .kernel import Kernel, KernelContext, PrefixOpenCLQuote, SegmentTable
from .secular import FluxWaveform


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
            table.shift(row),
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
        signatures.append(b"".join((*key, table.folded_channel_key(row))))
    matches: dict[tuple[bytes, ...], list[int]] = {}
    starts = np.searchsorted(table.timeline, table.timeline[candidates], side="left")
    for row, first in zip(candidates, starts, strict=True):
        key = (*signatures[first:row], table.level[row].tobytes(), table.shift(row))
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
            "shift_content",
            "complete_pulses",
            "has_collapse_operators",
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
        carrier_groups={
            row - first: table.carrier_groups[row]
            for row in range(first, stop)
            if row in table.carrier_groups
        }
        if table.carrier_groups
        else {},
        prefix_group=table.prefix_group[first:stop]
        if table.prefix_group is not None
        else None,
        flux_waveform=waveform,
        flux_waveform_available=table.flux_waveform_available[point : point + 1]
        if table.flux_waveform_available is not None
        else None,
    )
    return Geometry(
        subset,
        geometry.points[point : point + 1],
        geometry.flux_drive[first:stop],
        geometry.reject_jump_flux[first:stop],
        geometry.prefix_paths,
    )


def _cpu_context(geometry: Geometry, context: KernelContext) -> KernelContext:
    """Keep prefix execution CPU-only while quoting possible suffix endpoints."""
    if context.opencl is None:
        return context
    table = geometry.table
    certified = np.array(
        [i for i, group in enumerate(table.prefix_group or ()) if group is not None],
        dtype=np.int64,
    )
    latest = np.full(len(geometry.points), -1, dtype=np.int64)
    np.maximum.at(latest, table.timeline[certified], certified)
    suffix = np.arange(len(table.timeline)) >= latest[table.timeline]
    possible_direct = np.flatnonzero(
        suffix
        & (table.tones > 0)
        & ~geometry.flux_drive
        & ~table.has_collapse_operators
    )
    possible_direct.setflags(write=False)
    return replace(
        context,
        opencl=None,
        prefix_opencl=PrefixOpenCLQuote(context.opencl, possible_direct),
    )


class PrefixResolver:
    """One evolve's chronological prefix evaluation with ordinary scheduling."""

    def __init__(
        self,
        geometry: Geometry,
        kernels: Mapping[str, Kernel],
        context: KernelContext,
        cancel: CancellationQuery,
        capture: CaptureRun | None = None,
    ):
        self.geometry = geometry
        self.capture = capture
        self.kernels = kernels
        self.context = context
        self.cancel = cancel

    def resolve(self, pool: Executor, chunks: int) -> Geometry:
        """Resolve nested groups in time order with evolve-local input ownership."""
        geometry, kernels, cancel = self.geometry, self.kernels, self.cancel
        table = geometry.table
        representatives: dict[int, int] = {}
        for row, group in enumerate(table.prefix_group or ()):
            if group is not None:
                representatives.setdefault(group, row)
        if not representatives:
            return geometry
        context = _cpu_context(geometry, self.context)
        source = context.source
        inputs = {}
        paths = {}
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
            geometry = replace(geometry, table=table, prefix_paths=paths)
            prefix = _representative(geometry, first, row)
            capture = (
                None
                if self.capture is None
                else self.capture.prefix(geometry, first, row, group)
            )
            prefix_context = replace(context, complete_row_offset=first)
            dispatch = classify(
                prefix,
                kernels,
                prefix_context,
                trace=None if capture is None else capture.round(prefix),
            )
            schedule = Schedule(
                prefix,
                dispatch,
                prefix_context,
                cancel,
                initial,
                capture=capture,
            )
            output, diagnostics = schedule.run(kernels, pool, chunks)
            inputs[group] = owned(output[0], np.complex128, "prefix density")
            paths[group] = diagnostics
        return replace(
            geometry, table=replace(table, prefix_inputs=inputs), prefix_paths=paths
        )
