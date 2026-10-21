"""Sequential FTF density and dominant-branch cavity memory composition."""

from dataclasses import dataclass, replace
from hashlib import sha256
from threading import Event

import numpy as np
from numpy.typing import NDArray

from zcu_tools.resources.ftf import FTFSettings
from zcu_tools.simulate.adc import CavityAverageCache
from zcu_tools.simulate.evolution import (
    Engine,
    EvolutionResult,
    SegmentShift,
    TimelineBatch,
    merge_path_diagnostics,
    segment_spans,
)
from zcu_tools.simulate.ftf import CavityDrive

from .numeric import resonators
from .template import FTFModelFacts, FTFTemplatePlan


@dataclass
class _CavityTimeline:
    """One template's ragged span cursor, DC density and shared cavity memory."""

    batch: TimelineBatch
    spans: tuple[tuple[tuple[float, float], ...], ...]
    cavities: tuple[CavityDrive, ...]
    density: NDArray[np.complex128]
    fields: NDArray[np.complex128]

    def advance(self, step: int) -> TimelineBatch:
        starts = self.batch.end_ns.copy()
        ends = self.batch.end_ns.copy()
        shifts = []
        for point, (spans, cavity) in enumerate(
            zip(self.spans, self.cavities, strict=True)
        ):
            if step >= len(spans):
                shifts.append(())
                continue
            start, end = spans[step]
            starts[point], ends[point] = start, end
            populations = self.density[point].diagonal().real
            average, self.fields[point] = cavity.advance(
                start, end, populations, self.fields[point]
            )
            shifts.append(
                (
                    SegmentShift(
                        start, end, average.stark_shift_ghz, average.dephasing_per_ns
                    ),
                )
            )
        boundaries = tuple(
            tuple(t for t in times if t <= end)
            for times, end in zip(
                self.batch.segment_boundaries or ((),) * len(ends), ends, strict=True
            )
        )
        return replace(
            self.batch,
            start_ns=starts,
            end_ns=ends,
            initial_densities=self.density,
            segment_boundaries=boundaries,
            segment_shifts=tuple(shifts),
        )


def _span_key(key: bytes, step: int, batches: tuple[TimelineBatch, ...]) -> bytes:
    digest = sha256(b"ftf-cavity-span\0" + key + step.to_bytes(8, "little"))
    for batch in batches:
        # The caller's reuse identity also includes these continuation inputs.
        if batch.start_ns is not None and batch.initial_densities is not None:
            digest.update(batch.start_ns.tobytes())
            digest.update(batch.initial_densities.tobytes())
    return digest.digest()


def evolve_plans(
    engine: Engine,
    plans: tuple[FTFTemplatePlan, ...],
    facts: FTFModelFacts,
    settings: FTFSettings,
    *,
    key: bytes,
    telemetry: bool,
) -> tuple[EvolutionResult, tuple[bytes, ...]]:
    """Advance cavity-driven templates through evolution-owned final spans.

    plans share facts.source and contain absolute common-origin numeric controls.
    settings supplies the resonator parameters used by facts. key is the complete
    caller preparation digest; telemetry enables engine work reports. Each span
    receives the preceding DC density unchanged and all cavity branches start
    from the previous main branch's final field. The main branch is the maximum
    span-start population, ties choosing the lowest DC energy. This is a
    semiclassical approximation, not quantum cavity-state entanglement.

    With no cavity drives, retains the original single evolve call without
    continuation inputs. Undriven templates in mixed acquires also evolve once.
    Returns the final point-aligned EvolutionResult with merged author diagnostics
    and every span reuse digest. Call Engine.reuse for all digests before reusing
    a retained result. Density validation and propagation failures pass through
    unchanged; no normalization, clipping or solver correction is performed.
    No engine model source, drive phase, envelope origin or flux is rebuilt.
    One ADC-owned flat-average cache (at most 64 MiB) serves all g/e points in
    this composition, serially. Return and exception paths release its contents;
    no owner is retained by the engine or long-lived acquisition source.
    """
    if not any(any(plan.cavity_drives) for plan in plans):
        return engine.evolve(
            facts.source,
            tuple(plan.batch for plan in plans),
            cancel=Event(),
            telemetry=telemetry,
            reuse_key=key,
        ), (key,)
    cache = CavityAverageCache()
    try:
        pair = resonators(settings)
        timelines = [
            _CavityTimeline(
                plan.batch,
                segment_spans(plan.batch),
                tuple(
                    CavityDrive(facts.joint, pair, pulses, cache=cache)
                    for pulses in plan.cavity_drives
                ),
                np.repeat(
                    facts.source.initial_density[None], len(plan.batch.points), axis=0
                ),
                np.zeros((len(plan.batch.points), 2), dtype=np.complex128),
            )
            for plan in plans
            if any(plan.cavity_drives)
        ]
        return _evolve_cavities(
            engine, plans, facts, timelines, key=key, telemetry=telemetry
        )
    finally:
        cache.clear()


def _evolve_cavities(
    engine: Engine,
    plans: tuple[FTFTemplatePlan, ...],
    facts: FTFModelFacts,
    timelines: list[_CavityTimeline],
    *,
    key: bytes,
    telemetry: bool,
) -> tuple[EvolutionResult, tuple[bytes, ...]]:
    plain = tuple(plan.batch for plan in plans if not any(plan.cavity_drives))
    paths = []
    reuse_keys = []
    output_points = []
    output_densities = []
    if plain:
        plain_key = sha256(b"ftf-no-cavity\0" + key).digest()
        result = engine.evolve(
            facts.source,
            plain,
            cancel=Event(),
            telemetry=telemetry,
            reuse_key=plain_key,
        )
        paths.append(result.paths)
        reuse_keys.append(plain_key)
        output_points.append(result.points)
        output_densities.append(result.densities)
    points = np.concatenate([plan.batch.points for plan in plans])
    steps = max(
        (len(spans) for timeline in timelines for spans in timeline.spans), default=0
    )
    for step in range(steps):
        batches = tuple(timeline.advance(step) for timeline in timelines)
        reuse_key = _span_key(key, step, batches)
        result = engine.evolve(
            facts.source,
            batches,
            cancel=Event(),
            telemetry=telemetry,
            reuse_key=reuse_key,
        )
        paths.append(result.paths)
        reuse_keys.append(reuse_key)
        offset = 0
        for timeline in timelines:
            count = len(timeline.batch.points)
            timeline.density = result.densities[offset : offset + count]
            offset += count
    output_points.extend(timeline.batch.points for timeline in timelines)
    output_densities.extend(timeline.density for timeline in timelines)
    played_points = np.concatenate(output_points)
    order = np.argsort(played_points)
    positions = order[np.searchsorted(played_points[order], points)]
    return (
        EvolutionResult(
            points,
            np.concatenate(output_densities)[positions],
            merge_path_diagnostics(paths),
        ),
        tuple(reuse_keys),
    )
