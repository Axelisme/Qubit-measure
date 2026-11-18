"""Acquire-local dimension policy and terminal density pilot selection."""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from hashlib import sha256

import numpy as np
from numpy.typing import NDArray

from zcu_tools.simulate.evolution import (
    Engine,
    EvolutionResult,
    FluxSchedule,
    FluxWaveform,
    ModelSource,
    TimelineBatch,
    merge_path_diagnostics,
)
from zcu_tools.simulators.ftf.environment.cavity import CavityEvolutionOptions
from zcu_tools.simulators.ftf.environment.numeric import EVOLUTION_BUDGET
from zcu_tools.simulators.ftf.environment.template import FTFModelFacts, FTFTemplatePlan

# A short sequence does not amortize the dimension check over enough points.
_MIN_POINTS = 8
# Keep at least the smallest practical FTF evolution basis.
_MIN_DIMENSION = 12
# Leave half of the existing FTF Engine allowance for in-block error.
_DROP_BUDGET = 0.5 * EVOLUTION_BUDGET


@dataclass(frozen=True)
class DimensionSelection:
    """Selected state count and the diagnostics for that acquire's pilot.

    pilot_point is the zero-based C-order point in the acquire, or None if
    skipped. drop is the largest discarded density element magnitude, zero
    at the upper basis and None without a pilot. error is the zero-padded
    verification's maximum absolute density error, or None if not attempted.
    reused is true when this acquire uses a prior acquire's selection.
    """

    dimension: int
    pilot_point: int | None = None
    drop: float | None = None
    error: float | None = None
    reused: bool = False


def pilot_size(points: int, upper: int) -> bool:
    """Whether a request is large enough in points and dimension to justify a pilot."""
    return upper > _MIN_DIMENSION and points >= _MIN_POINTS


def can_pilot(plans: tuple[FTFTemplatePlan, ...]) -> bool:
    """Whether these timelines support one complete density per pilot evolve."""
    return not any(
        any(plan.projective_reads) or any(plan.cavity_drives) for plan in plans
    )


def _pilot_location(
    batches: Sequence[TimelineBatch], source: ModelSource
) -> tuple[int, int]:
    # Measure pulse area in play, scaling by the calibrated port's largest
    # matrix element. Ties prefer the largest detuning from the strongest
    # ground transition of an active port, then the lowest C-order point.
    dc = source.model((0,) * batches[0].flux.level.shape[1])
    ports = {}
    for drive in dc.drives:
        magnitude = np.hypot(np.abs(drive.in_phase_ghz), np.abs(drive.quadrature_ghz))
        target = int(np.argmax(magnitude[1:, 0])) + 1
        ports[drive.name] = (
            float(magnitude.max()),
            dc.energies_ghz[target] - dc.energies_ghz[0],
        )
    scores = []
    for index, batch in enumerate(batches):
        for row, point in enumerate(batch.points):
            area = detuning = 0.0
            for track in batch.tracks:
                duration = min(
                    track.duration_ns[row], batch.end_ns[row] - track.start_ns[row]
                )
                if not track.active[row] or duration <= 0 or not track.amplitude[row]:
                    continue
                scale, transition = ports[track.port]
                envelope = track.envelopes[track.envelope_index[row]]
                if envelope.kind == "constant":
                    envelope_area = duration
                else:
                    times = np.r_[
                        envelope.sample_ns[envelope.sample_ns < duration], duration
                    ]
                    envelope_area = np.trapezoid(np.abs(envelope.values(times)), times)
                area += abs(track.amplitude[row]) * scale * envelope_area
                if scale and envelope_area:
                    detuning = max(
                        detuning, abs(abs(track.frequency_ghz[row]) - transition)
                    )
            scores.append((area, detuning, -int(point), index, row))
    _, _, _, index, row = max(scores)
    return index, row


def _slice_batch(batch: TimelineBatch, rows: slice) -> TimelineBatch:
    first, stop, _ = rows.indices(len(batch.points))
    flux = batch.flux
    start = flux.offsets[first]
    flat = slice(start, flux.offsets[stop])
    waveform = batch.flux_waveform
    if waveform is not None:
        start_sample = waveform.offsets[first]
        samples = slice(start_sample, waveform.offsets[stop])
        waveform = FluxWaveform(
            waveform.offsets[first : stop + 1] - start_sample,
            waveform.time_ns[samples],
            waveform.flux[samples],
        )
    return replace(
        batch,
        points=batch.points[rows],
        end_ns=batch.end_ns[rows],
        tracks=tuple(
            replace(
                track,
                start_ns=track.start_ns[rows],
                duration_ns=track.duration_ns[rows],
                frequency_ghz=track.frequency_ghz[rows],
                phase_rad=track.phase_rad[rows],
                amplitude=track.amplitude[rows],
                envelope_index=track.envelope_index[rows],
                active=track.active[rows],
            )
            for track in batch.tracks
        ),
        flux=FluxSchedule(
            flux.offsets[first : stop + 1] - start,
            flux.start_ns[flat],
            flux.level[flat],
        ),
        flux_waveform=waveform,
        segment_boundaries=batch.segment_boundaries[rows],
    )


def _isolated_evolve(
    engine: Engine,
    source: ModelSource,
    batch: TimelineBatch,
    options: CavityEvolutionOptions,
) -> EvolutionResult:
    # Single-row pricing must not increment the production routing inventory.
    isolated = ModelSource(
        source.initial_density,
        source.model,
        source.flux_expansion(),
        slow_envelope_segments_per_period=source.slow_envelope_segments_per_period,
    )
    with options.evolving():
        return engine.evolve(
            isolated,
            (batch,),
            budget=EVOLUTION_BUDGET,
            cancel=options.cancel,
            telemetry=options.telemetry,
            capture=options.capture,
        )


def _candidate(density: NDArray[np.complex128]) -> tuple[int, float]:
    size = len(density)
    magnitudes = np.abs(density)
    # Every entry touching an omitted row OR column is zero after embedding.
    edges = np.maximum(magnitudes.max(axis=0), magnitudes.max(axis=1))
    drops = np.maximum.accumulate(edges[::-1])[::-1]
    for k in range(_MIN_DIMENSION, size):
        if drops[k] <= _DROP_BUDGET:
            return k, float(drops[k])
    return size, 0.0


def pilot_dimension(
    engine: Engine,
    plans: tuple[FTFTemplatePlan, ...],
    facts: FTFModelFacts,
    *,
    options: CavityEvolutionOptions,
) -> tuple[DimensionSelection, ModelSource, EvolutionResult]:
    """Run one upper reference and at most one smaller verification.

    Call for a pilot-eligible first round. Returns the selection, the model
    source for the remaining batch, and the upper result for the pilot point.
    The smaller verification is never used as production output.
    """
    batches = tuple(plan.batch for plan in plans)
    index, row = _pilot_location(batches, facts.source)
    batch = _slice_batch(batches[index], slice(row, row + 1))
    point = int(batch.points[0])
    reference = _isolated_evolve(engine, facts.source, batch, options)
    upper = reference.densities[0]
    k, drop = _candidate(upper)
    if k == len(upper):
        return (
            DimensionSelection(k, pilot_point=point, drop=drop),
            facts.source,
            reference,
        )
    reduced = facts.evolution_source(k)
    estimate = _isolated_evolve(engine, reduced, batch, options).densities[0]
    padded = np.zeros_like(upper)
    padded[:k, :k] = estimate
    error = float(np.max(np.abs(padded - upper)))
    if error <= EVOLUTION_BUDGET:
        selection = DimensionSelection(k, pilot_point=point, drop=drop, error=error)
        return selection, reduced, reference
    selection = DimensionSelection(len(upper), pilot_point=point, drop=0.0, error=error)
    return selection, facts.source, reference


def _without_reference(
    batches: tuple[TimelineBatch, ...], point: int
) -> tuple[TimelineBatch, ...]:
    kept = []
    for batch in batches:
        rows = np.flatnonzero(batch.points == point)
        if not len(rows):
            kept.append(batch)
            continue
        row = int(rows[0])
        if row:
            kept.append(_slice_batch(batch, slice(0, row)))
        if row + 1 < len(batch.points):
            kept.append(_slice_batch(batch, slice(row + 1, None)))
    return tuple(kept)


def evolve_terminal(
    engine: Engine,
    source: ModelSource,
    plans: tuple[FTFTemplatePlan, ...],
    *,
    options: CavityEvolutionOptions,
    upper: int,
    reference: EvolutionResult | None,
) -> tuple[EvolutionResult, tuple[bytes, ...]]:
    """Evolve terminal timelines, embedding in the upper basis, with route digests.

    reference is the isolated upper result for one point, or None to solve the
    whole batch. If supplied, only the other points register production routing
    facts. The split and full batches use different routing digests. Points keep
    their original C-order IDs.
    """
    batches = tuple(plan.batch for plan in plans)
    key = options.key
    if reference is not None:
        batches = _without_reference(batches, int(reference.points[0]))
        key = sha256(b"ftf-terminal-split\0" + key).digest()
    with options.evolving():
        result = engine.evolve(
            source,
            batches,
            budget=EVOLUTION_BUDGET,
            cancel=options.cancel,
            telemetry=options.telemetry,
            capture=options.capture,
            reuse_key=key,
        )
    result = embed_upper(result, upper)
    if reference is not None:
        result = EvolutionResult(
            np.concatenate([result.points, reference.points]),
            np.concatenate([result.densities, reference.densities]),
            merge_path_diagnostics((result.paths, reference.paths)),
        )
    return result, (key,)


def embed_upper(result: EvolutionResult, upper: int) -> EvolutionResult:
    """Zero-pad both density axes to upper, preserving point and basis order."""
    n = result.densities.shape[-1]
    if n == upper:
        return result
    densities = np.zeros((len(result.points), upper, upper), dtype=np.complex128)
    densities[:, :n, :n] = result.densities
    return replace(result, densities=densities)
