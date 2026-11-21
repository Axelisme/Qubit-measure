"""Sequential FTF density and dominant-branch cavity memory composition."""

from collections.abc import Callable, Generator, Mapping
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass, field, replace
from hashlib import sha256

import numpy as np
from numpy.typing import NDArray

from zcu_tools.simulate.adc import (
    CavityAverageCache,
    ProjectionReadout,
    normalize_populations,
)
from zcu_tools.simulate.boundary import CancellationQuery
from zcu_tools.simulate.evolution import (
    Cancelled,
    CaptureBuffer,
    Engine,
    EvolutionResult,
    FluxSchedule,
    FluxWaveform,
    NoValidKernel,
    PathDiagnostics,
    SegmentShift,
    StarkOrderInversion,
    TimelineBatch,
    merge_path_diagnostics,
    read_capture_metadata,
    segment_spans,
)
from zcu_tools.simulators.ftf.config import FTFSettings
from zcu_tools.simulators.ftf.environment.numeric import EVOLUTION_BUDGET, resonators
from zcu_tools.simulators.ftf.environment.template import FTFModelFacts, FTFTemplatePlan
from zcu_tools.simulators.ftf.physics import CavityDrive


class ProjectiveCapture:
    """FTF-owned provenance alongside a single acquire's Engine recordings."""

    def __init__(self, buffer: CaptureBuffer) -> None:
        self._buffer = buffer
        self.spans: dict[str, object] = {}

    @contextmanager
    def recording(
        self,
        round_index: int,
        point: int,
        span: tuple[float, float],
        histories: tuple[tuple[int, tuple[int, ...]], ...],
    ) -> Generator[None]:
        before = len(self._buffer.records)
        try:
            yield
        finally:
            records = self._buffer.records
            if len(records) > before:
                capture_id = read_capture_metadata(records[before])["capture_id"]
                self.spans[capture_id] = {
                    "round_index": round_index,
                    "point": point,
                    "span_ns": span,
                    "histories": {
                        str(row_id): history for row_id, history in histories
                    },
                }


@dataclass(frozen=True)
class CavityEvolutionOptions:
    """Controls shared by every span in one cavity composition.

    key: Complete caller preparation digest, including downstream ADC inputs.
    telemetry: Whether evolution collects per-family CPU and routing reports.
    cancel: Read-only query checked between chunks, or None to disable Stop.
        Cancellation raises Cancelled and returns no partial composition.
    evolving: Scope around each Engine.evolve, including failed calls.
    reading: Scope around cavity field propagation and population readout.
        Both scopes default to no observation and must not suppress errors.
    """

    key: bytes
    telemetry: bool
    capture: CaptureBuffer | None = None
    """Acquire-owned buffer forwarded to every evolve, or None to disable recording."""
    cancel: CancellationQuery | None = None
    evolving: Callable[[], AbstractContextManager[None]] = nullcontext
    reading: Callable[[], AbstractContextManager[None]] = nullcontext
    round_index: int = 0
    projective_capture: ProjectiveCapture | None = None


@dataclass
class _CavityTimeline:
    """One template's ragged span cursor, DC density and shared cavity memory."""

    batch: TimelineBatch
    spans: tuple[tuple[tuple[float, float], ...], ...]
    cavities: tuple[CavityDrive, ...]
    density: NDArray[np.complex128]
    fields: NDArray[np.complex128]

    def advance(
        self, step: int, *, points: NDArray[np.int64] | None = None
    ) -> TimelineBatch:
        """Build this span's inputs and advance fields; points can label the solve."""
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
            points=self.batch.points if points is None else points,
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
        digest.update(batch.points.tobytes())
        for shifts in batch.segment_shifts:
            for shift in shifts:
                digest.update(shift.energy_shift_ghz.tobytes())
                digest.update(shift.dephasing_per_ns.tobytes())
    return digest.digest()


@dataclass
class _Outcome:
    timeline: _CavityTimeline
    weight: float
    history: tuple[int, ...]


@dataclass(frozen=True)
class ProjectiveSpan:
    """One solved absolute span and its rows in caller and Engine coordinates.

    point is the original acquisition point. histories pairs each Engine point
    ID with its ordered projected states. Membership and Stark inversions use
    those same Engine IDs, independently of batch or result row order.
    """

    point: int
    span_ns: tuple[float, float]
    histories: tuple[tuple[int, tuple[int, ...]], ...]
    point_members: Mapping[str, tuple[int, ...]]
    stark_order_inversions: tuple[StarkOrderInversion, ...]


@dataclass(frozen=True)
class _ProjectiveSpanResult:
    key: bytes
    paths: PathDiagnostics
    diagnostic: ProjectiveSpan


@dataclass(frozen=True)
class ProjectionEvolution:
    """FTF outcome continuation products in result.points order.

    result: final density mixtures and merged evolution diagnostics.
    readout: normalized weights and conditional populations for every ordered
        logical event, including an unobserved terminal event.
    reuse_keys: all engine continuation digests, including projected densities.
    outcomes: retained history count per point, including zeros when floor=0.
    pruned_weight: cumulative removed probability per point before redistribution.
    spans: author diagnostics for every completed span, with Engine row IDs.
    """

    result: EvolutionResult
    readout: ProjectionReadout
    reuse_keys: tuple[bytes, ...]
    outcomes: tuple[int, ...]
    pruned_weight: tuple[float, ...]
    spans: tuple[ProjectiveSpan, ...]


def _point_batch(batch: TimelineBatch, point: int) -> TimelineBatch:
    row = slice(point, point + 1)
    flux_rows = slice(batch.flux.offsets[point], batch.flux.offsets[point + 1])
    flux = FluxSchedule(
        np.array([0, flux_rows.stop - flux_rows.start], dtype=np.int64),
        batch.flux.start_ns[flux_rows],
        batch.flux.level[flux_rows],
    )
    waveform = batch.flux_waveform
    if waveform is not None:
        samples = slice(waveform.offsets[point], waveform.offsets[point + 1])
        waveform = FluxWaveform(
            np.array([0, samples.stop - samples.start], dtype=np.int64),
            waveform.time_ns[samples],
            waveform.flux[samples],
        )
    return replace(
        batch,
        points=batch.points[row],
        end_ns=batch.end_ns[row],
        tracks=tuple(
            replace(
                track,
                start_ns=track.start_ns[row],
                duration_ns=track.duration_ns[row],
                frequency_ghz=track.frequency_ghz[row],
                phase_rad=track.phase_rad[row],
                amplitude=track.amplitude[row],
                envelope_index=track.envelope_index[row],
                active=track.active[row],
            )
            for track in batch.tracks
        ),
        flux=flux,
        flux_waveform=waveform,
        segment_boundaries=(batch.segment_boundaries[point],),
    )


def _project_outcomes(
    outcomes: list[_Outcome], floor: float
) -> tuple[list[_Outcome], float]:
    expanded = []
    removed = 0.0
    for outcome in outcomes:
        timeline = outcome.timeline
        probabilities = normalize_populations(timeline.density[0].diagonal().real)
        for state, probability in enumerate(probabilities):
            weight = outcome.weight * float(probability)
            if weight < floor:
                removed += weight
                continue
            density = np.zeros_like(timeline.density)
            density[0, state, state] = 1
            expanded.append(
                _Outcome(
                    replace(timeline, density=density, fields=timeline.fields.copy()),
                    weight,
                    (*outcome.history, state),
                )
            )
    if not expanded:
        point = outcomes[0].timeline.batch.points[0]
        raise ValueError(
            f"point {point}: projection_weight_floor {floor} removed every FTF outcome"
        )
    retained = sum(outcome.weight for outcome in expanded)
    for outcome in expanded:
        outcome.weight /= retained
    return expanded, removed


def _point_paths(
    paths: PathDiagnostics, identities: dict[int, tuple[int, tuple[int, ...]]]
) -> PathDiagnostics:
    """Attribute row membership and warnings to acquisition points, not histories."""
    members = {
        name: tuple(sorted({identities[row_id][0] for row_id in ids}))
        for name, ids in paths.point_members.items()
    }
    return replace(
        paths,
        points_by_class={name: len(ids) for name, ids in members.items()},
        point_members=members,
        stark_order_inversions=tuple(
            replace(entry, point=identities[entry.point][0])
            for entry in paths.stark_order_inversions
        ),
    )


@dataclass
class _ProjectionState:
    floor: float
    identities: dict[int, tuple[int, tuple[int, ...]]] = field(default_factory=dict)

    def row_id(self, point: int, history: tuple[int, ...]) -> int:
        # Keep all history lengths representable without truncating state indices.
        identity = np.asarray((point, *history), dtype="<i8")
        digest = sha256(b"ftf-projective-row\0" + identity.tobytes()).digest()
        row_id = int.from_bytes(digest[:8], "little") >> 1
        key = (point, history)
        if self.identities.setdefault(row_id, key) != key:
            raise ValueError(f"FTF projective row ID collision for {key}")
        return row_id


def _evolve_outcomes(
    engine: Engine,
    outcomes: list[_Outcome],
    facts: FTFModelFacts,
    read_starts: tuple[float, ...],
    projection: _ProjectionState,
    *,
    options: CavityEvolutionOptions,
) -> tuple[list[_Outcome], list[_ProjectiveSpanResult], float]:
    results = []
    survival = 1.0
    point = int(outcomes[0].timeline.batch.points[0])
    for step, span in enumerate(outcomes[0].timeline.spans[0]):
        start, end = span
        if start in read_starts:
            with options.reading():
                outcomes, removed = _project_outcomes(outcomes, projection.floor)
            survival *= 1 - removed
        # Engine owns routing usage and results (ADR-0078). Rows have caller
        # identities; the outcome list keeps its original sampling order.
        rows: dict[int, _Outcome] = {}
        with options.reading():
            batches = []
            for outcome in outcomes:
                row_id = projection.row_id(point, outcome.history)
                rows[row_id] = outcome
                batches.append(
                    outcome.timeline.advance(
                        step, points=np.array([row_id], dtype=np.int64)
                    )
                )
        histories = tuple((row_id, outcome.history) for row_id, outcome in rows.items())
        history_bytes = np.asarray(
            [history for _, history in histories], dtype=np.int64
        ).tobytes()
        point_key = sha256(
            b"ftf-projective-span\0"
            + options.key
            + point.to_bytes(8, "little")
            + history_bytes
        ).digest()
        key = _span_key(point_key, step, tuple(batches))
        recording = (
            nullcontext()
            if options.projective_capture is None
            else options.projective_capture.recording(
                options.round_index, point, span, histories
            )
        )
        with recording, options.evolving(), engine.defer_stark_order_warnings():
            try:
                result = engine.evolve(
                    facts.source,
                    batches,
                    budget=EVOLUTION_BUDGET,
                    cancel=options.cancel,
                    telemetry=options.telemetry,
                    capture=options.capture,
                    reuse_key=key,
                )
            except Cancelled:
                raise
            except NoValidKernel as error:
                rejected = [
                    rows[row_id].history for row_id in error.points if row_id in rows
                ]
                raise NoValidKernel(
                    f"FTF projective point {point} span [{start:g}, {end:g}] ns "
                    f"histories {rejected}: {error}",
                    domain=error.domain,
                    points=(point,),
                    segments=error.segments,
                    attempted=error.attempted,
                ) from error
            except Exception as error:
                error.add_note(
                    f"FTF projective point {point} span [{start:g}, {end:g}] ns "
                    f"batch histories {[history for _, history in histories]}"
                )
                raise
        for row_id, density in zip(result.points, result.densities, strict=True):
            rows[row_id].timeline.density = density[None]
        results.append(
            _ProjectiveSpanResult(
                key,
                result.paths,
                ProjectiveSpan(
                    point,
                    span,
                    histories,
                    result.paths.point_members,
                    result.paths.stark_order_inversions,
                ),
            )
        )
    return outcomes, results, 1 - survival


def evolve_projective_plans(
    engine: Engine,
    plans: tuple[FTFTemplatePlan, ...],
    facts: FTFModelFacts,
    settings: FTFSettings,
    *,
    options: CavityEvolutionOptions,
) -> ProjectionEvolution:
    """Project at each read start and continue each history through cavity spans.

    Plans use one common ordered event topology. Each outcome owns its DC
    density and semiclassical dominant-state cavity field; herald pulses drive
    those fields on the same span path as untriggered cavity pulses. Rank-1
    projection uses retained DC energy states, not IQ clusters. ADC CTMC jumps
    affect IQ only and never feed back into this continuation.

    Low-weight histories are pruned at settings.numerics.projection_weight_floor
    and retained weights redistributed proportionally after each projection.
    A zero floor retains even zero-weight histories. Removing every outcome
    raises ValueError naming its point and floor. Engine failures include the
    original point, absolute span, and history context. Cancellation and errors
    return no partial result. Each evolve batches the
    current absolute span of one point. A row ID identifies the original point
    and history across spans and rounds. Densities return by ID before the next span.
    One acquire-local flat cavity-average cache is cleared on success or failure.
    """
    points = np.concatenate([plan.batch.points for plan in plans])
    count = facts.source.initial_density.shape[0]
    rows = []
    spans = []
    pruned = []
    projection = _ProjectionState(settings.numerics.projection_weight_floor)
    cache = CavityAverageCache()
    try:
        pair = resonators(settings)
        for plan in plans:
            for point in range(len(plan.batch.points)):
                batch = _point_batch(plan.batch, point)
                reads = plan.projective_reads[point]
                pulses = (
                    *plan.cavity_drives[point],
                    *(p for read in reads for p in read.pulses),
                )
                timeline = _CavityTimeline(
                    batch,
                    segment_spans(batch),
                    (CavityDrive(facts.joint, pair, pulses, cache=cache),),
                    facts.source.initial_density[None].copy(),
                    np.zeros((1, 2), dtype=np.complex128),
                )
                outcomes, results, removed = _evolve_outcomes(
                    engine,
                    [_Outcome(timeline, 1.0, ())],
                    facts,
                    tuple(read.start_ns for read in reads),
                    projection,
                    options=options,
                )
                rows.append(outcomes)
                spans.extend(results)
                pruned.append(removed)
    finally:
        cache.clear()
    width = max(map(len, rows))
    weights = np.zeros((len(rows), width))
    events = len(rows[0][0].history) + 1
    populations = [np.zeros((len(rows), width, count)) for _ in range(events)]
    for values in populations:
        values[..., 0] = 1  # Valid populations for zero-weight padding.
    densities = np.zeros((len(rows), count, count), dtype=np.complex128)
    for point, outcomes in enumerate(rows):
        for index, outcome in enumerate(outcomes):
            weights[point, index] = outcome.weight
            for event, state in enumerate(outcome.history):
                populations[event][point, index] = np.eye(count)[state]
            with options.reading():
                populations[-1][point, index] = normalize_populations(
                    outcome.timeline.density[0].diagonal().real
                )
            densities[point] += outcome.weight * outcome.timeline.density[0]
    paths = _point_paths(
        merge_path_diagnostics([span.paths for span in spans]), projection.identities
    )
    return ProjectionEvolution(
        EvolutionResult(points, densities, paths),
        ProjectionReadout(weights, tuple(populations)),
        tuple(span.key for span in spans),
        tuple(map(len, rows)),
        tuple(pruned),
        tuple(span.diagnostic for span in spans),
    )


def evolve_plans(
    engine: Engine,
    plans: tuple[FTFTemplatePlan, ...],
    facts: FTFModelFacts,
    settings: FTFSettings,
    *,
    options: CavityEvolutionOptions,
) -> tuple[EvolutionResult, tuple[bytes, ...]]:
    """Advance cavity-driven templates through evolution-owned final spans.

    plans share facts.source and contain absolute common-origin numeric controls.
    settings supplies the resonator parameters used by facts. options supplies
    the preparation digest, telemetry switch and read-only cancel query. Each span
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
    options.cancel is forwarded to every evolution span's chunk checkpoints; None
    disables cancellation. Cancelled propagates without a partial result.
    One ADC-owned cache (at most 64 MiB for moments and all-state end fields)
    serves all g/e points in this composition, serially. Return and exception
    paths release its contents;
    no owner is retained by the engine or long-lived acquisition source.
    """
    if not any(any(plan.cavity_drives) for plan in plans):
        with options.evolving():
            return engine.evolve(
                facts.source,
                tuple(plan.batch for plan in plans),
                budget=EVOLUTION_BUDGET,
                cancel=options.cancel,
                telemetry=options.telemetry,
                capture=options.capture,
                reuse_key=options.key,
            ), (options.key,)
    # ADR-0078: this composition alone owns the ADC memo's lifetime.
    cache = CavityAverageCache()
    try:
        pair = resonators(settings)
        timelines = [
            _CavityTimeline(
                plan.batch,
                segment_spans(plan.batch),
                tuple(
                    CavityDrive(facts.joint, pair, pulses, span_cache=cache)
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
        return _evolve_cavities(engine, plans, facts, timelines, options=options)
    finally:
        cache.clear()


def _evolve_cavities(
    engine: Engine,
    plans: tuple[FTFTemplatePlan, ...],
    facts: FTFModelFacts,
    timelines: list[_CavityTimeline],
    *,
    options: CavityEvolutionOptions,
) -> tuple[EvolutionResult, tuple[bytes, ...]]:
    plain = tuple(plan.batch for plan in plans if not any(plan.cavity_drives))
    paths = []
    reuse_keys = []
    output_points = []
    output_densities = []
    if plain:
        plain_key = sha256(b"ftf-no-cavity\0" + options.key).digest()
        with options.evolving():
            result = engine.evolve(
                facts.source,
                plain,
                budget=EVOLUTION_BUDGET,
                cancel=options.cancel,
                telemetry=options.telemetry,
                capture=options.capture,
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
        with options.reading():
            batches = tuple(timeline.advance(step) for timeline in timelines)
        reuse_key = _span_key(options.key, step, batches)
        with options.evolving(), engine.defer_stark_order_warnings():
            result = engine.evolve(
                facts.source,
                batches,
                budget=EVOLUTION_BUDGET,
                cancel=options.cancel,
                telemetry=options.telemetry,
                capture=options.capture,
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
