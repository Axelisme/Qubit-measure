# tolerated(file-size): FTF acquire assembly shares one source's Engine, effective
# settings, preparation cache and current capture/telemetry state. Keeping round
# preparation, invalidation and completion together preserves their lifetime rules.
"""FTF round snapshots, persistent engine ownership and compact readout sampling."""

import time
import warnings
from collections import OrderedDict
from collections.abc import Collection, Generator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from hashlib import sha256
from math import prod, sqrt
from pathlib import Path
from weakref import ReferenceType, ref

import numpy as np
from qick import QickConfig

from zcu_tools.device import DeviceManager
from zcu_tools.device.fake import FakeDevice, FakeDeviceInfo
from zcu_tools.experiment.simulation import EnvironmentWarmup, SimulationEnvironment
from zcu_tools.program.v2.simulation import (
    AcquireOutcome,
    IntegratedRound,
    SimulationBinding,
    SimulationCallbacks,
    SimulationClock,
    SimulationSoc,
    SimulationTiming,
)
from zcu_tools.simulate.adc import (
    ChannelReadout,
    ConditionalReadout,
    ConditionalSums,
    PopulationValidationError,
    PreparedRawShots,
    PreparedStatistics,
    ProjectionReadout,
    ReadoutBatch,
    normalize_populations,
    sample_traces,
)
from zcu_tools.simulate.boundary import (
    AcquireRequest,
    CancellationQuery,
    ChannelSignals,
    EnvironmentSnapshot,
    RoundRequest,
    RoundSignals,
    SamplingIdentity,
)
from zcu_tools.simulate.evolution import (
    Basis,
    CaptureBuffer,
    Engine,
    EvolutionResult,
    ModelSource,
    PathDiagnostics,
    RoutingHistory,
    batch_populations,
)
from zcu_tools.simulators.capture import AcquireCaptureSettings
from zcu_tools.simulators.ftf.config import FTFSettings, load_ftf, parse_ftf
from zcu_tools.simulators.ftf.environment.acquisition import FTFAcquisitionOptions
from zcu_tools.simulators.ftf.environment.cavity import (
    CavityEvolutionOptions,
    ProjectionEvolution,
    ProjectiveCapture,
    ProjectiveSpan,
    evolve_plans,
    evolve_projective_plans,
)
from zcu_tools.simulators.ftf.environment.dimension import (
    DimensionSelection,
    can_pilot,
    evolve_terminal,
    pilot_dimension,
    pilot_size,
)
from zcu_tools.simulators.ftf.environment.drift import FTFCoherenceDrift
from zcu_tools.simulators.ftf.environment.numeric import EVOLUTION_BUDGET
from zcu_tools.simulators.ftf.environment.preparation import (
    EnvelopeEncoder,
    PreparationCache,
    preparation_key,
)
from zcu_tools.simulators.ftf.environment.readout import (
    FTFReadoutConfig,
    readout_configuration,
)
from zcu_tools.simulators.ftf.environment.template import (
    BACKEND_VERSION,
    FTFModelFacts,
    FTFTemplatePlan,
    plan_template,
)
from zcu_tools.simulators.telemetry import RoutingTelemetry

# Fixed-size digests and state counts outlive the larger byte-bounded preparations.
_MAX_DIMENSION_RECORDS = 256


@dataclass(frozen=True)
class FTFNumericUse:
    """Author-only latest solve diagnostics, never ordinary acquisition truth.

    acquire_sequence/round_index identify the solve's sampling scope. paths counts
    actual engine segment paths; it contains no densities or noise realizations.
    rejected_rows is zero after successful ADC preparation, or the number of
    rows ADC rejected under its reported validation rule. The record survives
    that failure. A reused snapshot does not create a new solve or change it.
    projection_outcomes counts retained histories per C-order point, not the
    semiclassical cavity main state. projection_pruned_weight gives cumulative
    removed probability before proportional redistribution. Both are empty for
    terminal-only templates. projective_spans locates solved histories and their
    diagnostics at each absolute span. Reuse preserves these author-only values.
    """

    acquire_sequence: int
    round_index: int
    paths: PathDiagnostics
    rejected_rows: int = 0
    projection_outcomes: tuple[int, ...] = ()
    projection_pruned_weight: tuple[float, ...] = ()
    projective_spans: tuple[ProjectiveSpan, ...] = ()


@dataclass(frozen=True)
class _Prepared:
    readout: ProjectionReadout
    channels: tuple[ChannelReadout, ...]
    paths: PathDiagnostics
    sampler: PreparedStatistics | None
    raw: PreparedRawShots
    reuse_keys: tuple[bytes, ...]
    projection_outcomes: tuple[int, ...]
    projection_pruned_weight: tuple[float, ...]
    projective_spans: tuple[ProjectiveSpan, ...]
    split_pilot: bool
    pilot_dimension: int | None


def _readout_distribution(
    evolved: EvolutionResult,
    plans: tuple[FTFTemplatePlan, ...],
    point_dims: tuple[int, ...],
    projected: ProjectionReadout | None,
) -> ProjectionReadout:
    if projected is not None:
        order = np.argsort(evolved.points)
        return ProjectionReadout(
            projected.weights[order].reshape(
                (*point_dims, projected.weights.shape[-1])
            ),
            tuple(
                values[order].reshape((*point_dims, *values.shape[-2:]))
                for values in projected.populations
            ),
        )
    state_count = evolved.densities.shape[-1]
    probabilities = np.empty((prod(point_dims), state_count))
    dc_model = plans[0].source.model((0,) * plans[0].batch.flux.level.shape[1])
    # Evolution and readout use the same retained DC energy coordinates.
    basis = Basis(
        dc_model.identity,
        plans[0].readouts[0].batch.model.state_labels,
        dc_model.eigenvectors,
    )
    populations = batch_populations(evolved.densities, basis, basis)
    probabilities[evolved.points] = normalize_populations(populations)
    return ProjectionReadout.unprojected(
        probabilities.reshape((*point_dims, state_count))
    )


def _same_readout(first: ProjectionReadout, second: ProjectionReadout) -> bool:
    return np.array_equal(first.weights, second.weights) and all(
        np.array_equal(current, old)
        for current, old in zip(first.populations, second.populations, strict=True)
    )


def _validate_event_layouts(plans: tuple[FTFTemplatePlan, ...]) -> None:
    # Single-template planners already require point-consistent event contents.
    layouts = [
        (
            len(plan.projective_reads[0]),
            tuple(
                sorted(
                    (
                        readout.channel,
                        tuple(len(event.grids) for event in readout.batch.events[0]),
                    )
                    for readout in plan.readouts
                )
            ),
        )
        for plan in plans
    ]
    for index, layout in enumerate(layouts[1:], 1):
        if layout != layouts[0]:
            raise ValueError(
                f"templates 0 and {index} have different projective/ADC event layouts: "
                f"{layouts[0]} != {layout}"
            )


def _prepare_channels(
    plans: tuple[FTFTemplatePlan, ...],
    acquire: AcquireRequest,
    settings: FTFSettings,
) -> tuple[tuple[ChannelReadout, ...], dict[int, ReadoutBatch]]:
    channels = []
    raw_batches = {}
    for channel_index, first in enumerate(plans[0].readouts):
        conditionals = []
        traces = []
        raw_events = []
        output_orders = []
        content_index = np.empty(prod(acquire.point_dims), dtype=np.int64)
        offset = 0
        for plan in plans:
            readout = plan.readouts[channel_index].batch
            conditional = ConditionalReadout(
                readout.model,
                rtol=settings.numerics.solver.rtol,
                atol=settings.numerics.solver.atol,
            ).responses(replace(readout, include_traces=acquire.mode == "decimated"))
            conditionals.append(conditional)
            raw_events.extend(readout.events)
            output_orders.append(readout.column_order)
            if conditional.traces is not None:
                traces.append(conditional.traces)
            content_index[plan.batch.points] = offset + readout.content_index
            offset += len(conditional.sums)
        conditional = ConditionalSums(
            np.concatenate([item.sums for item in conditionals]),
            conditionals[0].sample_counts,
            np.concatenate(traces) if traces else None,
            conditionals[0].event_indices,
            np.concatenate(output_orders),
        )
        raw_batches[first.channel] = ReadoutBatch(
            first.batch.model,
            tuple(raw_events),
            content_index,
            output_order=conditional.output_order,
        )
        channels.append(
            ChannelReadout(
                first.channel,
                conditional,
                content_index.reshape(acquire.point_dims),
                first.batch.model.receiver,
            )
        )
    return tuple(channels), raw_batches


class FTFAcquire:
    """One fixed request with at most the immediately preceding round's preparation.

    Construct through FTFSource.open_acquire. Each round draws a fresh stream.
    After a round uses split pilot preparation, the next round solves the full
    batch at the selected dimension without a pilot, even for an equal snapshot.
    Later rounds reuse that full preparation while the snapshot stays equal.
    Opening the next acquire invalidates this session and releases its references;
    source-owned preparations and selected-dimension records can survive. Using
    an invalidated session raises RuntimeError. Calls are synchronous and
    must be serialized by the caller (the binding provides acquire admission).
    evolve_wall_s sums Engine.evolve wall seconds across this session's rounds.
    readout_wall_s sums cavity propagation, ADC preparation and round sampling.
    Both include elapsed work in failed calls and stay zero with telemetry off.
    """

    def __init__(
        self,
        source: "FTFSource",
        request: AcquireRequest,
        dimensions: OrderedDict[bytes, int],
        *,
        projective_capture: ProjectiveCapture | None = None,
    ) -> None:
        """Bind the owning source, fixed request, and source's dimension records.

        Use FTFSource.open_acquire to validate the request and manage sessions.
        The source supplies its bounded LRU of pilot-selected state counts,
        keyed by the same preparation digest as its noiseless preparations.
        A record can outlive its preparation: a later acquire then skips only
        the pilot, not evolution. Changes invalidating model/preparation facts,
        preparation failure, and source close clear the records. A round with
        a different snapshot does not publish the old selection for that snapshot.
        """
        self._source = source
        self._dimensions = dimensions
        self._projective_capture = projective_capture
        self._active = True
        self.request = request
        self._snapshot: EnvironmentSnapshot | None = None
        self._prepared: _Prepared | None = None
        self.routing: list[PathDiagnostics] = []
        self._stark_limits: dict[
            tuple[int, int], dict[tuple[int, float, float], float]
        ] = {}
        self.evolve_wall_s = 0.0
        self.readout_wall_s = 0.0
        self._dimension: DimensionSelection | None = None
        self._pilot_facts: ReferenceType[FTFModelFacts] | None = None
        self.pilot_wall_s = 0.0

    @property
    def dimension(self) -> DimensionSelection | None:
        """The first round's selected dimension; None before preparation."""
        return self._dimension

    @contextmanager
    def _timing_pilot(self) -> Generator[None]:
        if self._source.telemetry is None:
            yield
            return
        start = time.monotonic()
        try:
            yield
        finally:
            self.pilot_wall_s += time.monotonic() - start

    @contextmanager
    def _evolving(self) -> Generator[None]:
        if self._source.telemetry is None:
            yield
            return
        start = time.monotonic()
        try:
            yield
        finally:
            self.evolve_wall_s += time.monotonic() - start

    @contextmanager
    def _reading(self) -> Generator[None]:
        if self._source.telemetry is None:
            yield
            return
        start = time.monotonic()
        try:
            yield
        finally:
            self.readout_wall_s += time.monotonic() - start

    def invalidate(self) -> None:
        """Release deterministic arrays and reject further rounds; safe to repeat."""
        self._active = False
        self._prepared = None
        self._projective_capture = None
        self.routing.clear()
        self._stark_limits.clear()

    def _record_stark_order(self, paths: PathDiagnostics) -> None:
        for entry in paths.stark_order_inversions:
            # Scaling all cavity gains (and inherited fields) by x scales the
            # differential Stark shift by x², for the observed branch history.
            limit = sqrt(entry.max_shift_difference_ghz / entry.shift_difference_ghz)
            sites = self._stark_limits.setdefault(entry.states, {})
            site = (entry.point, *entry.span_ns)
            sites[site] = min(sites.get(site, 1.0), limit)

    def warn_stark_order(self) -> None:
        """Report and clear the Stark limits from this acquire's preparations.

        Each reversed state pair emits one UserWarning, with distinct point
        and span counts and the tightest gain limit across rounds and histories.
        Calling again without further preparation emits nothing.
        """
        limits, self._stark_limits = self._stark_limits, {}
        for (lower, upper), sites in limits.items():
            first, start, end = next(iter(sites))
            points = len({point for point, _, _ in sites})
            warnings.warn(
                f"FTF Stark shift reverses energy ordering of states {lower} and {upper} "
                f"(affected points: {points}, affected segments: {len(sites)}); first at "
                f"point {first} segment [{start:g}, {end:g}] ns. "
                f"Use a common gain scale factor of at most {min(sites.values())} on "
                "the configured cavity drives, holding gain ratios and observed "
                "cavity histories fixed. Results above this limit are outside the "
                "dispersive model's validity.",
                UserWarning,
                stacklevel=3,
            )

    def sample_round(
        self,
        snapshot: EnvironmentSnapshot,
        index: int,
        *,
        cancel: CancellationQuery | None = None,
    ) -> IntegratedRound | RoundSignals:
        """Sample integrated totals or decimated shots for a detached live snapshot.

        index is the zero-based software round index for the fresh RNG stream.
        After split pilot preparation, solve the full batch at the selected
        dimension on the next round even if snapshot is unchanged. Reuse full
        density and conditional responses on later rounds with equal snapshots.
        Numeric/ADC errors propagate without partial results or changing the
        cached snapshot. Debug capture is published at acquire completion.
        Decimated shots are rearranged from point-major C6 order to QICK loop
        order (including the reps axis). Neither mode averages or rounds data.
        cancel is checked between evolution chunks; None disables cancellation.
        Cancelled propagates without retaining a partial preparation.
        """
        owner = self._source
        if not self._active:
            raise RuntimeError("FTF acquire session is no longer active")
        if (
            self._prepared is None
            or snapshot != self._snapshot
            or self._prepared.split_pilot
        ):
            # The first round used a split solve; later rounds need the full batch.
            try:
                prepared = self._reuse_or_prepare(snapshot, index, cancel=cancel)
            except Exception:
                # Do not retain partial products from a failed preparation.
                owner.preparation_cache.clear()
                self._dimensions.clear()
                self._prepared = None
                self._snapshot = None
                raise
            self._prepared, self._snapshot = prepared, snapshot
            self._record_stark_order(prepared.paths)
        elif owner.telemetry is not None:
            self.routing.append(self._prepared.paths.for_reuse(telemetry=True))
        prepared = self._prepared
        acquire = self.request
        with self._reading():
            if acquire.mode == "integrated":
                if prepared.sampler is None:
                    raise RuntimeError("integrated preparation requires a sampler")
                statistics = prepared.sampler.draw(
                    seed=acquire.root_seed,
                    acquire_sequence=acquire.sequence,
                    round_index=index,
                )
                result = IntegratedRound(
                    statistics, prepared.raw.bind(statistics.raw_identity)
                )
            else:
                traces = sample_traces(
                    prepared.readout,
                    dict(prepared.raw.batches),
                    rtol=owner.settings.numerics.solver.rtol,
                    atol=owner.settings.numerics.solver.atol,
                    reps=acquire.reps,
                    identity=SamplingIdentity(
                        acquire.root_seed, (acquire.sequence, index, 0)
                    ),
                )
                channels = []
                for channel in traces.channels:
                    shaped = channel.iq.reshape(
                        (*acquire.point_dims, acquire.reps, *channel.iq.shape[1:])
                    )
                    ordered = np.moveaxis(
                        shaped, len(acquire.point_dims), acquire.avg_level
                    )
                    channels.append(
                        ChannelSignals(
                            channel.channel, ordered.reshape(channel.iq.shape)
                        )
                    )
                result = RoundSignals(tuple(channels))
        return result

    def _reuse_or_prepare(
        self,
        snapshot: EnvironmentSnapshot,
        index: int,
        *,
        cancel: CancellationQuery | None = None,
    ) -> _Prepared:
        owner = self._source
        # Invalidate stale generations before looking up equal-content requests.
        facts = owner.model_facts(snapshot)
        key = preparation_key(
            self.request, owner.readout, encoder=owner.envelope_encoder
        )
        previous = None
        # Only another acquire can consume a retained split preparation. Within
        # this acquire, the next round after a split always solves the full batch.
        if self._pilot_facts is None:
            previous = owner.preparation_cache.get(key)
            dimension = (
                self._dimensions.get(key)
                if previous is None
                else previous.pilot_dimension
            )
            if dimension is not None:
                self._dimension = DimensionSelection(dimension, reused=True)
                self._pilot_facts = ref(facts)
            if previous is not None:
                model = (
                    facts.source
                    if dimension is None
                    else facts.evolution_source(dimension)
                )
                # Reprice the original solve shape before reusing its populations.
                if all(
                    owner.engine.reuse(model, reuse_key, budget=EVOLUTION_BUDGET)
                    for reuse_key in previous.reuse_keys
                ):
                    self._remember_dimension(key, dimension)
                    return self._reuse(previous, index, len(model.initial_density))
        options = CavityEvolutionOptions(
            key,
            owner.telemetry is not None,
            cancel=cancel,
            capture=owner.capture,
            evolving=self._evolving,
            reading=self._reading,
            round_index=index,
            projective_capture=self._projective_capture,
        )
        prepared = self._prepare(snapshot, index, facts, options, previous)
        # A changed live snapshot uses the session's N, but has not validated
        # that N for a future acquire starting at the changed snapshot.
        if self._pilot_facts is None or self._pilot_facts() is facts:
            owner.preparation_cache.put(key, prepared)
            self._remember_dimension(key, prepared.pilot_dimension)
        return prepared

    def _remember_dimension(self, key: bytes, dimension: int | None) -> None:
        if dimension is not None:
            self._dimensions[key] = dimension
            self._dimensions.move_to_end(key)
            if len(self._dimensions) > _MAX_DIMENSION_RECORDS:
                self._dimensions.popitem(last=False)

    def _reuse(self, prepared: _Prepared, index: int, dimension: int) -> _Prepared:
        owner = self._source
        if self._dimension is None:
            self._dimension = DimensionSelection(dimension)
        if owner.telemetry is not None:
            self.routing.append(prepared.paths.for_reuse(telemetry=True))
        owner.evaluations = (
            FTFNumericUse(
                self.request.sequence,
                index,
                prepared.paths,
                projection_outcomes=prepared.projection_outcomes,
                projection_pruned_weight=prepared.projection_pruned_weight,
                projective_spans=prepared.projective_spans,
            ),
        )
        prepared.paths.warn_exceeded()
        return prepared

    def _round_model(
        self,
        plans: tuple[FTFTemplatePlan, ...],
        facts: FTFModelFacts,
        options: CavityEvolutionOptions,
    ) -> tuple[ModelSource, EvolutionResult | None]:
        upper = len(facts.source.initial_density)
        if self._dimension is None:
            if pilot_size(prod(self.request.point_dims), upper) and can_pilot(plans):
                with self._timing_pilot():
                    self._dimension, model, reference = pilot_dimension(
                        self._source.engine, plans, facts, options=options
                    )
                self._pilot_facts = ref(facts)
                return model, reference
            self._dimension = DimensionSelection(upper)
        return facts.evolution_source(self._dimension.dimension), None

    def _evolve(
        self,
        plans: tuple[FTFTemplatePlan, ...],
        facts: FTFModelFacts,
        options: CavityEvolutionOptions,
        model: ModelSource,
        reference: EvolutionResult | None,
    ) -> tuple[EvolutionResult, tuple[bytes, ...], ProjectionEvolution | None]:
        owner = self._source
        settings = owner.effective_settings
        if model is not facts.source or reference is not None:
            evolved, keys = evolve_terminal(
                owner.engine,
                model,
                plans,
                options=options,
                upper=len(facts.source.initial_density),
                reference=reference,
            )
            return evolved, keys, None
        if any(any(plan.projective_reads) for plan in plans):
            projection = evolve_projective_plans(
                owner.engine, plans, facts, settings, options=options
            )
            return projection.result, projection.reuse_keys, projection
        evolved, keys = evolve_plans(
            owner.engine, plans, facts, settings, options=options
        )
        return evolved, keys, None

    def _prepare(
        self,
        snapshot: EnvironmentSnapshot,
        index: int,
        facts: FTFModelFacts,
        options: CavityEvolutionOptions,
        previous: _Prepared | None,
    ) -> _Prepared:
        acquire = self.request
        owner = self._source
        plans = tuple(
            plan_template(
                owner.effective_settings,
                snapshot,
                template,
                time_base=acquire.time_base,
                soccfg=owner.soccfg,
                readout=owner.readout,
                facts=facts,
            )
            for template in acquire.templates
        )
        _validate_event_layouts(plans)
        model, reference = self._round_model(plans, facts, options)
        evolved, reuse_keys, projection = self._evolve(
            plans, facts, options, model, reference
        )
        if owner.telemetry is not None:
            self.routing.append(evolved.paths)
        # Retain solve diagnostics even when ADC preparation rejects a row.
        order = np.argsort(evolved.points)
        outcome_counts = (
            () if projection is None else tuple(projection.outcomes[p] for p in order)
        )
        pruned_weight = (
            ()
            if projection is None
            else tuple(projection.pruned_weight[p] for p in order)
        )
        report = FTFNumericUse(
            acquire.sequence,
            index,
            evolved.paths,
            projection_outcomes=outcome_counts,
            projection_pruned_weight=pruned_weight,
            projective_spans=() if projection is None else projection.spans,
        )
        owner.evaluations = (report,)
        with self._reading():
            try:
                distribution = _readout_distribution(
                    evolved,
                    plans,
                    acquire.point_dims,
                    None if projection is None else projection.readout,
                )
            except PopulationValidationError as error:
                owner.evaluations = (
                    replace(report, rejected_rows=error.rejected_rows),
                )
                raise
            if previous is None:
                channels, raw_batches = _prepare_channels(
                    plans, acquire, owner.settings
                )
            else:
                channels, raw_batches = previous.channels, dict(previous.raw.batches)
            if previous is not None and _same_readout(distribution, previous.readout):
                sampler = previous.sampler
                raw = previous.raw
            else:
                sampler = (
                    PreparedStatistics(
                        distribution,
                        tuple(channels),
                        reps=acquire.reps,
                    )
                    if acquire.mode == "integrated"
                    else None
                )
                raw = PreparedRawShots(
                    distribution,
                    raw_batches,
                    reps=acquire.reps,
                    rtol=owner.settings.numerics.solver.rtol,
                    atol=owner.settings.numerics.solver.atol,
                )
        return _Prepared(
            distribution,
            tuple(channels),
            evolved.paths.for_reuse(telemetry=False),
            sampler,
            raw,
            reuse_keys,
            outcome_counts,
            pruned_weight,
            report.projective_spans,
            reference is not None,
            len(model.initial_density) if self._pilot_facts is not None else None,
        )


class FTFSource:
    """Environment coordinator with one persistent Engine and one current acquire.

    soccfg is the required acquisition board profile, supplied by the caller.
    Its actual PFB clocks select mux carriers. Keep it unchanged for this source's
    lifetime; construct a new source when changing board capabilities.
    settings is a validated immutable FTF resource; devices supplies startup DC
    identities whose live values are read by snapshot, even after session
    disconnects or registry renames. engine optionally supplies a configured numerical
    executor; this source takes ownership, including close(). Without it, use
    workers=1 and a 4 MiB routing-facts allowance. readout optionally supplies
    detached author capabilities; None adapts settings.

    open_acquire drops the previous session's preparation references. execute
    opens a session only when its request object changes, preserving preparation
    across equal-snapshot full-batch rounds. The engine and routing facts survive
    acquires.
    preparation_cache owns at most 4 MiB of read-only probabilities, conditionals,
    content mappings, path diagnostics and deterministic integrated samplers.
    Exact current C1 content, layout, mode, reps and readout values select entries.
    History-sensitive hits ask Engine to reprice cached immutable routing facts.
    Only a changed selection or evicted facts evolves and refreshes populations;
    stable hits retain populations, channels, samplers and the preparation entry.
    Pilot preparations retain N. An independent bounded LRU keeps the selected N
    after a preparation is evicted, so a repeat skips the pilot on a miss.
    Physical/numeric generation and engine replacement clear both caches before
    lookup. Its public max_bytes may be adjusted between calls; zero disables
    preparation residency but not remembered dimensions.
    Cumulative builds/hits and resident_bytes report reuse, never noisy results
    or request sampling identities. A hit updates this acquire's numeric use and
    reemits any report-only accuracy warning. Failed preparation clears both
    caches. Active sessions/caller references are outside budgets.
    At most one complete snapshot's physical model survives across acquires;
    full settings and numeric revisions also participate in its exact key.
    Conditional ADC solvers live only during preparation; the complete preparation
    retains their deterministic sums. close releases physical facts and complete
    preparations, invalidates the session and
    closes the engine. envelope_encoder owns at most 32 MiB of immutable
    PulseEnvelope content fingerprints, including full retained envelopes and
    Python containers. Its cumulative builds/hits and resident_bytes report
    fingerprint reuse; max_bytes is adjustable between calls. Borrowed arrays
    are always reread. Fingerprints contain no physical generation data and
    survive generation changes; preparation invalidation is unchanged. Close
    releases all fingerprints and retained envelopes.
    capture is the current acquire's CaptureBuffer, or None when debug_capture
    is disabled. begin_acquire creates it; all spans borrow it. finish_acquire
    publishes once per observer call, before any telemetry summary.
    telemetry optionally supplies an already configured local routing writer.
    Enabled sources collect stage wall and calling-thread routing CPU, and expose
    finish_acquire for the binding completion callback. None (default) creates no
    directory, reads no telemetry clocks, and retains no routing reports.
    acquisition supplies FTFAcquisitionOptions for the environment clock and
    private drift truth log; None uses its defaults. acquisition.clock=None
    creates a clock. begin_acquire freezes losses once per acquisition.
    acquisition.truth_log optionally selects
    the private JSONL path; None uses ~/.local/state/zcu-tools/simulation/ftf/
    <session>/truth.jsonl. Enabled drift writes one row per loop per begin and log
    I/O errors propagate. Disabled drift creates no file or drift RNG streams.
    history is borrowed by every rebuilt ModelSource; None creates owned
    histories. Clearing it invalidates complete-preparation reuse, not sampling.
    Calls are synchronous, serialized by SimulationBinding, not thread-safe.
    """

    def __init__(  # noqa: PLR0913  # tolerated: seven independent inputs with distinct owners (settings, snapshot/devices, template, time base, readout calibration, model facts, board profile); pairing any two would add a class with one caller
        self,
        settings: FTFSettings,
        devices: DeviceManager,
        *,
        soccfg: QickConfig,
        engine: Engine | None = None,
        acquisition: FTFAcquisitionOptions | None = None,
        readout: FTFReadoutConfig | None = None,
        telemetry: RoutingTelemetry | None = None,
        history: RoutingHistory | None = None,
    ) -> None:
        options = FTFAcquisitionOptions() if acquisition is None else acquisition
        self.readout = readout_configuration(settings) if readout is None else readout
        self.settings = settings
        self.soccfg = soccfg
        self.devices = devices
        # The physical source survives session disconnects and registry renames.
        self._source_devices = devices.get_all_devices()
        self._effective_settings = settings
        self._clock = SimulationClock() if options.clock is None else options.clock
        self._truth_log = options.truth_log
        self._drift: FTFCoherenceDrift | None = None
        self._pending_acquire = False
        self.telemetry = telemetry
        self._history = history
        self.engine = Engine(1) if engine is None else engine
        self.evaluations: tuple[FTFNumericUse, ...] = ()
        self._session: FTFAcquire | None = None
        self._facts: FTFModelFacts | None = None
        self.preparation_cache: PreparationCache[_Prepared] = PreparationCache()
        self._dimensions: OrderedDict[bytes, int] = OrderedDict()
        self.envelope_encoder = EnvelopeEncoder()
        self._preparation_engine = self.engine
        self.capture: CaptureBuffer | None = None
        self._capture_settings: AcquireCaptureSettings | None = None
        self._projective_capture: ProjectiveCapture | None = None
        self._closed = False

    @property
    def effective_settings(self) -> FTFSettings:
        """Current frozen losses, with drift disabled after an enabled begin.

        Before begin, returns configured settings. Disabled drift follows the
        source's settings unchanged, including author updates between acquires.
        """
        return (
            self.settings if self.settings.drift is None else self._effective_settings
        )

    @property
    def identity(self) -> str:
        """SHA256 of the current effective settings, used by round snapshots."""
        return sha256(self.effective_settings.model_dump_json().encode()).hexdigest()

    def begin_acquire(self, *, seed: int | None = None) -> None:
        """Freeze f1/coupler/f2 losses at the shared clock's current seconds.

        seed supplies the binding root seed on first use; None uses numerics.seed
        (or fresh entropy). Subsequent calls keep those streams and start a new
        acquisition even without a request. All points/rounds share the values.
        Enabled drift writes three private truth rows and invalidates physical
        facts, complete preparations and remembered dimensions. Disabled drift draws no RNG and
        writes nothing. Closed sources raise RuntimeError; log I/O errors propagate.
        """
        if self._closed:
            raise RuntimeError("FTF source is closed")
        if self._session is not None:
            self._session.invalidate()
        if self.settings.drift is not None:
            if self._drift is None:
                self._drift = FTFCoherenceDrift(
                    self.settings,
                    seed=self.settings.numerics.seed if seed is None else seed,
                    truth_log=self._truth_log,
                )
            self._effective_settings = self._drift.sample(self._clock.now())
            if self._facts is not None:
                self._facts.close()
            self._facts = None
            self._clear_preparations()
        self.capture = CaptureBuffer() if self.settings.numerics.debug_capture else None
        self._capture_settings = (
            None
            if self.capture is None
            else AcquireCaptureSettings(self.effective_settings)
        )
        self._projective_capture = (
            None if self.capture is None else ProjectiveCapture(self.capture)
        )
        self._pending_acquire = True

    def model_facts(self, snapshot: EnvironmentSnapshot) -> FTFModelFacts:
        """Return physical inputs for this full snapshot/settings/numeric key.

        Retains only the latest key, dropping the previous model before building
        a replacement. No pulse/content/receiver results or evolved densities are
        cached here. Invalid numeric inputs propagate; a closed source raises.
        """
        if self._closed:
            raise RuntimeError("FTF source is closed")
        if self.engine is not self._preparation_engine:
            self._clear_preparations()
            self._preparation_engine = self.engine
        if self._facts is None or not self._facts.matches(
            self.effective_settings, snapshot
        ):
            self._clear_preparations()
            if self._facts is not None:
                self._facts.close()
            self._facts = None
            self._facts = FTFModelFacts(
                self.effective_settings, snapshot, history=self._history
            )
        return self._facts

    def _clear_preparations(self) -> None:
        self.preparation_cache.clear()
        self._dimensions.clear()

    def snapshot(self) -> EnvironmentSnapshot:
        """Read bound startup DC values once, rejecting missing/invalid sources.

        Source identities survive GUI registry disconnects/name changes; their
        explicitly addressed FakeDevices retain the physical values and units.
        """
        values = []
        for name, source in self.settings.dc_bias.sources.items():
            if name not in self._source_devices:
                raise ValueError(f"dc_bias.sources.{name}: missing startup device")
            info = self._source_devices[name].get_info()
            if not isinstance(info, FakeDeviceInfo) or info.value_unit != source.unit:
                raise ValueError(
                    f"dc_bias.sources.{name}.unit: fixed source unit mismatch"
                )
            if (
                not np.isfinite(info.value)
                or not source.limits[0] <= info.value <= source.limits[1]
            ):
                raise ValueError(f"dc_bias.sources.{name}.value: outside source limits")
            values.append((name, info.value))
        snapshot = EnvironmentSnapshot("ftf-v2", self.identity, tuple(values))
        if self._capture_settings is not None:
            self._capture_settings.record(snapshot, self.effective_settings)
        return snapshot

    def open_acquire(self, request: AcquireRequest) -> FTFAcquire:
        """Start a fixed template request, invalidating/releasing the previous session.

        Requires nonempty C1 templates and the current RNG stream version. Legacy
        execution timelines are not consumed. ValueError rejects unsupported
        inputs before the engine runs; a closed source raises RuntimeError.
        Active pulse and mux readout generators are mutually exclusive across
        all request templates. Mixing them raises ValueError naming both indices.
        Separate requests may select different readout paths. Nonempty charge
        requirements raise ValueError: physical charge admission is not supported.
        """
        if self._closed:
            raise RuntimeError("FTF source is closed")
        if request.charge_requirements:
            raise ValueError("FTF charge drive requirements are not supported")
        if not request.templates:
            raise ValueError("FTF requires timeline templates")
        if request.rng_version != f"PCG64/SeedSequence-v2/numpy-{np.__version__}":
            raise ValueError("FTF RNG version mismatch")
        self.readout.validate_generators(
            self.settings.readout.drive.gen_ch,
            {
                event.gen_ch
                for template in request.templates
                for event in template.events
                if np.any(event.active)
            },
        )
        if not self._pending_acquire:
            self.begin_acquire(seed=request.root_seed)
        self._pending_acquire = False
        if self._session is not None:
            self._session.invalidate()
        self._session = FTFAcquire(
            self, request, self._dimensions, projective_capture=self._projective_capture
        )
        self.evaluations = ()
        return self._session

    def execute(self, request: RoundRequest) -> IntegratedRound | RoundSignals:
        """Execute a detached round with bounded exact noiseless reuse and fresh draws."""
        if self._session is None or self._session.request is not request.acquire:
            self.open_acquire(request.acquire)
        session = self._session
        if session is None:
            raise RuntimeError("FTF acquire was not opened")
        return session.sample_round(
            request.snapshot, request.index, cancel=request.cancel
        )

    def finish_acquire(self, outcome: AcquireOutcome) -> None:
        """Warn about Stark inversions and write a best-effort acquire summary.

        outcome retains captured round inputs (including failed attempts), wall
        seconds including pacing, builder seconds and normal-return status.
        Only the matching current session contributes warnings and routing work.
        Direct callers may supply their own outcome after the final round and cleanup. Debug capture
        keeps detached rounds but omits settings for identities it did not observe.
        The buffer publishes before summary construction, even with telemetry disabled.
        capture_file is the successful basename or None when recording is off or
        publication fails.
        """
        capture_settings, self._capture_settings = self._capture_settings, None
        projective_capture, self._projective_capture = self._projective_capture, None
        capture_file = None
        if self.capture is not None and capture_settings is not None:
            metadata: dict[str, object] = {}
            if projective_capture is not None and projective_capture.spans:
                metadata["ftf"] = {"projective_spans": projective_capture.spans}
            capture_file = capture_settings.write(
                self.capture, outcome, metadata=metadata
            )
        rounds = outcome.rounds
        request = rounds[0].acquire if rounds else None
        session = self._session
        if session is None or session.request is not request:
            session = None
        if session is not None:
            session.warn_stark_order()
        if self.telemetry is None:
            return
        paths = () if session is None else session.routing
        self.telemetry.write(
            {
                "points": prod(request.point_dims) if request is not None else 0,
                "rounds": len(rounds),
                "wall_s": outcome.wall_s,
                "compile_wall_s": outcome.compile_wall_s,
                "evolve_wall_s": 0.0 if session is None else session.evolve_wall_s,
                "readout_wall_s": 0.0 if session is None else session.readout_wall_s,
                "complete": outcome.complete,
                "capture_file": capture_file,
                "extras": {
                    **_dimension_telemetry(session),
                    "ftf.projection_outcomes": list(
                        self.evaluations[-1].projection_outcomes
                        if paths and self.evaluations
                        else ()
                    ),
                    "ftf.projection_pruned_weight": list(
                        self.evaluations[-1].projection_pruned_weight
                        if paths and self.evaluations
                        else ()
                    ),
                },
            },
            paths=paths,
        )

    def close(self) -> None:
        """Release preparations, dimensions, model/source usage and Engine; safe to repeat."""
        if self._session is not None:
            self._session.invalidate()
            self._session = None
        if self._facts is not None:
            self._facts.close()
        self._facts = None
        self._clear_preparations()
        self.envelope_encoder.clear()
        self._closed = True
        self.engine.close()


def _dimension_telemetry(
    session: FTFAcquire | None,
) -> dict[str, int | float | bool | None]:
    dimension = None if session is None else session.dimension
    price = None if dimension is None else dimension.verification_price
    return {
        "ftf.selected_dimension": None if dimension is None else dimension.dimension,
        "ftf.dimension_reused": dimension is not None and dimension.reused,
        "ftf.pilot_point": None if dimension is None else dimension.pilot_point,
        "ftf.pilot_drop": None if dimension is None else dimension.drop,
        "ftf.pilot_error": None if dimension is None else dimension.error,
        "ftf.pilot_wall_s": 0.0 if session is None else session.pilot_wall_s,
        "ftf.pilot_verify_price_skipped": price is not None and price.skipped,
        "ftf.pilot_candidate_dimension": None if price is None else price.candidate,
        "ftf.pilot_candidate_drop": None if price is None else price.candidate_drop,
        "ftf.pilot_upper_dimension": None if price is None else price.upper,
        "ftf.pilot_upper_wall_s": None if price is None else price.upper_wall_s,
        "ftf.pilot_timelines": None if price is None else price.timelines,
        "ftf.pilot_workers": None if price is None else price.workers,
        "ftf.pilot_remaining_waves": None if price is None else price.remaining_waves,
        "ftf.pilot_estimated_verify_s": None if price is None else price.verify_s,
        "ftf.pilot_estimated_savings_s": None if price is None else price.savings_s,
    }


def _routing_telemetry(settings: FTFSettings) -> RoutingTelemetry | None:
    """Resolve the YAML routing-log policy without changing acquisition options."""
    policy = settings.numerics.telemetry
    if not policy.enabled:
        return None
    directory = policy.directory
    if directory is None:
        directory = Path(__file__).resolve().parents[5] / ".sim_telemetry"
    return RoutingTelemetry(directory.absolute())


def _flux_generator_channels(settings: FTFSettings) -> tuple[int, ...]:
    """Collect declared fast-flux generators for DAC admission."""
    return tuple(
        line.gen_ch
        for line in (
            settings.flux_lines.f1,
            settings.flux_lines.coupler,
            settings.flux_lines.f2,
        )
        if line is not None
    )


def ftf(  # noqa: PLR0913  # tolerated: independent optional inputs have distinct owners; grouping any two adds a single-caller class
    configuration: FTFSettings | str | Path,
    *,
    soccfg: QickConfig,
    acquisition: FTFAcquisitionOptions | None = None,
    workers: int = 1,
    warmup: bool | EnvironmentWarmup = False,
    time_scale: float | None = None,
    history: RoutingHistory | None = None,
    families: Collection[str] | None = None,
) -> SimulationEnvironment:
    """Build an FTF environment from physical settings and a caller-owned board.

    Startup creates fixed V/A FakeDevices and applies initial values once. The
    returned manager owns cleanup; builder failures close all devices and preserve
    both original and cleanup errors. soccfg is the required QICK compile profile,
    built by the caller from independent board YAML. It must include all configured
    drive/flux channels, pulse readout and optional mux/PFB routes. The settings
    bind each ADC to its readout generator and list drive/flux DACs separately.
    No channel counts or board settings are inferred from physical wiring.
    The same soccfg supplies the SoC and numeric PFB carrier selection. One
    environment supports separate pulse and mux acquires without reconfiguration.
    The host receives no model truth.
    acquisition selects the optional clock and drift truth log. One engine
    persists across acquires, using workers (default 1) as its sole parallel width.
    Worker threads live only during multi-timeline evolve; dropping the environment
    releases its in-memory routing facts. FTFSource.close explicitly clears it.
    warmup=False starts no background work. True prepares common CPU paths, then
    precompiles the OpenCL program for retained_levels if the device and dimension
    are supported. No GPU item is scheduled without pyopencl or when families
    excludes dop853. A CPU preparation failure stops the manifest before GPU work.
    A failure after GPU detection succeeds also appears in the failed warmup
    status. Neither failure blocks construction or prevents on-demand acquisition.
    An explicit EnvironmentWarmup runs only the caller's items, which must not sample
    or mutate live devices. env.wait_ready() waits before caller timing; acquire
    also waits automatically.
    After acquisitions drain, call env.close_backend to wait for startup work
    and release numeric resources. The caller closes the device manager separately.
    time_scale overrides numerics.time_scale (default 1.0 = physical time)
    for binding rounds, including snapshot/execution time. None uses
    settings; finite nonnegative values override them. Zero disables pacing for
    profiling/benchmarks without changing numeric settings or sampling. Slow rounds
    and failed calls do not wait. Stop interrupts pacing at 10 ms checkpoints.
    Invalid scales raise ValueError.
    numerics.telemetry.enabled defaults to True. Use
    settings.with_telemetry(enabled=False) for profiles/benchmarks without routing
    timers or I/O. numerics.telemetry.directory defaults to None, selecting
    <project root>/.sim_telemetry. Relative paths resolve at startup.
    Each admitted acquire writes one bounded JSONL summary;
    persistence errors warn once without changing acquisition or RNG results.
    acquisition supplies FTFAcquisitionOptions; None uses its defaults.
    acquisition.clock optionally injects one SimulationClock shared by admission
    and round duration accounting; None creates an environment clock. Successful rounds
    credit unscaled physical seconds not spent on the wall clock, even with
    time_scale=0. acquisition.truth_log optionally selects the private drift JSONL file;
    None uses ~/.local/state/zcu-tools/simulation/ftf/<session>/truth.jsonl.
    history is borrowed across model generations; None keeps source-owned history.
    clear() forces a fresh solve without resetting sampling or clocks. families
    restricts every span, including projective continuations; None allows all.
    Unknown names raise at build; unavailable domains raise at acquire. Neither
    routing guards nor applicability checks are bypassed.
    Enabled drift freezes losses and writes three loop rows before each request;
    I/O errors propagate. Disabled drift creates no truth log or drift RNG stream.
    """
    settings = (
        parse_ftf(configuration)
        if isinstance(configuration, FTFSettings)
        else load_ftf(configuration)
    )
    devices = DeviceManager()
    try:
        for name, source in settings.dc_bias.sources.items():
            device = FakeDevice(address=name, fast_mode=True, value_unit=source.unit)
            devices.register_device(name, device)
            device.setup(
                FakeDeviceInfo(
                    address=device.address,
                    value_unit=source.unit,
                    output="on",
                    value=source.initial,
                    rampstep=source.ramp_step,
                ),
                progress=False,
            )
        telemetry = _routing_telemetry(settings)
        options = FTFAcquisitionOptions() if acquisition is None else acquisition
        sim_clock = SimulationClock() if options.clock is None else options.clock
        coordinator = FTFSource(
            settings,
            devices,
            soccfg=soccfg,
            engine=Engine(workers, families=families),
            history=history,
            acquisition=replace(options, clock=sim_clock),
            telemetry=telemetry,
        )
        # ADR-0072: the environment declares the model's physical wiring.
        readout = settings.readout
        routes: tuple[tuple[int, int], ...] = (
            (readout.drive.gen_ch, readout.adc.ro_ch),
        )
        mux = readout.mux
        if mux is not None:
            routes += tuple((mux.gen_ch, adc.ro_ch) for adc in mux.adc)
        from zcu_tools.simulators.ftf.environment.warmup import startup_warmup

        preparation = (
            startup_warmup(
                settings.numerics.truncation.retained_levels,
                families=families,
            )
            if warmup is True
            else (warmup if isinstance(warmup, EnvironmentWarmup) else None)
        )

        def close_backend() -> None:
            # This callback owns only backend cleanup; readiness failures are status values.
            if preparation is not None:
                preparation.wait_ready()
            coordinator.close()

        binding = SimulationBinding(
            coordinator.snapshot,
            coordinator.execute,
            backend_version=BACKEND_VERSION,
            seed=settings.numerics.seed,
            timing=SimulationTiming(
                time_scale=(
                    settings.numerics.time_scale if time_scale is None else time_scale
                ),
                clock=sim_clock,
            ),
            callbacks=SimulationCallbacks(
                wait_ready=None if preparation is None else preparation.wait_ready,
                on_begin=lambda: coordinator.begin_acquire(seed=binding.root_seed),
                on_end=coordinator.finish_acquire
                if coordinator.telemetry is not None or settings.numerics.debug_capture
                else None,
            ),
        )
        soc = SimulationSoc(
            soccfg,
            binding,
            routes=routes,
            drive_generators=(
                *(line.gen_ch for line in settings.drive_lines.values()),
                *_flux_generator_channels(settings),
            ),
        )
        environment = SimulationEnvironment(
            soc,
            soccfg,
            devices,
            "FTF: independent shots, terminal shared-line readout, synchronous acquire/decimated",
            preparation,
            close_backend=close_backend,
        )
        if preparation is not None:
            preparation.start()
        return environment
    except Exception as error:
        try:
            devices.close_all_devices()
        except ExceptionGroup as cleanup:
            raise ExceptionGroup(
                "FTF startup and cleanup failed", [error, cleanup]
            ) from error
        raise
