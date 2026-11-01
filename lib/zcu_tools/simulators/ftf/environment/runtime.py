"""FTF round snapshots, persistent engine ownership and compact readout sampling."""

from dataclasses import dataclass, replace
from hashlib import sha256
from math import prod
from pathlib import Path

import numpy as np
from qick import QickConfig

from zcu_tools.core.simulation import (
    AcquireRequest,
    CancellationQuery,
    ChannelSignals,
    EnvironmentSnapshot,
    RoundRequest,
    RoundSignals,
    SamplingIdentity,
)
from zcu_tools.device import DeviceManager
from zcu_tools.device.fake import FakeDevice, FakeDeviceInfo
from zcu_tools.experiment.simulation import EnvironmentWarmup, SimulationEnvironment
from zcu_tools.program.v2.simulation import (
    IntegratedRound,
    SimulationBinding,
    SimulationCallbacks,
    SimulationClock,
    SimulationSoc,
    SimulationTiming,
)
from zcu_tools.simulators.ftf.config import FTFSettings
from zcu_tools.simulators.ftf.config import load_ftf
from zcu_tools.simulators.ftf.config import parse_ftf
from zcu_tools.simulators.replay import ReplayAcquire
from zcu_tools.simulators.replay import save_replay
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
from zcu_tools.simulate.evolution import (
    Basis,
    Engine,
    EvolutionResult,
    PathDiagnostics,
    batch_populations,
)

from zcu_tools.simulators.ftf.environment.acquisition import FTFAcquisitionOptions
from zcu_tools.simulators.ftf.environment.cavity import CavityEvolutionOptions
from zcu_tools.simulators.ftf.environment.cavity import evolve_plans
from zcu_tools.simulators.ftf.environment.cavity import evolve_projective_plans
from zcu_tools.simulators.ftf.environment.drift import FTFCoherenceDrift
from zcu_tools.simulators.ftf.environment.preparation import EnvelopeEncoder
from zcu_tools.simulators.ftf.environment.preparation import PreparationCache
from zcu_tools.simulators.ftf.environment.preparation import preparation_key
from zcu_tools.simulators.ftf.environment.readout import FTFReadoutConfig
from zcu_tools.simulators.ftf.environment.readout import readout_configuration
from zcu_tools.simulators.ftf.environment.replay import BACKEND_VERSION
from zcu_tools.simulators.ftf.environment.replay import replay_versions
from zcu_tools.simulators.ftf.environment.telemetry import RoutingTelemetry
from zcu_tools.simulators.ftf.environment.template import FTFModelFacts
from zcu_tools.simulators.ftf.environment.template import FTFTemplatePlan
from zcu_tools.simulators.ftf.environment.template import plan_template


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
    terminal-only templates. Reuse preserves these author-only values.
    """

    acquire_sequence: int
    round_index: int
    paths: PathDiagnostics
    rejected_rows: int = 0
    projection_outcomes: tuple[int, ...] = ()
    projection_pruned_weight: tuple[float, ...] = ()


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

    Construct through FTFSource.open_acquire. sample_round takes a detached live
    snapshot and a zero-based software round index. Equal consecutive snapshots
    reuse density and conditional responses, but draw a fresh round stream.
    Opening the next acquire invalidates this session and releases its references;
    source-owned bounded noiseless records can survive. Using an invalidated
    session raises RuntimeError. Calls are synchronous and
    must be serialized by the caller (the binding provides acquire admission).
    """

    def __init__(self, source: "FTFSource", request: AcquireRequest) -> None:
        self._source = source
        self._active = True
        self.request = request
        self._snapshot: EnvironmentSnapshot | None = None
        self._prepared: _Prepared | None = None
        self.routing: list[PathDiagnostics] = []

    def invalidate(self) -> None:
        """Release deterministic arrays and reject further rounds; safe to repeat."""
        self._active = False
        self._prepared = None
        self.routing.clear()

    def sample_round(
        self,
        snapshot: EnvironmentSnapshot,
        index: int,
        *,
        cancel: CancellationQuery | None = None,
    ) -> IntegratedRound | RoundSignals:
        """Return integrated totals or decimated shots for this snapshot/index.

        Numeric/ADC errors propagate without partial results or changing the
        cached snapshot. Replay writes, when enabled, finish before returning.
        Decimated shots are rearranged from point-major C6 order to QICK loop
        order (including the reps axis). Neither mode averages or rounds data.
        cancel is checked between evolution chunks; None disables cancellation.
        Cancelled propagates without retaining a partial preparation.
        """
        owner = self._source
        if not self._active:
            raise RuntimeError("FTF acquire session is no longer active")
        if self._prepared is None or snapshot != self._snapshot:
            try:
                prepared = self._reuse_or_prepare(snapshot, index, cancel=cancel)
            except Exception:
                # Do not retain partial products from a failed preparation.
                owner.preparation_cache.clear()
                self._prepared = None
                self._snapshot = None
                raise
            self._prepared, self._snapshot = prepared, snapshot
        elif owner.telemetry is not None:
            self.routing.append(self._prepared.paths.for_reuse(telemetry=True))
        prepared = self._prepared
        acquire = self.request
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
                    ChannelSignals(channel.channel, ordered.reshape(channel.iq.shape))
                )
            result = RoundSignals(tuple(channels))
        if owner.private_replay is not None:
            save_replay(
                owner.private_replay,
                ReplayAcquire(
                    settings=owner.effective_settings,
                    request=acquire,
                    versions=replay_versions(),
                ),
                replace(snapshot, model_identity=owner.identity),
                index,
            )
        return result

    def _reuse_or_prepare(
        self,
        snapshot: EnvironmentSnapshot,
        index: int,
        *,
        cancel: CancellationQuery | None = None,
    ) -> _Prepared:
        owner = self._source
        # This check must precede every lookup, including equal-content requests.
        facts = owner.model_facts(snapshot)
        key = preparation_key(
            self.request, owner.readout, encoder=owner.envelope_encoder
        )
        prepared = owner.preparation_cache.get(key)
        if prepared is None or not all(
            owner.engine.reuse(facts.source, reuse_key)
            for reuse_key in prepared.reuse_keys
        ):
            # Evolution replays inventory and reprices immutable policy facts.
            # Only a changed selection or evicted facts requires a fresh solve.
            prepared = self._prepare(
                snapshot, index, key=key, previous=prepared, cancel=cancel
            )
            owner.preparation_cache.put(key, prepared)
        else:
            if owner.telemetry is not None:
                self.routing.append(prepared.paths.for_reuse(telemetry=True))
            owner.evaluations = (
                FTFNumericUse(
                    self.request.sequence,
                    index,
                    prepared.paths,
                    projection_outcomes=prepared.projection_outcomes,
                    projection_pruned_weight=prepared.projection_pruned_weight,
                ),
            )
            prepared.paths.warn_exceeded()
        return prepared

    def _prepare(
        self,
        snapshot: EnvironmentSnapshot,
        index: int,
        *,
        key: bytes,
        previous: _Prepared | None = None,
        cancel: CancellationQuery | None = None,
    ) -> _Prepared:
        acquire = self.request
        owner = self._source
        facts = owner.model_facts(snapshot)
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
        projection = None
        options = CavityEvolutionOptions(key, owner.telemetry is not None, cancel)
        if any(any(plan.projective_reads) for plan in plans):
            projection = evolve_projective_plans(
                owner.engine, plans, facts, owner.effective_settings, options=options
            )
            evolved, reuse_keys = projection.result, projection.reuse_keys
        else:
            evolved, reuse_keys = evolve_plans(
                owner.engine, plans, facts, owner.effective_settings, options=options
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
        owner.evaluations = (
            FTFNumericUse(
                acquire.sequence,
                index,
                evolved.paths,
                projection_outcomes=outcome_counts,
                projection_pruned_weight=pruned_weight,
            ),
        )
        try:
            distribution = _readout_distribution(
                evolved,
                plans,
                acquire.point_dims,
                None if projection is None else projection.readout,
            )
        except PopulationValidationError as error:
            owner.evaluations = (
                FTFNumericUse(
                    acquire.sequence, index, evolved.paths, error.rejected_rows
                ),
            )
            raise
        if previous is None:
            channels, raw_batches = _prepare_channels(plans, acquire, owner.settings)
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
    workers=1 and a 4 MiB routing-facts allowance. acquisition.private_replay
    selects an optional absolute author-only input archive directory; None performs
    no replay I/O. readout optionally supplies detached author capabilities,
    otherwise adapts settings.
    Replay rejects overrides that cannot be reconstructed from those settings.

    open_acquire drops the previous session's preparation references. execute
    opens a session only when its request object changes, preserving preparation
    across equal-snapshot rounds. The engine and routing facts survive acquires.
    preparation_cache owns at most 4 MiB of read-only probabilities, conditionals,
    content mappings, path diagnostics and deterministic integrated samplers.
    Exact current C1 content, layout, mode, reps and readout values select entries.
    History-sensitive hits ask Engine to reprice cached immutable routing facts.
    Only a changed selection or evicted facts evolves and refreshes populations;
    stable hits retain populations, channels, samplers and the preparation entry.
    Physical/numeric generation and engine replacement clear them before lookup.
    Its public max_bytes may be adjusted between calls; zero disables residency.
    Cumulative builds/hits and resident_bytes report reuse, never noisy results
    or request sampling identities. A hit updates this acquire's numeric use and
    reemits any report-only accuracy warning. Failed preparation clears the
    complete preparation cache. Active sessions/caller references are outside budgets.
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
    telemetry optionally supplies an already configured local routing writer.
    Enabled sources collect calling-thread routing CPU and expose finish_acquire
    for the binding completion callback. None (default) creates no directory,
    reads no telemetry CPU clocks, and retains no routing reports.
    acquisition supplies FTFAcquisitionOptions for the environment clock and
    private truth/replay paths; None uses its defaults. acquisition.clock=None
    creates a clock. begin_acquire freezes losses once per acquisition.
    acquisition.truth_log optionally selects
    the private JSONL path; None uses ~/.local/state/zcu-tools/simulation/ftf/
    <session>/truth.jsonl. Enabled drift writes one row per loop per begin and log
    I/O errors propagate. Disabled drift creates no file or drift RNG streams.
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
    ) -> None:
        options = FTFAcquisitionOptions() if acquisition is None else acquisition
        private_replay = options.private_replay
        if private_replay is not None and not private_replay.is_absolute():
            raise ValueError("private replay requires an explicit absolute root")
        self.readout = readout_configuration(settings) if readout is None else readout
        if private_replay is not None and self.readout != readout_configuration(
            settings
        ):
            raise ValueError(
                "private replay requires readout settings to match author configuration"
            )
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
        self.private_replay = private_replay
        self.telemetry = telemetry
        self.engine = Engine(1) if engine is None else engine
        self.evaluations: tuple[FTFNumericUse, ...] = ()
        self._session: FTFAcquire | None = None
        self._facts: FTFModelFacts | None = None
        self.preparation_cache: PreparationCache[_Prepared] = PreparationCache()
        self.envelope_encoder = EnvelopeEncoder()
        self._preparation_engine = self.engine
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
        facts plus complete preparations. Disabled drift draws no RNG and
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
                self._facts.source.close()
            self._facts = None
            self.preparation_cache.clear()
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
            self.preparation_cache.clear()
            self._preparation_engine = self.engine
        if self._facts is None or not self._facts.matches(
            self.effective_settings, snapshot
        ):
            self.preparation_cache.clear()
            if self._facts is not None:
                self._facts.source.close()
            self._facts = None
            self._facts = FTFModelFacts(self.effective_settings, snapshot)
        return self._facts

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
        return EnvironmentSnapshot("ftf-v2", self.identity, tuple(values))

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
        self._session = FTFAcquire(self, request)
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

    def finish_acquire(self, rounds: tuple[RoundRequest, ...], wall_s: float) -> None:
        """Write one best-effort summary at the binding's completion boundary.

        rounds includes failed captured attempts, and may be empty if request
        construction failed. wall_s is admitted acquire wall time including
        pacing. Only the matching current session contributes routing work.
        Direct execute users may call this explicitly after their final round.
        """
        if self.telemetry is None:
            return
        request = rounds[0].acquire if rounds else None
        session = self._session
        paths = (
            session.routing
            if session is not None and session.request is request
            else ()
        )
        self.telemetry.write(
            points=prod(request.point_dims) if request is not None else 0,
            rounds=len(rounds),
            wall_s=wall_s,
            paths=paths,
            projection_outcomes=self.evaluations[-1].projection_outcomes
            if paths and self.evaluations
            else (),
            projection_pruned_weight=self.evaluations[-1].projection_pruned_weight
            if paths and self.evaluations
            else (),
        )

    def close(self) -> None:
        """Release preparations, model/source usage and Engine; safe to repeat."""
        if self._session is not None:
            self._session.invalidate()
            self._session = None
        if self._facts is not None:
            self._facts.source.close()
        self._facts = None
        self.preparation_cache.clear()
        self.envelope_encoder.clear()
        self._closed = True
        self.engine.close()


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
    """Collect declared fast-flux generators for mock firmware capacity/routes."""
    return tuple(
        line.gen_ch
        for line in (
            settings.flux_lines.f1,
            settings.flux_lines.coupler,
            settings.flux_lines.f2,
        )
        if line is not None
    )


def ftf(
    configuration: FTFSettings | str | Path,
    *,
    soccfg: QickConfig,
    acquisition: FTFAcquisitionOptions | None = None,
    workers: int = 1,
    warmup: bool | EnvironmentWarmup = False,
    time_scale: float | None = None,
) -> SimulationEnvironment:
    """Build an FTF environment from physical settings and a caller-owned board.

    Startup creates fixed V/A FakeDevices and applies initial values once. The
    returned manager owns cleanup; builder failures close all devices and preserve
    both original and cleanup errors. soccfg is the required QICK compile profile,
    built by the caller from independent board YAML. It must include all configured
    drive/flux channels, pulse readout and optional mux/PFB routes. No channel
    counts or board settings are inferred from physical wiring. The same soccfg
    supplies the SoC and numeric PFB carrier selection. One environment supports
    separate pulse and mux acquires without reconfiguration. Host receives no model truth.
    acquisition.private_replay explicitly enables author-only checked input archives at an
    absolute directory. Replay recomputes with checked numeric and RNG versions;
    no U table or exact density record is consulted. One engine persists across
    acquires, using workers (default 1) as its sole parallel width.
    Worker threads live only during multi-timeline evolve; dropping the environment
    releases its in-memory routing facts. FTFSource.close explicitly clears it.
    warmup=False keeps direct callers/tests free of background CPU work. True
    starts process-wide single-flight numeric preparation. The Notebook example
    opts in and reserves two host CPUs; GUI startup uses one worker without
    warmup. An explicit EnvironmentWarmup supplies named preparation items
    instead. It must not
    sample or mutate live devices. env.wait_ready() waits before caller timing;
    acquire also waits automatically. Failure status permits on-demand compiling.
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
            engine=Engine(workers),
            acquisition=replace(options, clock=sim_clock),
            telemetry=telemetry,
        )
        readout = coordinator.readout
        mux = readout.mux
        gens = (
            settings.readout.drive.gen_ch,
            *(s.gen_ch for s in settings.drive_lines.values()),
            *_flux_generator_channels(settings),
            *((mux.gen_ch,) if mux is not None else ()),
        )
        mux_receivers = () if mux is None else tuple(r.ro_ch for r in mux.receivers)
        receivers = (readout.pulse.ro_ch, *mux_receivers)
        from zcu_tools.simulators.ftf.environment.warmup import startup_warmup

        preparation = (
            startup_warmup()
            if warmup is True
            else (warmup if isinstance(warmup, EnvironmentWarmup) else None)
        )
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
                if coordinator.telemetry is not None
                else None,
            ),
        )
        soc = SimulationSoc(
            soccfg,
            binding,
            routes=tuple((gen, ch) for gen in gens for ch in receivers),
        )
        environment = SimulationEnvironment(
            soc,
            soccfg,
            devices,
            "FTF: independent shots, terminal shared-line readout, synchronous acquire/decimated",
            preparation,
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
