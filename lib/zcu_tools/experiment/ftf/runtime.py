"""FTF round snapshots, persistent engine ownership and compact readout sampling."""

from dataclasses import dataclass, replace
from hashlib import sha256
from math import prod
from pathlib import Path
from threading import Event

import numpy as np
from numpy.typing import NDArray

from zcu_tools.core.simulation import (
    AcquireRequest,
    ChannelSignals,
    EnvironmentSnapshot,
    RoundRequest,
    RoundSignals,
    RoundStatistics,
)
from zcu_tools.device import DeviceManager
from zcu_tools.device.fake import FakeDevice, FakeDeviceInfo
from zcu_tools.experiment.simulation import EnvironmentWarmup, SimulationEnvironment
from zcu_tools.program.v2.mocksoc import make_mock_soccfg, make_mux_mock_soccfg
from zcu_tools.program.v2.simulation import (
    SimulationBinding,
    SimulationCallbacks,
    SimulationSoc,
)
from zcu_tools.resources.ftf import FTFSettings, load_ftf, parse_ftf
from zcu_tools.resources.simulation_replay import ReplayAcquire, save_replay
from zcu_tools.simulate.adc import (
    ChannelReadout,
    ConditionalSums,
    PopulationValidationError,
    PreparedStatistics,
    ReadoutCache,
    normalize_populations,
    sample_traces,
)
from zcu_tools.simulate.evolution import (
    Basis,
    Engine,
    PathDiagnostics,
    batch_populations,
)

from .preparation import EnvelopeEncoder, PreparationCache, preparation_key
from .readout import FTFReadoutConfig, readout_configuration
from .replay import BACKEND_VERSION, replay_versions
from .telemetry import RoutingTelemetry
from .template import FTFModelFacts, plan_template


@dataclass(frozen=True)
class FTFNumericUse:
    """Author-only latest solve diagnostics, never ordinary acquisition truth.

    acquire_sequence/round_index identify the solve's sampling scope. paths counts
    actual engine segment paths; it contains no densities or noise realizations.
    rejected_rows is zero after successful ADC preparation, or the number of
    rows ADC rejected under its reported validation rule. The record survives
    that failure. A reused snapshot does not create a new solve or change it.
    """

    acquire_sequence: int
    round_index: int
    paths: PathDiagnostics
    rejected_rows: int = 0


@dataclass(frozen=True)
class _Prepared:
    populations: NDArray[np.float64]
    channels: tuple[ChannelReadout, ...]
    paths: PathDiagnostics
    sampler: PreparedStatistics | None


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
        self, snapshot: EnvironmentSnapshot, index: int
    ) -> RoundStatistics | RoundSignals:
        """Return integrated totals or decimated shots for this snapshot/index.

        Numeric/ADC errors propagate without partial results or changing the
        cached snapshot. Replay writes, when enabled, finish before returning.
        Decimated shots are rearranged from point-major C6 order to QICK loop
        order (including the reps axis). Neither mode averages or rounds data.
        """
        owner = self._source
        if not self._active:
            raise RuntimeError("FTF acquire session is no longer active")
        if self._prepared is None or snapshot != self._snapshot:
            try:
                prepared = self._reuse_or_prepare(snapshot, index)
            except Exception:
                # Preparation may have filled only part of either deterministic
                # cache. Do not keep any generation's partial products.
                owner.preparation_cache.clear()
                owner.readout_cache.clear()
                self._prepared = None
                self._snapshot = None
                raise
            self._prepared, self._snapshot = prepared, snapshot
        elif owner.telemetry is not None:
            self.routing.append(self._prepared.paths.for_reuse(telemetry=True))
        prepared = self._prepared
        acquire = self.request
        arguments = dict(
            reps=acquire.reps,
            seed=acquire.root_seed,
            acquire_sequence=acquire.sequence,
            round_index=index,
        )
        if acquire.mode == "integrated":
            if prepared.sampler is None:
                raise RuntimeError("integrated preparation requires a sampler")
            result = prepared.sampler.draw(
                seed=acquire.root_seed,
                acquire_sequence=acquire.sequence,
                round_index=index,
            )
        else:
            traces = sample_traces(prepared.populations, prepared.channels, **arguments)
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
                    settings=owner.settings, request=acquire, versions=replay_versions()
                ),
                snapshot,
                index,
            )
        return result

    def _reuse_or_prepare(self, snapshot: EnvironmentSnapshot, index: int) -> _Prepared:
        owner = self._source
        # This check must precede every lookup, including equal-content requests.
        facts = owner.model_facts(snapshot)
        key = preparation_key(
            self.request, owner.readout, encoder=owner.envelope_encoder
        )
        prepared = owner.preparation_cache.get(key)
        if prepared is None or not owner.engine.reuse(facts.source, key):
            # Evolution replays inventory and reprices immutable policy facts.
            # Only a changed selection or evicted facts requires a fresh solve.
            prepared = self._prepare(snapshot, index, key=key, previous=prepared)
            owner.preparation_cache.put(key, prepared)
        else:
            if owner.telemetry is not None:
                self.routing.append(prepared.paths.for_reuse(telemetry=True))
            owner.evaluations = (
                FTFNumericUse(self.request.sequence, index, prepared.paths),
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
    ) -> _Prepared:
        acquire = self.request
        owner = self._source
        facts = owner.model_facts(snapshot)
        plans = tuple(
            plan_template(
                owner.settings, snapshot, template, readout=owner.readout, facts=facts
            )
            for template in acquire.templates
        )
        evolved = owner.engine.evolve(
            plans[0].source,
            tuple(plan.batch for plan in plans),
            cancel=Event(),
            telemetry=owner.telemetry is not None,
            reuse_key=key,
        )
        if owner.telemetry is not None:
            self.routing.append(evolved.paths)
        # Retain solve diagnostics even when ADC preparation rejects a row.
        owner.evaluations = (FTFNumericUse(acquire.sequence, index, evolved.paths),)
        state_count = evolved.densities.shape[-1]
        probabilities = np.empty((prod(acquire.point_dims), state_count))
        dc_model = plans[0].source.model((0,) * plans[0].batch.flux.level.shape[1])
        # Evolution returns DC energy coordinates; readout uses that same basis
        # with the ADC model's explicit state order.
        readout_basis = Basis(
            dc_model.identity,
            plans[0].readouts[0].batch.model.state_labels,
            dc_model.eigenvectors,
        )
        populations = batch_populations(evolved.densities, readout_basis, readout_basis)
        try:
            probabilities[evolved.points] = normalize_populations(populations)
        except PopulationValidationError as error:
            owner.evaluations = (
                FTFNumericUse(
                    acquire.sequence, index, evolved.paths, error.rejected_rows
                ),
            )
            raise
        channels = [] if previous is None else list(previous.channels)
        if previous is None:
            for channel_index, first in enumerate(plans[0].readouts):
                conditionals = []
                traces = []
                content_index = np.empty(prod(acquire.point_dims), dtype=np.int64)
                offset = 0
                for plan in plans:
                    readout = plan.readouts[channel_index].batch
                    conditional = owner.readout_cache.responses(
                        replace(readout, include_traces=acquire.mode == "decimated"),
                        rtol=owner.settings.numerics.solver.rtol,
                        atol=owner.settings.numerics.solver.atol,
                    )
                    conditionals.append(conditional)
                    if conditional.traces is not None:
                        traces.append(conditional.traces)
                    content_index[plan.batch.points] = offset + readout.content_index
                    offset += len(conditional.sums)
                conditional = ConditionalSums(
                    np.concatenate([item.sums for item in conditionals]),
                    conditionals[0].sample_counts,
                    np.concatenate(traces) if traces else None,
                )
                channels.append(
                    ChannelReadout(
                        first.channel,
                        conditional,
                        content_index.reshape(acquire.point_dims),
                        first.batch.model.receiver,
                    )
                )
        probabilities = probabilities.reshape((*acquire.point_dims, state_count))
        probabilities.setflags(write=False)
        if previous is not None and np.array_equal(probabilities, previous.populations):
            sampler = previous.sampler
        else:
            sampler = (
                PreparedStatistics(probabilities, tuple(channels), reps=acquire.reps)
                if acquire.mode == "integrated"
                else None
            )
        return _Prepared(
            probabilities,
            tuple(channels),
            evolved.paths.for_reuse(telemetry=False),
            sampler,
        )


class FTFSource:
    """Environment coordinator with one persistent Engine and one current acquire.

    settings is a validated immutable FTF resource; devices supplies startup DC
    identities whose live values are read by snapshot, even after session
    disconnects or registry renames. engine optionally supplies a configured numerical
    executor; this source takes ownership, including close(). Without it, use
    workers=1 and a 2 GiB prepare cache. private_replay is an optional absolute
    author-only input archive directory; None performs no replay I/O. readout
    optionally supplies detached author capabilities, otherwise adapts settings.
    Replay rejects overrides that cannot be reconstructed from those settings.

    open_acquire drops the previous session's preparation references. execute
    opens a session only when its request object changes, preserving preparation
    across equal-snapshot rounds. The engine/cache survives across acquires.
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
    reemits any report-only accuracy warning. Failed preparation clears both
    deterministic caches. Active sessions/caller references are outside budgets.
    At most one complete snapshot's physical model survives across acquires;
    full settings and numeric revisions also participate in its exact key.
    readout_cache owns at most 8 MiB of physical ADC solvers and exact conditional
    contents across acquires, never noisy IQ. Its cumulative builds/hits and
    resident_bytes are author diagnostics. Full model/receiver/tolerance content
    selects solvers; physical-facts generation changes clear all readout residents.
    close releases physical facts/readout residents, invalidates the session and
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
    Calls are synchronous, serialized by SimulationBinding, not thread-safe.
    """

    def __init__(
        self,
        settings: FTFSettings,
        devices: DeviceManager,
        *,
        engine: Engine | None = None,
        private_replay: Path | None = None,
        readout: FTFReadoutConfig | None = None,
        telemetry: RoutingTelemetry | None = None,
    ) -> None:
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
        self.devices = devices
        # The physical source survives session disconnects and registry renames.
        self._source_devices = devices.get_all_devices()
        self.identity = sha256(settings.model_dump_json().encode()).hexdigest()
        self.private_replay = private_replay
        self.telemetry = telemetry
        self.engine = Engine(1, 2 * 1024**3) if engine is None else engine
        self.evaluations: tuple[FTFNumericUse, ...] = ()
        self._session: FTFAcquire | None = None
        self._facts: FTFModelFacts | None = None
        self.readout_cache = ReadoutCache(8 * 1024**2)
        self.preparation_cache: PreparationCache[_Prepared] = PreparationCache()
        self.envelope_encoder = EnvelopeEncoder()
        self._preparation_engine = self.engine
        self._closed = False

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
        if self._facts is None or not self._facts.matches(self.settings, snapshot):
            self.readout_cache.clear()
            self.preparation_cache.clear()
            if self._facts is not None:
                self._facts.source.close()
            self._facts = None
            self._facts = FTFModelFacts(self.settings, snapshot)
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
        """
        if self._closed:
            raise RuntimeError("FTF source is closed")
        if not request.templates:
            raise ValueError("FTF requires timeline templates")
        if request.rng_version != f"PCG64/SeedSequence-v2/numpy-{np.__version__}":
            raise ValueError("FTF RNG version mismatch")
        if self._session is not None:
            self._session.invalidate()
        self._session = FTFAcquire(self, request)
        self.evaluations = ()
        return self._session

    def execute(self, request: RoundRequest) -> RoundStatistics | RoundSignals:
        """Execute a detached round with bounded exact noiseless reuse and fresh draws."""
        if self._session is None or self._session.request is not request.acquire:
            self.open_acquire(request.acquire)
        session = self._session
        if session is None:
            raise RuntimeError("FTF acquire was not opened")
        return session.sample_round(request.snapshot, request.index)

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
        )

    def close(self) -> None:
        """Release preparations, model/source usage and Engine; safe to repeat."""
        if self._session is not None:
            self._session.invalidate()
            self._session = None
        if self._facts is not None:
            self._facts.source.close()
        self._facts = None
        self.readout_cache.clear()
        self.preparation_cache.clear()
        self.envelope_encoder.clear()
        self._closed = True
        self.engine.close()


def ftf(
    configuration: FTFSettings | str | Path,
    *,
    private_replay: Path | None = None,
    workers: int = 1,
    warmup: bool | EnvironmentWarmup = False,
    min_round_duration_s: float | None = None,
) -> SimulationEnvironment:
    """Build a fixed-profile FTF environment from named values or a YAML path.

    Startup creates fixed V/A FakeDevices and applies initial values once. The
    returned manager owns cleanup; builder failures close all devices and preserve
    both original and cleanup errors. Channels use make_mock_soccfg's supported
    profile, never arbitrary firmware overrides. Host receives no model truth.
    private_replay explicitly enables author-only checked input archives at an
    absolute directory. Replay recomputes with checked numeric and RNG versions;
    no U table or exact density record is consulted. One engine persists across
    acquires, using workers (default 1) as its sole parallel width.
    Worker threads live only during multi-timeline evolve; dropping the environment
    releases its in-memory engine cache. FTFSource.close explicitly clears it.
    warmup=False keeps direct callers/tests free of background CPU work. True
    starts process-wide single-flight numeric preparation. The Notebook example
    opts in and reserves two host CPUs; GUI startup uses one worker without
    warmup. An explicit EnvironmentWarmup supplies named preparation items
    instead. It must not
    sample or mutate live devices. env.wait_ready() waits before caller timing;
    acquire also waits automatically. Failure status permits on-demand compiling.
    min_round_duration_s overrides numerics.min_round_duration_s (default 0.2 s)
    for successful binding rounds, including snapshot/execution time. None uses
    settings; finite nonnegative values override them. Zero disables pacing for
    profiling/benchmarks without changing numeric settings or sampling. Slow rounds
    and failed calls do not wait. Invalid durations raise ValueError.
    numerics.telemetry.enabled defaults to True. Use
    settings.with_telemetry(enabled=False) for profiles/benchmarks without routing
    timers or I/O. numerics.telemetry.directory defaults to None, selecting
    <project root>/.sim_telemetry. Relative paths resolve at startup.
    Each admitted acquire writes one bounded JSONL summary;
    persistence errors warn once without changing acquisition or RNG results.
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
        telemetry = None
        if settings.numerics.telemetry.enabled:
            directory = settings.numerics.telemetry.directory
            if directory is None:
                directory = Path(__file__).resolve().parents[4] / ".sim_telemetry"
            telemetry = RoutingTelemetry(directory.absolute())
        coordinator = FTFSource(
            settings,
            devices,
            engine=Engine(workers, 2 * 1024**3),
            private_replay=private_replay,
            telemetry=telemetry,
        )
        gens = (
            settings.readout.drive.gen_ch,
            *(s.gen_ch for s in settings.drive_lines.values()),
        )
        readout = coordinator.readout
        receivers = tuple(r.ro_ch for r in readout.receivers)
        if readout.kind == "mux":
            soccfg = make_mux_mock_soccfg(
                max(gens) + 1,
                max(receivers) + 1,
                gen_ch=settings.readout.drive.gen_ch,
                ro_chs=receivers,
            )
        else:
            soccfg = make_mock_soccfg(max(gens) + 1, max(receivers) + 1)
        from .warmup import startup_warmup

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
            min_round_duration_s=(
                settings.numerics.min_round_duration_s
                if min_round_duration_s is None
                else min_round_duration_s
            ),
            callbacks=SimulationCallbacks(
                wait_ready=None if preparation is None else preparation.wait_ready,
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
