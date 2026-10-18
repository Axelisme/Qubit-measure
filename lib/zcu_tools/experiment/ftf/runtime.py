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
from zcu_tools.experiment.simulation import SimulationEnvironment
from zcu_tools.program.v2.mocksoc import make_mock_soccfg, make_mux_mock_soccfg
from zcu_tools.program.v2.simulation import SimulationBinding, SimulationSoc
from zcu_tools.resources.ftf import FTFSettings, load_ftf, parse_ftf
from zcu_tools.resources.simulation_replay import ReplayAcquire, save_replay
from zcu_tools.simulate.adc import (
    ChannelReadout,
    ConditionalReadout,
    ConditionalSums,
    PopulationValidationError,
    normalize_populations,
    round_statistics,
    sample_traces,
)
from zcu_tools.simulate.evolution import (
    Basis,
    Engine,
    PathDiagnostics,
    batch_populations,
)

from .readout import FTFReadoutConfig, readout_configuration
from .replay import BACKEND_VERSION, replay_versions
from .template import plan_template


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


class FTFAcquire:
    """One fixed request with at most the immediately preceding round's preparation.

    Construct through FTFSource.open_acquire. sample_round takes a detached live
    snapshot and a zero-based software round index. Equal consecutive snapshots
    reuse density and conditional responses, but draw a fresh round stream.
    Opening the next acquire invalidates this session and releases preparations;
    using an invalidated session raises RuntimeError. Calls are synchronous and
    must be serialized by the caller (the binding provides acquire admission).
    """

    def __init__(self, source: "FTFSource", request: AcquireRequest) -> None:
        self._source = source
        self._active = True
        self.request = request
        self._snapshot: EnvironmentSnapshot | None = None
        self._prepared: _Prepared | None = None

    def invalidate(self) -> None:
        """Release deterministic arrays and reject further rounds; safe to repeat."""
        self._active = False
        self._prepared = None

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
            prepared = self._prepare(snapshot, index)
            self._prepared, self._snapshot = prepared, snapshot
        prepared = self._prepared
        acquire = self.request
        arguments = dict(
            reps=acquire.reps,
            seed=acquire.root_seed,
            acquire_sequence=acquire.sequence,
            round_index=index,
        )
        if acquire.mode == "integrated":
            result = round_statistics(
                prepared.populations, prepared.channels, **arguments
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

    def _prepare(self, snapshot: EnvironmentSnapshot, index: int) -> _Prepared:
        acquire = self.request
        owner = self._source
        plans = tuple(
            plan_template(owner.settings, snapshot, template, readout=owner.readout)
            for template in acquire.templates
        )
        evolved = owner.engine.evolve(
            plans[0].source, tuple(plan.batch for plan in plans), cancel=Event()
        )
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
        channels = []
        for channel_index, first in enumerate(plans[0].readouts):
            response = ConditionalReadout(
                first.batch.model,
                rtol=owner.settings.numerics.solver.rtol,
                atol=owner.settings.numerics.solver.atol,
            )
            conditionals = []
            traces = []
            content_index = np.empty(prod(acquire.point_dims), dtype=np.int64)
            offset = 0
            for plan in plans:
                readout = plan.readouts[channel_index].batch
                conditional = response.responses(
                    replace(readout, include_traces=acquire.mode == "decimated")
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
        return _Prepared(
            probabilities.reshape((*acquire.point_dims, state_count)), tuple(channels)
        )


class FTFSource:
    """Environment coordinator with one persistent Engine and one current acquire.

    settings is a validated immutable FTF resource; devices supplies live DC
    readbacks to snapshot. engine optionally supplies a configured numerical
    executor; this source takes ownership, including close(). Without it, use
    workers=1 and a 3 GiB prepare cache. private_replay is an optional absolute
    author-only input archive directory; None performs no replay I/O. readout
    optionally supplies detached author capabilities, otherwise adapts settings.
    Replay rejects overrides that cannot be reconstructed from those settings.

    open_acquire drops the previous session's density/conditional arrays. execute
    opens a session only when its request object changes, preserving preparation
    across equal-snapshot rounds. The engine/cache survives across acquires.
    close invalidates the current session and closes the engine idempotently.
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
        self.identity = sha256(settings.model_dump_json().encode()).hexdigest()
        self.private_replay = private_replay
        self.engine = Engine(1, 3 * 1024**3) if engine is None else engine
        self.evaluations: tuple[FTFNumericUse, ...] = ()
        self._session: FTFAcquire | None = None
        self._closed = False

    def snapshot(self) -> EnvironmentSnapshot:
        """Read fixed-unit live DC values once, rejecting missing/invalid devices."""
        values = []
        for name, source in self.settings.dc_bias.sources.items():
            info = self.devices.get_info(name)
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
        """Execute a detached round, reusing only this acquire's last equal snapshot."""
        if self._session is None or self._session.request is not request.acquire:
            self.open_acquire(request.acquire)
        session = self._session
        if session is None:
            raise RuntimeError("FTF acquire was not opened")
        return session.sample_round(request.snapshot, request.index)

    def close(self) -> None:
        """Release current deterministic preparations and the engine; safe to repeat."""
        if self._session is not None:
            self._session.invalidate()
            self._session = None
        self._closed = True
        self.engine.close()


def ftf(
    configuration: FTFSettings | str | Path,
    *,
    private_replay: Path | None = None,
    workers: int = 1,
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
    """
    settings = (
        parse_ftf(configuration)
        if isinstance(configuration, FTFSettings)
        else load_ftf(configuration)
    )
    devices = DeviceManager()
    try:
        for name, source in settings.dc_bias.sources.items():
            device = FakeDevice(fast_mode=True, value_unit=source.unit)
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
        coordinator = FTFSource(
            settings,
            devices,
            engine=Engine(workers, 3 * 1024**3),
            private_replay=private_replay,
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
        binding = SimulationBinding(
            coordinator.snapshot,
            coordinator.execute,
            backend_version=BACKEND_VERSION,
            seed=settings.numerics.seed,
        )
        soc = SimulationSoc(
            soccfg,
            binding,
            routes=tuple((gen, ch) for gen in gens for ch in receivers),
        )
        return SimulationEnvironment(
            soc,
            soccfg,
            devices,
            "FTF: independent shots, terminal shared-line readout, synchronous acquire/decimated",
        )
    except Exception as error:
        try:
            devices.close_all_devices()
        except ExceptionGroup as cleanup:
            raise ExceptionGroup(
                "FTF startup and cleanup failed", [error, cleanup]
            ) from error
        raise
