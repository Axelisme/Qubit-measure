"""FTF environment ownership, live round snapshots and numeric-tool composition."""

from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path

import numpy as np

from zcu_tools.core.simulation import (
    ChannelSignals,
    EnvironmentSnapshot,
    PointTimeline,
    RoundRequest,
    RoundSignals,
)
from zcu_tools.device import DeviceManager
from zcu_tools.device.fake import FakeDevice, FakeDeviceInfo
from zcu_tools.experiment.simulation import SimulationEnvironment
from zcu_tools.program.v2.mocksoc import make_mock_soccfg
from zcu_tools.program.v2.simulation import SimulationBinding, SimulationSoc
from zcu_tools.resources.ftf import FTFSettings, load_ftf, parse_ftf
from zcu_tools.resources.simulation_replay import ReplayAcquire, save_replay
from zcu_tools.simulate.adc import ADC, PopulationInput, SamplingIdentity
from zcu_tools.simulate.evolution import (
    EvaluationReceipt,
    Evolution,
    EvolutionLibrary,
    LibrarySnapshot,
    Populations,
    SolverSettings,
    populations,
)

from .point import FTFPointPlan, plan_point
from .replay import replay_versions


@dataclass(frozen=True)
class FTFNumericUse:
    """Author-only actual evolution content used by one round execution group.

    acquire_sequence/round_index: detached request sampling scope, not cache key.
    group_index: zero-based deterministic timeline group in that round. receipt:
    origin, immutable numeric content identities and full evolve cost. These
    records contain no density/noise state and are not ordinary GUI/MCP truth.
    """

    acquire_sequence: int
    round_index: int
    group_index: int
    receipt: EvaluationReceipt


@dataclass(frozen=True)
class _Prepared:
    plan: FTFPointPlan
    populations: Populations
    adc: ADC
    receipt: EvaluationReceipt


class FTFSource:
    """Environment coordinator with immutable settings and one driver manager.

    settings: validated FTF parameters retained for private in-memory reproduction;
    devices: the exact manager shared by the host. snapshot reads live DC once
    per round; execute uses only RoundRequest numeric inputs, not drivers.
    library: optional explicitly pathed numeric owner for frozen U tables/exact
    records. Each round fixes published versions. No point preparation, density,
    source tail or receiver noise is retained across calls. evaluations holds
    actual numeric receipts for every group/round of the latest acquire only;
    a new acquire sequence clears earlier receipts, never published disk data.
    private_replay: optional absolute author-only archive directory. Successful
    rounds save detached controls/settings/snapshots/seeds and checked numeric
    references there. None disables all private input I/O. Archive errors fail the
    acquire; public provenance remains only simulation_id. This is author-side
    composition, not an ordinary GUI/MCP truth interface.
    """

    def __init__(
        self,
        settings: FTFSettings,
        devices: DeviceManager,
        *,
        library: EvolutionLibrary | None = None,
        private_replay: Path | None = None,
    ) -> None:
        self.settings = settings
        self.devices = devices
        self.identity = sha256(settings.model_dump_json().encode()).hexdigest()
        self.library = library
        if private_replay is not None and not private_replay.is_absolute():
            raise ValueError("private replay requires an explicit absolute root")
        self.private_replay = private_replay
        self.evaluations: tuple[FTFNumericUse, ...] = ()

    def snapshot(self) -> EnvironmentSnapshot:
        """Capture current fixed-unit live values, without startup writes.

        Missing/disconnected devices, mismatched units, nonfinite values and
        source-limit violations fail before acquisition; no cached readback.
        """
        values = []
        for source in self.settings.devices:
            info = self.devices.get_info(source.name)
            if not isinstance(info, FakeDeviceInfo) or info.value_unit != source.unit:
                raise ValueError(
                    f"devices.{source.name}.unit: fixed source unit mismatch"
                )
            if (
                not np.isfinite(info.value)
                or not source.limits[0] <= info.value <= source.limits[1]
            ):
                raise ValueError(f"devices.{source.name}.value: outside source limits")
            values.append((source.name, info.value))
        return EnvironmentSnapshot("ftf-v1", self.identity, tuple(values))

    def _prepare(
        self,
        snapshot: EnvironmentSnapshot,
        timeline: PointTimeline,
        tables: LibrarySnapshot | None,
    ) -> _Prepared:
        plan = plan_point(self.settings, snapshot, timeline)
        solver = self.settings.solver
        # Explicit three-tool composition: density -> target populations -> ADC.
        evolution = Evolution(
            plan.evolution_model,
            SolverSettings(solver.rtol, solver.atol, solver.max_step_ns),
            library=self.library,
            tables=tables,
        )
        state = evolution.evolve(
            plan.joint.initial_state(), plan.controls, end_ns=plan.readout_start_ns
        )
        mixture = populations(state, plan.readout_basis)
        adc = ADC(plan.readout_model, rtol=solver.rtol, atol=solver.atol)
        return _Prepared(plan, mixture, adc, evolution.last_evaluation)

    def execute(
        self, request: RoundRequest, *, tables: LibrarySnapshot | None = None
    ) -> RoundSignals:
        """Generate this round's single ADC sums/traces from detached timelines.

        Equal physical executions share evolution and conditional response, but
        receive independent joint-initial/noise draws. Point grouping does not
        depend on sweep names. No state/filter tail crosses shots/rounds. Raw and
        averaged acquisition share this one numeric realization via the adapter.
        tables optionally supplies the exact frozen numeric publication set for
        replay; None captures current publications once at this round boundary.
        private_replay, if enabled, persists the successful round before return;
        I/O and numeric-reference errors propagate rather than falling back.
        """
        # Publication versions are fixed for this round, not for each group.
        if tables is None and self.library is not None:
            tables = self.library.snapshot()
        acquire = request.acquire
        if (
            self.evaluations
            and self.evaluations[-1].acquire_sequence != acquire.sequence
        ):
            self.evaluations = ()
        if len(acquire.timelines) != acquire.executions:
            raise ValueError("FTF requires per-execution point timelines")
        groups: dict[PointTimeline, list[int]] = {}
        for i, timeline in enumerate(acquire.timelines):
            key = replace(
                timeline,
                coordinates=(),
                events=tuple(replace(e, iterations=()) for e in timeline.events),
            )
            groups.setdefault(key, []).append(i)
        window = acquire.readouts[0]
        shape = (
            (acquire.executions, window.reads, window.samples, 2)
            if acquire.mode == "decimated"
            else (acquire.executions, window.reads, 2)
        )
        output = np.empty(shape, dtype=float)
        for group, (timeline, indices) in enumerate(groups.items()):
            prepared = self._prepare(request.snapshot, timeline, tables)
            self.evaluations += (
                FTFNumericUse(acquire.sequence, request.index, group, prepared.receipt),
            )
            plan = prepared.plan
            mixture = PopulationInput(
                prepared.populations.basis.labels, prepared.populations.values
            )
            identity = SamplingIdentity(
                acquire.root_seed, (acquire.sequence, request.index, group)
            )
            windows = prepared.adc.sample_windows(
                mixture, plan.pulses, plan.grids, identity, shots=len(indices)
            )
            for read, samples in enumerate(windows):
                output[indices, read] = (
                    samples.traces if acquire.mode == "decimated" else samples.sums
                )
        if self.private_replay is not None:
            identities = tuple(
                identity
                for use in self.evaluations
                if use.round_index == request.index
                for identity in use.receipt.content_identities
            )
            manifest = (
                self.library.replay_manifest(tables, identities)
                if self.library is not None and tables is not None
                else None
            )
            save_replay(
                self.private_replay,
                ReplayAcquire(
                    settings=self.settings,
                    request=acquire,
                    versions=replay_versions(),
                    library_path=self.library.path
                    if self.library is not None
                    else None,
                ),
                request.snapshot,
                request.index,
                manifest,
            )
        return RoundSignals((ChannelSignals(self.settings.receiver.ro_ch, output),))


def ftf(
    configuration: FTFSettings | str | Path,
    *,
    library: EvolutionLibrary | None = None,
    private_replay: Path | None = None,
) -> SimulationEnvironment:
    """Build a fixed-profile FTF environment from named values or a YAML path.

    Startup creates fixed V/A FakeDevices and applies initial values once. The
    returned manager owns cleanup; builder failures close all devices and preserve
    both original and cleanup errors. Channels use make_mock_soccfg's supported
    profile, never arbitrary firmware overrides. Host receives no model truth.
    library optionally injects author-selected EvolutionLibrary storage; no path
    search/default cache. Numeric owner handles table parsing, misses and saving.
    private_replay explicitly enables author-only checked input archives at an
    absolute directory. None (default) writes no private replay data. Archives
    contain model truth and must not be published through normal Run metadata.
    Successful round capture failures propagate. replay_round reloads one round
    independently; numeric references remain in the injected library and must
    be retained. The module README documents weak dispersive/Markov/secular
    approximations and readout-origin flux freezing. No runtime applicability
    gate is added.
    """
    settings = (
        parse_ftf(configuration)
        if isinstance(configuration, FTFSettings)
        else load_ftf(configuration)
    )
    devices = DeviceManager()
    try:
        for source in settings.devices:
            device = FakeDevice(fast_mode=True, value_unit=source.unit)
            devices.register_device(source.name, device)
            device.setup(
                FakeDeviceInfo(
                    address=device.address,
                    value_unit=source.unit,
                    output="on",
                    value=source.initial,
                    rampstep=source.rampstep,
                ),
                progress=False,
            )
        coordinator = FTFSource(
            settings, devices, library=library, private_replay=private_replay
        )
        soccfg = make_mock_soccfg(
            settings.channels.generators, settings.channels.readouts
        )
        binding = SimulationBinding(
            coordinator.snapshot,
            coordinator.execute,
            backend_version="ftf-direct-v1",
            seed=settings.seed,
        )
        gens = (
            settings.readout_source.gen_ch,
            *(s.gen_ch for s in settings.flux_sources),
            *(s.gen_ch for s in settings.microwave_sources),
        )
        soc = SimulationSoc(
            soccfg,
            binding,
            routes=tuple((gen, settings.receiver.ro_ch) for gen in gens),
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
