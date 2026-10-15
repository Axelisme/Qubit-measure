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
from zcu_tools.simulate.adc import ADC, PopulationInput, SamplingIdentity
from zcu_tools.simulate.evolution import (
    Evolution,
    Populations,
    SolverSettings,
    populations,
)

from .point import FTFPointPlan, plan_point


@dataclass(frozen=True)
class _Prepared:
    plan: FTFPointPlan
    populations: Populations
    adc: ADC


class FTFSource:
    """Environment coordinator with immutable settings and one driver manager.

    settings: validated FTF parameters retained for private in-memory reproduction;
    devices: the exact manager shared by the host. snapshot reads live DC once
    per round; execute uses only RoundRequest numeric inputs, not drivers. Up to
    32 deterministic point preparations are retained; shot/noise draws never are.
    This is author-side composition, not an ordinary GUI/MCP truth interface.
    """

    def __init__(self, settings: FTFSettings, devices: DeviceManager) -> None:
        self.settings = settings
        self.devices = devices
        self.identity = sha256(settings.model_dump_json().encode()).hexdigest()
        self._prepared: dict[
            tuple[tuple[tuple[str, float], ...], PointTimeline], _Prepared
        ] = {}

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
        self, snapshot: EnvironmentSnapshot, timeline: PointTimeline
    ) -> _Prepared:
        key = snapshot.values, timeline
        if key not in self._prepared:
            plan = plan_point(self.settings, snapshot, timeline)
            solver = self.settings.solver
            # Explicit three-tool composition: density -> target populations -> ADC.
            state = Evolution(
                plan.evolution_model,
                SolverSettings(solver.rtol, solver.atol, solver.max_step_ns),
            ).evolve(
                plan.joint.initial_state(), plan.controls, end_ns=plan.readout_start_ns
            )
            mixture = populations(state, plan.readout_basis)
            adc = ADC(plan.readout_model, rtol=solver.rtol, atol=solver.atol)
            if len(self._prepared) >= 32:
                del self._prepared[next(iter(self._prepared))]
            self._prepared[key] = _Prepared(plan, mixture, adc)
        return self._prepared[key]

    def execute(self, request: RoundRequest) -> RoundSignals:
        """Generate this round's single ADC sums/traces from detached timelines.

        Equal physical executions share evolution and conditional response, but
        receive independent joint-initial/noise draws. Point grouping does not
        depend on sweep names. No state/filter tail crosses shots/rounds. Raw and
        averaged acquisition share this one numeric realization via the adapter.
        """
        acquire = request.acquire
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
            prepared = self._prepare(request.snapshot, timeline)
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
        return RoundSignals((ChannelSignals(self.settings.receiver.ro_ch, output),))


def ftf(configuration: FTFSettings | str | Path) -> SimulationEnvironment:
    """Build a fixed-profile FTF environment from named values or a YAML path.

    Startup creates fixed V/A FakeDevices and applies initial values once. The
    returned manager owns cleanup; builder failures close all devices and preserve
    both original and cleanup errors. Channels use make_mock_soccfg's supported
    profile, never arbitrary firmware overrides. Host receives no model truth.
    Weak dispersive/Markov/secular approximations and readout-origin flux freezing
    are documented in the module README; no runtime applicability gate is added.
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
        coordinator = FTFSource(settings, devices)
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
