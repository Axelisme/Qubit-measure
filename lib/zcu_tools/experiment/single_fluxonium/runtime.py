"""Single-fluxonium runtime; acquisition awaits single-cavity ADC integration."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import NoReturn

import numpy as np

from zcu_tools.core.simulation import (
    AcquireRequest,
    EnvironmentSnapshot,
    RoundRequest,
    TimelineTemplate,
)
from zcu_tools.device import DeviceManager, FakeDevice, FakeDeviceInfo
from zcu_tools.experiment.simulation import SimulationEnvironment
from zcu_tools.program.v2.mocksoc import make_mock_soccfg
from zcu_tools.program.v2.simulation import SimulationBinding, SimulationSoc
from zcu_tools.resources.single_fluxonium import (
    load_single_fluxonium,
    parse_single_fluxonium,
)
from zcu_tools.resources.single_fluxonium.schema import SingleFluxoniumSettings
from zcu_tools.simulate.evolution import DriveSource, ModelSource
from zcu_tools.simulate.fluxonium import (
    DispersiveResonator,
    FluxoniumNoise,
    SingleFluxoniumModel,
)

from . import replay
from .numeric import dc_flux
from .template import SingleFluxoniumPlan, plan_template


@dataclass(frozen=True)
class SingleFluxoniumFacts:
    """Physical inputs shared by templates at one exact round snapshot.

    model: fixed-DC oscillator projection, retained eigenstates and charge n.
    source: Gibbs initial density, full-charge microwave quadratures in GHz/gain,
        quantized baths and continuous FluxExpansion for the evolution engine.
    Both use settings' retained dimension. No ADC model, density propagation,
    receiver, pulse or random output is retained. Treat their arrays as read-only.
    """

    model: SingleFluxoniumModel
    source: ModelSource


def _build_facts(
    settings: SingleFluxoniumSettings, snapshot: EnvironmentSnapshot
) -> SingleFluxoniumFacts:
    qubit = settings.chip.qubit
    truncation = settings.numerics.truncation
    model = SingleFluxoniumModel(
        (qubit.ej_ghz, qubit.ec_ghz, qubit.el_ghz),
        dc_flux(settings, snapshot),
        cutoff=truncation.native_cutoff,
        retained_levels=truncation.levels,
        temperature_k=settings.chip.temperature_k,
    )
    coefficient = settings.drive_line.charge_drive_ghz_per_gain
    charge = model.charge_operator()
    drive = DriveSource(
        settings.drive_line.name, coefficient.real * charge, -coefficient.imag * charge
    )
    resonator = settings.readout.resonator
    source = model.model_source(
        (drive,),
        noise=FluxoniumNoise(qubit.q_cap, qubit.q_ind, qubit.xqp, qubit.aphi),
        resonator=DispersiveResonator(
            resonator.frequency_ghz,
            resonator.charge_coefficient_ghz,
            2
            * np.pi
            * (resonator.kappa_external_mhz + resonator.kappa_internal_mhz)
            / 1000,
            resonator.thermal_occupation,
        ),
    )
    return SingleFluxoniumFacts(model, source)


def _template_digest(template: TimelineTemplate) -> bytes:
    digest = sha256(b"single-fluxonium-template-v1")
    for array in (template.points, template.duration_us):
        digest.update(repr((array.dtype.str, array.shape)).encode())
        digest.update(array.tobytes())
    for event in template.events:
        digest.update(repr((event.gen_ch, event.phase_reference_us)).encode())
        for array in (
            event.start_us,
            event.duration_us,
            event.frequency_mhz,
            event.phase_deg,
            event.gain,
            event.envelope_index,
            event.active,
        ):
            digest.update(repr((array.dtype.str, array.shape)).encode())
            digest.update(array.tobytes())
        # PulseEnvelope uses immutable tuples; repr retains every complex value
        # and local sample coordinate, unlike numpy's abbreviated array repr.
        digest.update(repr(event.envelopes).encode())
    for window in template.readouts:
        digest.update(
            repr((window.channel, window.samples, window.sample_rate_mhz)).encode()
        )
        for array in (
            window.trigger_us,
            window.frequency_mhz,
            window.phase_deg,
            window.generator_channels,
        ):
            digest.update(repr((array.dtype.str, array.shape)).encode())
            digest.update(array.tobytes())
    return digest.digest()


class SingleFluxoniumSource:
    """Synchronous source with one snapshot's facts and one template-plan cache.

    settings: validated immutable single-fluxonium settings, retained as .settings.
    devices: caller-owned manager with the named startup DC driver, retained as
        .devices. Snapshot retains that driver's identity across registry removal
        or renaming; no real device is opened and no value is reset here.
    Only latest full settings/snapshot/backend/numeric/dependency key is resident.
    Plan keys additionally hash every borrowed control/readout array and complete
    waveform samples/times, never array or template identity. Cache hits return
    unchanged detached plans; consumers must not modify them. No RNG or measured
    outputs are cached. Failed builds drop previous residents. close releases
    cached model/plan state but never closes caller-owned devices.
    Direct callers must serialize all methods. Acquisition is deliberately
    unavailable: open_acquire and execute raise NotImplementedError until the
    single-cavity readout integration is connected. No placeholder IQ is returned.
    """

    def __init__(
        self, settings: SingleFluxoniumSettings, devices: DeviceManager
    ) -> None:
        self.settings = settings
        self.devices = devices
        self._source_device = devices.get_all_devices().get(
            settings.dc_bias.source.name
        )
        self._identity = sha256(settings.model_dump_json().encode()).hexdigest()
        self._facts: SingleFluxoniumFacts | None = None
        self._facts_key: (
            tuple[str, EnvironmentSnapshot, tuple[tuple[str, str], ...]] | None
        ) = None
        self._plan: SingleFluxoniumPlan | None = None
        self._plan_key: (
            tuple[str, EnvironmentSnapshot, tuple[tuple[str, str], ...], bytes] | None
        ) = None
        self._closed = False

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("single-fluxonium source is closed")

    def snapshot(self) -> EnvironmentSnapshot:
        """Capture the bound driver's current value, not startup initial, in A/V.

        Return a single-fluxonium-v1 EnvironmentSnapshot with settings identity
        and exactly one (source name, native value) pair. Missing device, wrong
        fixed unit or nonfinite/out-of-limits values raise ValueError. A closed
        source raises RuntimeError. No device state is changed.
        """
        self._require_open()
        source = self.settings.dc_bias.source
        if self._source_device is None:
            raise ValueError("dc_bias.source: missing startup device")
        info = self._source_device.get_info()
        if not isinstance(info, FakeDeviceInfo) or info.value_unit != source.unit:
            raise ValueError("dc_bias.source.unit: fixed source unit mismatch")
        snapshot = EnvironmentSnapshot(
            "single-fluxonium-v1", self._identity, ((source.name, info.value),)
        )
        dc_flux(self.settings, snapshot)
        return snapshot

    def model_facts(self, snapshot: EnvironmentSnapshot) -> SingleFluxoniumFacts:
        """Return lazy fixed-DC model/source for this full snapshot and settings.

        snapshot follows snapshot()'s native-value contract, including limits.
        Equal keys reuse facts; changed settings, snapshot or replay/numeric
        versions clear both caches before rebuilding. Numeric/model errors
        propagate without retaining a failed build. No evolution, ADC or sampling
        runs. RuntimeError follows close; invalid DC raises ValueError.
        """
        self._require_open()
        key = (self.settings.model_dump_json(), snapshot, replay.replay_versions())
        if self._facts is None or self._facts_key != key:
            self._clear()
            self._facts = _build_facts(self.settings, snapshot)
            self._facts_key = key
        return self._facts

    def plan_template(
        self, snapshot: EnvironmentSnapshot, template: TimelineTemplate
    ) -> SingleFluxoniumPlan:
        """Plan one complete template, reusing only its exact current content.

        Inputs/outputs and ValueError conditions follow template.plan_template.
        The snapshot/settings/numeric key and SHA256 of every current array and
        waveform sample/time select one resident plan; borrowed arrays are reread
        on every lookup. No physical facts need to be built for planning alone.
        This method does not evolve or acquire. RuntimeError follows close.
        """
        self._require_open()
        generation = (
            self.settings.model_dump_json(),
            snapshot,
            replay.replay_versions(),
        )
        if self._facts_key is not None and self._facts_key != generation:
            self._clear()
        key = (*generation, _template_digest(template))
        if self._plan is None or self._plan_key != key:
            self._plan = None
            self._plan_key = None
            self._plan = plan_template(self.settings, snapshot, template)
            self._plan_key = key
        return self._plan

    def open_acquire(self, request: AcquireRequest) -> NoReturn:
        """Reject either acquisition mode before creating a session or drawing RNG.

        request is the program's detached AcquireRequest. Always raises
        NotImplementedError for the unconnected single-cavity readout, or
        RuntimeError after close. No request state is retained.
        """
        self._require_open()
        raise NotImplementedError(
            f"single-cavity readout {request.mode} acquisition is not connected"
        )

    def execute(self, request: RoundRequest) -> NoReturn:
        """Reject a numeric round without producing fake IQ or bypassing ADC.

        request is program's RoundRequest for either mode. Raises the same
        NotImplementedError/RuntimeError as open_acquire; no state is retained.
        """
        self.open_acquire(request.acquire)

    def _clear(self) -> None:
        if self._facts is not None:
            self._facts.source.close()
        self._facts = None
        self._facts_key = None
        self._plan = None
        self._plan_key = None

    def close(self) -> None:
        """Release facts/plan caches and reject future use; idempotent, no device I/O.

        Call only after concurrent work has drained. The caller still owns driver
        cleanup. Already returned immutable numeric inputs remain usable.
        """
        self._clear()
        self._closed = True


def single_fluxonium(
    configuration: SingleFluxoniumSettings | str | Path,
) -> SimulationEnvironment:
    """Assemble one DC driver and a bound mock SoC, without acquiring or warmup.

    configuration is validated settings or a physical YAML path, relative to cwd.
    Startup applies initial once in fixed A/V units. Returned SimulationEnvironment
    owns the manager; caller closes its devices, or transfers each to the GUI.
    Source models/plans remain lazy. All three generators bind to the one ADC
    for transport admission. The planner still requires readout-generator ADC
    windows; transport routes do not enable microwave/fast-flux readout physics.
    The independent single-fluxonium backend deliberately raises NotImplementedError
    on any acquisition until single-cavity readout is connected. Successful assembly
    is not a measurement or MCP startup acceptance. No real hardware is opened.
    Invalid YAML/settings/profile/binding errors propagate. Builder failure closes
    its devices and preserves cleanup errors alongside the original failure.
    """
    settings = (
        parse_single_fluxonium(configuration)
        if isinstance(configuration, SingleFluxoniumSettings)
        else load_single_fluxonium(configuration)
    )
    devices = DeviceManager()
    try:
        dc = settings.dc_bias.source
        device = FakeDevice(address=dc.name, fast_mode=True, value_unit=dc.unit)
        devices.register_device(dc.name, device)
        device.setup(
            FakeDeviceInfo(
                address=device.address,
                value_unit=dc.unit,
                output="on",
                value=dc.initial,
                rampstep=dc.ramp_step,
            ),
            progress=False,
        )
        source = SingleFluxoniumSource(settings, devices)
        gens = (
            settings.drive_line.gen_ch,
            settings.flux_line.gen_ch,
            settings.readout.drive.gen_ch,
        )
        adc = settings.readout.adc.ro_ch
        soccfg = make_mock_soccfg(max(gens) + 1, adc + 1)
        binding = SimulationBinding(
            source.snapshot,
            source.execute,
            backend_version=replay.BACKEND_VERSION,
            seed=settings.numerics.seed,
            min_round_duration_s=settings.numerics.min_round_duration_s,
        )
        soc = SimulationSoc(soccfg, binding, routes=tuple((gen, adc) for gen in gens))
        return SimulationEnvironment(
            soc,
            soccfg,
            devices,
            "Single fluxonium: assembly only; single-cavity acquisition is not connected",
        )
    except Exception as error:
        try:
            devices.close_all_devices()
        except ExceptionGroup as cleanup:
            raise ExceptionGroup(
                "single-fluxonium startup and cleanup failed", [error, cleanup]
            ) from error
        raise
