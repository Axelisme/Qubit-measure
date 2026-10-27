"""Single-fluxonium acquisition with complete timelines and single-cavity ADC."""

import logging
from dataclasses import dataclass
from hashlib import sha256
from math import prod
from pathlib import Path
from typing import Literal

import numpy as np

from zcu_tools.core.simulation import (
    AcquireRequest,
    CancellationQuery,
    EnvironmentSnapshot,
    RoundRequest,
    RoundSignals,
    TimelineTemplate,
)
from zcu_tools.device import DeviceManager, FakeDevice, FakeDeviceInfo
from zcu_tools.experiment.simulation import SimulationEnvironment
from zcu_tools.program.v2.mocksoc import make_mock_soccfg
from zcu_tools.program.v2.simulation import (
    IntegratedRound,
    SimulationBinding,
    SimulationCallbacks,
    SimulationClock,
    SimulationSoc,
    SimulationTiming,
)
from zcu_tools.resources.single_fluxonium import (
    load_single_fluxonium,
    parse_single_fluxonium,
)
from zcu_tools.resources.single_fluxonium.schema import SingleFluxoniumSettings
from zcu_tools.simulate.adc import (
    PreparedRawShots,
    ReadoutModel,
    normalize_populations,
)
from zcu_tools.simulate.evolution import (
    Basis,
    DriveSource,
    Engine,
    ModelSource,
    batch_populations,
)
from zcu_tools.simulate.fluxonium import (
    FluxoniumNoise,
    SingleFluxoniumModel,
)

from . import replay
from .acquire import SingleFluxoniumAcquire, SingleFluxoniumPreparation
from .drift import CoherenceDrift
from .numeric import dc_flux
from .readout import dispersive_resonator, prepare_readout, readout_model
from .template import SingleFluxoniumPlan, plan_template


@dataclass(frozen=True)
class SingleFluxoniumFacts:
    """Physical inputs shared by templates at one exact round snapshot.

    model: fixed-DC oscillator projection, retained eigenstates and charge n.
    source: Gibbs initial density, full-charge microwave quadratures in GHz/gain,
        quantized baths and continuous FluxExpansion for the evolution engine.
    readout: one physical cavity with Phi0 dressed frequencies and DC rates.
    All use settings' retained dimension. No density propagation, calibrated
    receiver, pulse or random output is retained. Treat arrays as read-only.
    """

    model: SingleFluxoniumModel
    source: ModelSource
    readout: ReadoutModel


def _build_structure(
    settings: SingleFluxoniumSettings, snapshot: EnvironmentSnapshot
) -> SingleFluxoniumModel:
    qubit = settings.chip.qubit
    truncation = settings.numerics.truncation
    return SingleFluxoniumModel(
        (qubit.ej_ghz, qubit.ec_ghz, qubit.el_ghz),
        dc_flux(settings, snapshot),
        cutoff=truncation.native_cutoff,
        retained_levels=truncation.levels,
        temperature_k=settings.chip.temperature_k,
    )


def _build_facts(
    settings: SingleFluxoniumSettings, model: SingleFluxoniumModel
) -> SingleFluxoniumFacts:
    qubit = settings.chip.qubit
    coefficient = settings.drive_line.charge_drive_ghz_per_gain
    charge = model.charge_operator()
    drive = DriveSource(
        settings.drive_line.name, coefficient.real * charge, -coefficient.imag * charge
    )
    source = model.model_source(
        (drive,),
        noise=FluxoniumNoise(qubit.q_cap, qubit.q_ind, qubit.xqp, qubit.aphi),
        resonator=dispersive_resonator(settings),
        slow_envelope_segments_per_period=settings.numerics.slow_envelope_segments_per_period,
    )
    try:
        return SingleFluxoniumFacts(model, source, readout_model(settings, model, source))
    except Exception:
        source.close()
        raise


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


@dataclass(frozen=True)
class SingleFluxoniumRuntimeOptions:
    """Execution resources selected at source construction, not during acquisition.

    engine: fresh exclusive Engine transferred to the source, closed by close().
        None creates an Engine with 4 MiB routing-facts residency.
    clock: shared environment SimulationClock; source never closes/resets it.
        None creates a standalone clock. All qubits in an environment can share it.
    truth_log: private JSONL file path, or None for a fresh session directory
        under ~/.local/state/zcu-tools/simulation/single-fluxonium. Only enabled
        drift writes files. The source opens/closes each append; I/O errors propagate.
    """

    engine: Engine | None = None
    clock: SimulationClock | None = None
    truth_log: Path | None = None


class SingleFluxoniumSource:
    """Synchronous source with one snapshot's facts and one template-plan cache.

    settings: validated immutable single-fluxonium settings, retained as .settings.
    devices: caller-owned manager with the named startup DC driver, retained as
        .devices. Snapshot retains that driver's identity across registry removal
        or renaming; no real device is opened and no value is reset here.
    Full facts use the latest settings/snapshot/backend/numeric/dependency key.
    Loss-independent structure uses a separate key excluding losses and drift.
    Plan keys additionally hash every borrowed control/readout array and complete
    waveform samples/times, never array or template identity. Cache hits return
    unchanged detached plans; consumers must not modify them. No RNG or measured
    outputs are cached. Failed builds drop full facts and preparation. close releases
    cached model/plan state but never closes caller-owned devices.
    Direct callers must serialize all methods. One Engine owns routing facts,
    and one complete preparation survives across acquires. Loss-independent
    projection/eigensystems survive loss changes; full models and preparations
    do not. No noisy result or sampling identity enters retained preparations.
    readout_kind selects one terminal pulse or coherent mux tones on the same
    physical cavity. It does not add receivers, coupling or hardware routes.
    runtime selects execution resources through SingleFluxoniumRuntimeOptions;
    None uses that record's defaults.
    An injected engine transfers exclusive ownership to this source; shared
    clocks remain environment-owned. Its cache caps do not bound process RSS.
    private_replay optionally selects an explicit absolute author archive root;
    None disables I/O. Private single-model replay accepts pulse mode only;
    combining mux mode with private_replay raises ValueError before acquisition.
    runtime.clock optionally supplies the environment's shared SimulationClock;
    None creates a standalone clock. runtime.truth_log selects a private JSONL
    file; None uses a fresh session directory. Only enabled drift writes truth.
    The caller owns devices, including after source close.
    """

    def __init__(
        self,
        settings: SingleFluxoniumSettings,
        devices: DeviceManager,
        *,
        private_replay: Path | None = None,
        readout_kind: Literal["pulse", "mux"] = "pulse",
        runtime: SingleFluxoniumRuntimeOptions | None = None,
    ) -> None:
        if private_replay is not None and not private_replay.is_absolute():
            raise ValueError("private replay requires an explicit absolute root")
        if readout_kind not in ("pulse", "mux"):
            raise ValueError("readout_kind must be pulse or mux")
        if readout_kind == "mux" and private_replay is not None:
            raise ValueError("single-fluxonium private replay requires pulse readout")
        self._readout_kind: Literal["pulse", "mux"] = readout_kind
        self.settings = settings
        self.devices = devices
        self.private_replay = private_replay
        runtime = SingleFluxoniumRuntimeOptions() if runtime is None else runtime
        self._clock = SimulationClock() if runtime.clock is None else runtime.clock
        self._truth_log = runtime.truth_log
        self._drift: CoherenceDrift | None = None
        self._effective_settings = settings
        self._pending_acquire = False
        self.engine = (
            Engine(workers=1)
            if runtime.engine is None
            else runtime.engine
        )
        self._prepared: SingleFluxoniumPreparation | None = None
        self._session: SingleFluxoniumAcquire | None = None
        self._source_device = devices.get_all_devices().get(
            settings.dc_bias.source.name
        )
        self._structure: SingleFluxoniumModel | None = None
        self._structure_key: bytes | None = None
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

    @property
    def effective_settings(self) -> SingleFluxoniumSettings:
        """Current acquisition's frozen losses, or configured settings before begin.

        Reassigning settings discards the old effective values immediately;
        the next begin initializes drift from the new configuration. With drift
        disabled, this returns settings unchanged. Replay archives this snapshot.
        """
        if self._drift is not None and self._drift.settings == self.settings:
            return self._effective_settings
        return self.settings

    def begin_acquire(self, *, seed: int | None = None) -> None:
        """Sample once at admission, before request construction or round snapshots.

        The binding calls this with its root seed; direct callers may omit seed
        to use numerics.seed. open_acquire consumes this pending preparation
        without another sample/log row. Repeated begin calls start new
        acquisitions, even if the prior call constructed no request. The clock
        is shared by the environment; first drift use starts at YAML losses.
        Errors, including truth-log I/O, propagate before a session is opened.
        """
        self._require_open()
        if self._session is not None:
            self._session.invalidate()
        if self.settings.drift is not None:
            if self._drift is None or self._drift.settings != self.settings:
                self._drift = CoherenceDrift(
                    self.settings,
                    seed=self.settings.numerics.seed if seed is None else seed,
                    truth_log=self._truth_log,
                )
            self._effective_settings = self._drift.sample(self._clock.now())
        self._pending_acquire = True

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
            "single-fluxonium-v1", self.identity, ((source.name, info.value),)
        )
        dc_flux(self.settings, snapshot)
        return snapshot

    def model_facts(self, snapshot: EnvironmentSnapshot) -> SingleFluxoniumFacts:
        """Return current-loss models with reusable loss-independent structure.

        snapshot follows snapshot()'s native-value contract, including limits.
        Loss changes discard full models, plans and preparation, but retain the
        fixed-DC projection and eigensystems. Other settings, DC values or numeric
        revisions discard structure too. Failed builds retain no partial facts.
        No density evolution, ADC response solve or random sampling runs.
        RuntimeError follows close; invalid DC raises ValueError.
        """
        self._require_open()
        key = (
            self.effective_settings.model_dump_json(),
            snapshot,
            replay.replay_versions(),
        )
        structure_key = sha256(
            repr(
                (
                    self.effective_settings.model_dump_json(
                        exclude={
                            "chip": {"qubit": {"q_cap", "q_ind", "xqp", "aphi"}},
                            "drift": True,
                        }
                    ),
                    snapshot.version,
                    snapshot.values,
                    key[2],
                )
            ).encode()
        ).digest()
        if self._structure_key != structure_key:
            self._clear()
            self._structure = None
            self._structure_key = None
            self._structure = _build_structure(self.effective_settings, snapshot)
            self._structure_key = structure_key
        if self._facts is None or self._facts_key != key:
            self._clear()
            if self._structure is None:
                raise RuntimeError("single-fluxonium structure is unavailable")
            self._facts = _build_facts(self.effective_settings, self._structure)
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
            self.effective_settings.model_dump_json(),
            snapshot,
            replay.replay_versions(),
        )
        if self._facts_key is not None and self._facts_key != generation:
            self._clear()
        key = (*generation, _template_digest(template))
        if self._plan is None or self._plan_key != key:
            self._plan = None
            self._plan_key = None
            self._plan = plan_template(
                self.settings, snapshot, template, readout_kind=self._readout_kind
            )
            self._plan_key = key
        return self._plan

    @property
    def identity(self) -> str:
        """SHA256 of current effective settings, used by snapshot and replay.

        Reassigning .settings changes this identity and invalidates preparation
        keys on the next call. The startup driver object remains unchanged.
        """
        return sha256(self.effective_settings.model_dump_json().encode()).hexdigest()

    def prepare(
        self,
        request: AcquireRequest,
        snapshot: EnvironmentSnapshot,
        *,
        cancel: CancellationQuery | None = None,
    ) -> SingleFluxoniumPreparation:
        """Prepare noiseless ADC inputs; used by the serialized acquire session.

        Requires an admitted request (see open_acquire) and matching DC snapshot.
        Complete template bytes, layout, settings, snapshot and numeric versions
        key a single resident. Evolution alone segments the complete timelines.
        ADC normalization is the sole probability repair. Errors clear partial
        preparations; they propagate without a result.
        cancel is checked between evolution chunks; None disables cancellation.
        """
        self._require_open()
        try:
            facts = self.model_facts(snapshot)
            digest = sha256(
                repr(
                    (
                        self._facts_key,
                        request.point_dims,
                        request.loop_dims,
                        request.avg_level,
                        request.mode,
                        request.reps,
                        request.readouts,
                    )
                ).encode()
            )
            for template in request.templates:
                digest.update(_template_digest(template))
            key = digest.digest()
            if (
                self._prepared is not None
                and self._prepared.key == key
                and self.engine.reuse(facts.source, key)
            ):
                self._prepared.paths.warn_exceeded()
                return self._prepared
            plans = tuple(
                self.plan_template(snapshot, template) for template in request.templates
            )
            if any(any(plan.cavity_drives) for plan in plans):
                raise ValueError(
                    "single fluxonium supports only terminal readout; earlier readout-generator pulses are unsupported"
                )
            evolved = self.engine.evolve(
                facts.source,
                tuple(plan.batch for plan in plans),
                cancel=cancel,
                reuse_key=key,
            )
            dc = facts.source.model((0,))
            basis = Basis(dc.identity, facts.readout.state_labels, dc.eigenvectors)
            probabilities = np.empty((prod(request.point_dims), len(basis.labels)))
            probabilities[evolved.points] = normalize_populations(
                batch_populations(evolved.densities, basis, basis)
            )
            probabilities = probabilities.reshape(
                (*request.point_dims, len(basis.labels))
            )
            probabilities.setflags(write=False)
            channel, batch = prepare_readout(
                self.settings, facts.readout, plans, request
            )
            raw = PreparedRawShots(
                probabilities,
                {channel.channel: batch},
                reps=request.reps,
                rtol=self.settings.numerics.solver.rtol,
                atol=self.settings.numerics.solver.atol,
            )
            self._prepared = SingleFluxoniumPreparation(
                key, probabilities, channel, raw, evolved.paths
            )
            return self._prepared
        except Exception:
            self._prepared = None
            raise

    def open_acquire(self, request: AcquireRequest) -> SingleFluxoniumAcquire:
        """Start a request, invalidating the previous session and freezing losses.

        Nonempty templates and the current PCG64 v2 stream version are required.
        Unsupported inputs raise ValueError before numeric execution. Closed
        sources raise RuntimeError. A pending begin_acquire is consumed exactly
        once; otherwise this call samples drift and writes truth itself. All
        sweep points and rounds share those losses; sampling RNG is independent.
        The binding serializes calls and owns Stop.
        """
        self._require_open()
        if not request.templates:
            raise ValueError("single fluxonium requires timeline templates")
        if request.rng_version != f"PCG64/SeedSequence-v2/numpy-{np.__version__}":
            raise ValueError("single-fluxonium RNG version mismatch")
        if not self._pending_acquire:
            self.begin_acquire()
        self._pending_acquire = False
        if self._session is not None:
            self._session.invalidate()
        self._session = SingleFluxoniumAcquire(self, request)
        return self._session

    def execute(self, request: RoundRequest) -> IntegratedRound | RoundSignals:
        """Execute detached round inputs with fresh round RNG and deferred raw.

        Reuses the active session only for the identical request object. A new
        request invalidates the old session. No backend work is repeated when
        consumers later request raw. Numeric/ADC/replay failures propagate.
        """
        self._require_open()
        if self._session is None or self._session.request is not request.acquire:
            self.open_acquire(request.acquire)
        if self._session is None:
            raise RuntimeError("single-fluxonium acquire was not opened")
        return self._session.sample_round(
            request.snapshot, request.index, cancel=request.cancel
        )

    def _clear(self) -> None:
        if self._facts is not None:
            self._facts.source.close()
        self._prepared = None
        self._facts = None
        self._facts_key = None
        self._plan = None
        self._plan_key = None

    def close(self) -> None:
        """Release facts/plan caches and reject future use; idempotent, no device I/O.

        Call only after concurrent work has drained. The caller still owns driver
        cleanup. Already returned immutable numeric inputs remain usable.
        """
        if self._session is not None:
            self._session.invalidate()
        self._clear()
        self._structure = None
        self._structure_key = None
        self.engine.close()
        self._closed = True


def single_fluxonium(
    configuration: SingleFluxoniumSettings | str | Path,
    *,
    private_replay: Path | None = None,
    clock: SimulationClock | None = None,
    truth_log: Path | None = None,
) -> SimulationEnvironment:
    """Assemble one DC driver and a bound mock SoC, without acquiring or warmup.

    configuration is validated settings or a physical YAML path, relative to cwd.
    Startup applies initial once in fixed A/V units. Returned SimulationEnvironment
    owns the manager; caller closes its devices, or transfers each to the GUI.
    Source models/plans remain lazy. The microwave and readout generators, plus
    the optional fast-flux generator, bind to the one ADC
    for transport admission. The planner still requires readout-generator ADC
    windows; transport routes do not enable microwave/fast-flux readout physics.
    The backend evolves complete timelines and uses one physical ADC cavity.
    private_replay enables author-only detached round inputs at an absolute root.
    clock optionally injects a shared monotonic physical clock, otherwise a new
    clock is owned by this environment. truth_log selects a private JSONL file
    for enabled drift, defaulting to a new session under ~/.local/state/
    zcu-tools/simulation/single-fluxonium. No truth I/O occurs with drift off.
    Successful assembly is not MCP acceptance. No real hardware is opened.
    Invalid YAML/settings/profile/binding errors propagate. Builder failure closes
    its devices and preserves cleanup errors alongside the original failure.
    """
    settings = (
        parse_single_fluxonium(configuration)
        if isinstance(configuration, SingleFluxoniumSettings)
        else load_single_fluxonium(configuration)
    )
    devices = DeviceManager()
    source: SingleFluxoniumSource | None = None
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
        sim_clock = SimulationClock() if clock is None else clock
        source = SingleFluxoniumSource(
            settings,
            devices,
            private_replay=private_replay,
            runtime=SingleFluxoniumRuntimeOptions(clock=sim_clock, truth_log=truth_log),
        )
        gens = (
            settings.drive_line.gen_ch,
            settings.readout.drive.gen_ch,
            *((settings.flux_line.gen_ch,) if settings.flux_line is not None else ()),
        )
        adc = settings.readout.adc.ro_ch
        soccfg = make_mock_soccfg(max(gens) + 1, adc + 1)
        binding = SimulationBinding(
            source.snapshot,
            source.execute,
            backend_version=replay.BACKEND_VERSION,
            seed=settings.numerics.seed,
            timing=SimulationTiming(
                time_scale=settings.numerics.time_scale, clock=sim_clock
            ),
            callbacks=SimulationCallbacks(
                on_begin=lambda: source.begin_acquire(seed=binding.root_seed)
            ),
        )
        soc = SimulationSoc(soccfg, binding, routes=tuple((gen, adc) for gen in gens))
        return SimulationEnvironment(
            soc,
            soccfg,
            devices,
            "Single fluxonium with microwave control and one readout cavity",
            close_backend=source.close,
        )
    except Exception as error:
        errors = [error]
        if source is not None:
            try:
                source.close()
            except Exception as backend_error:
                logging.exception("Single-fluxonium backend cleanup failed")
                errors.append(backend_error)
        try:
            devices.close_all_devices()
        except Exception as device_error:
            logging.exception("Single-fluxonium device cleanup failed")
            errors.append(device_error)
        if len(errors) > 1:
            raise ExceptionGroup(
                "single-fluxonium startup and cleanup failed", errors
            ) from error
        raise
