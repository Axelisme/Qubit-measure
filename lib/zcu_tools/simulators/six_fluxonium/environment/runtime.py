"""Six single-model sources, seven DC currents and bounded ADC composition."""

import logging
from collections.abc import Collection
from dataclasses import replace
from hashlib import sha256
from math import isfinite, prod
from pathlib import Path
from time import monotonic
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray
from qick import QickConfig

from zcu_tools.device import DeviceManager, FakeDevice, FakeDeviceInfo
from zcu_tools.experiment.simulation import SimulationEnvironment
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
    round_statistics,
    sample_traces,
)
from zcu_tools.simulate.boundary import (
    AcquireRequest,
    ChannelSignals,
    EnvironmentSnapshot,
    RoundRequest,
    RoundSignals,
    SamplingIdentity,
)
from zcu_tools.simulate.evolution import Engine, RoutingHistory
from zcu_tools.simulators.single_fluxonium.config.schema import (
    DCBias,
    DCSource,
    SingleFluxoniumSettings,
)
from zcu_tools.simulators.single_fluxonium.environment import (
    SingleFluxoniumPreflight,
    SingleFluxoniumRuntimeOptions,
    SingleFluxoniumSource,
)
from zcu_tools.simulators.single_fluxonium.environment.drift import CoherenceDrift
from zcu_tools.simulators.six_fluxonium.config import (
    SixFluxoniumSettings,
    load_six_fluxonium,
    parse_six_fluxonium,
)
from zcu_tools.simulators.six_fluxonium.config.schema import CircuitBias
from zcu_tools.simulators.six_fluxonium.environment.results import (
    CavityRound,
    combine_integrated,
)
from zcu_tools.simulators.six_fluxonium.environment.routing import (
    RoutedGroup,
    route_request,
)
from zcu_tools.simulators.telemetry import RoutingTelemetry


def _reduced_flux(
    bias: CircuitBias, coil_current: float, local_current: float
) -> float:
    return (coil_current - bias.coil_offset) / bias.coil_period + (
        local_current - bias.offset
    ) / bias.period


def _single_settings(
    settings: SixFluxoniumSettings,
    cavity: int,
    channel: int,
    gen: int,
    flux_phi0: float,
) -> SingleFluxoniumSettings:
    qubits = settings.qubits.ordered()
    qubit = qubits[cavity]
    dc = qubit.dc_bias.source
    # The pulse receiver uses Q1's calibration; mux receivers follow YAML IDs.
    receiver = (
        qubits[0]
        if channel == settings.pulse_ro_ch
        else next(q for q in qubits if q.readout.adc.ro_ch == channel)
    )
    adc = receiver.readout.adc.model_copy(update={"ro_ch": channel})
    return SingleFluxoniumSettings(
        format=qubit.format,
        version=qubit.version,
        chip=qubit.chip,
        dc_bias=DCBias(
            offset_phi0=flux_phi0,
            source=DCSource(
                name=dc.name,
                unit=dc.unit,
                limits=dc.limits,
                initial=dc.initial,
                ramp_step=dc.ramp_step,
                # Six owns both current conversions; single receives fixed flux.
                flux_per_unit=0.0,
            ),
        ),
        drive_lines={qubit.drive_line.name: qubit.drive_line},
        flux_line=None,
        readout=qubit.readout.model_copy(
            update={
                "drive": qubit.readout.drive.model_copy(update={"gen_ch": gen}),
                "adc": adc,
            }
        ),
        numerics=qubit.numerics.model_copy(
            update={
                "slow_envelope_segments_per_period": settings.numerics.slow_envelope_segments_per_period
            }
        ),
    )


class SixFluxoniumSource:
    """Serialized six-model backend with six local DC drivers and one coil.

    settings: validated six-system material. devices: manager retaining seven
    addressed FakeDevices. This source retains those exact drivers. The six
    SingleFluxoniumSources keep independent oscillator/evolution/ADC caches and
    physical parameters. The public sources tuple is ordered Q1..Q6, independent
    of YAML mapping order. Each source exposes the effective settings last sent
    to that circuit's physics preparation, including its routed receiver/flux.
    Each preparation receives the sum of local and coil
    reduced flux as a fixed offset, without changing single-source contracts.
    settings.numerics.routing_cache_bytes caps only environment-wide routing
    facts; each source receives one sixth, rounded down. Retained models,
    complete preparations and ADC numerical data are outside this cap.
    No joint Hilbert space or pulsed flux route is created. Call snapshot/execute
    only serially; close releases numeric resources but never caller devices.
    clock is an environment-owned monotonic clock in seconds, shared with every
    source and the outer binding; None creates one. truth_log selects private
    JSONL output for enabled qubits, or a fresh six-fluxonium session directory.
    begin_acquire freezes losses once for every point/round; execute also admits
    new standalone requests. Truth I/O failures propagate. No private replay is
    provided. The caller retains device and injected clock ownership.
    history is borrowed by all six sources; None gives each ModelSource its own.
    families restricts every source's Engine, including projection continuations;
    unknown names raise on construction, unavailable domains raise on acquire.
    workers selects each source Engine's parallel width across timelines (default 1).
    Values below 1 raise ValueError. Sources are still prepared serially.
    readout_wall_s sums six-owner ADC sampling and result assembly seconds;
    source-owned ADC preparation and Engine counters are added at finish_acquire.
    begin_acquire resets all counters. telemetry is the optional writer used
    only by finish_acquire, never by an individual source or routed group.
    """

    def __init__(  # noqa: PLR0913  # tolerated: independent optional controls retain caller-selected workers, clock, truth log and routing
        self,
        settings: SixFluxoniumSettings,
        devices: DeviceManager,
        *,
        workers: int = 1,
        clock: SimulationClock | None = None,
        truth_log: Path | None = None,
        history: RoutingHistory | None = None,
        families: Collection[str] | None = None,
    ) -> None:
        self.settings = settings
        policy = settings.numerics.telemetry
        self.telemetry = (
            RoutingTelemetry(
                (Path(__file__).resolve().parents[5] / ".sim_telemetry")
                if policy.directory is None
                else policy.directory.resolve()
            )
            if policy.enabled
            else None
        )
        self.readout_wall_s = 0.0
        self._clock = SimulationClock() if clock is None else clock
        self._session = uuid4().hex
        self._truth_log = (
            truth_log
            if truth_log is not None
            else Path.home()
            / ".local/state/zcu-tools/simulation/six-fluxonium"
            / self._session
            / "truth.jsonl"
        )
        self._drifts: tuple[CoherenceDrift | None, ...] | None = None
        self._effective_settings = settings
        self._request: AcquireRequest | None = None
        self._pending_acquire = False
        self._dc_sources = (
            settings.dc_source,
            *(q.dc_bias.source for q in settings.qubits.ordered()),
        )
        self._dc_devices = tuple(
            devices.get_all_devices().get(dc.name) for dc in self._dc_sources
        )
        self.sources = tuple(
            SingleFluxoniumSource(
                _single_settings(
                    settings,
                    index,
                    qubit.readout.adc.ro_ch,
                    settings.mux_gen_ch,
                    _reduced_flux(
                        qubit.dc_bias,
                        settings.dc_source.initial,
                        qubit.dc_bias.source.initial,
                    ),
                ),
                devices,
                readout_kind="mux",
                runtime=SingleFluxoniumRuntimeOptions(
                    engine=Engine(
                        workers=workers,
                        routing_cache_bytes=settings.numerics.routing_cache_bytes // 6,
                        families=families,
                    ),
                    clock=self._clock,
                    history=history,
                ),
            )
            for index, qubit in enumerate(settings.qubits.ordered())
        )
        self._closed = False
        self._identity = sha256(settings.model_dump_json().encode()).hexdigest()

    def begin_acquire(self, *, seed: int | None = None) -> None:
        """Freeze each circuit's losses at one shared simulated time in seconds.

        seed is the binding root seed; omitted uses Q1 numerics.seed (or fresh
        entropy). Stable Q1..Q6 identities namespace independent drift streams.
        Repeated calls start new acquisitions, even without a request. All
        points, routed ADCs and rounds consume this frozen set. Closed sources
        raise RuntimeError; truth-log I/O errors propagate before admission.
        """
        if self._closed:
            raise RuntimeError("six-fluxonium source is closed")
        if self._drifts is None:
            root = self.settings.qubits.Q1.numerics.seed if seed is None else seed
            if root is None and any(
                qubit.drift is not None for qubit in self.settings.qubits.ordered()
            ):
                root = int(
                    np.random.SeedSequence().generate_state(1, dtype=np.uint64)[0]
                )
            self._drifts = tuple(
                CoherenceDrift(
                    source.settings.model_copy(update={"drift": qubit.drift}),
                    seed=root,
                    truth_log=self._truth_log,
                    qubit=f"Q{index + 1}",
                    session=self._session,
                )
                if qubit.drift is not None
                else None
                for index, (qubit, source) in enumerate(
                    zip(self.settings.qubits.ordered(), self.sources, strict=True)
                )
            )
        if any(drift is not None for drift in self._drifts):
            sim_now = self._clock.now()
            qubits = {
                f"Q{index + 1}": qubit.model_copy(
                    update={"chip": drift.sample(sim_now).chip}
                )
                if drift is not None
                else qubit
                for index, (qubit, drift) in enumerate(
                    zip(self.settings.qubits.ordered(), self._drifts, strict=True)
                )
            }
            self._effective_settings = self.settings.model_copy(
                update={"qubits": self.settings.qubits.model_copy(update=qubits)}
            )
        for source in self.sources:
            source.begin_acquire(seed=seed)
        self.readout_wall_s = 0.0
        self._request = None
        self._pending_acquire = True

    def finish_acquire(self, outcome: AcquireOutcome) -> None:
        """Write one aggregate for all routed sources from binding's outcome.

        Captured attempts supply points/rounds. Engine and ADC totals include
        failed spans; binding alone supplies compile time and complete. Writer
        failures warn once without changing results; disabled policy writes nothing.
        """
        if self.telemetry is not None:
            self.telemetry.write(
                {
                    "points": prod(outcome.rounds[0].acquire.point_dims)
                    if outcome.rounds
                    else 0,
                    "rounds": len(outcome.rounds),
                    "wall_s": outcome.wall_s,
                    "compile_wall_s": outcome.compile_wall_s,
                    "evolve_wall_s": sum(
                        source.evolve_wall_s for source in self.sources
                    ),
                    "readout_wall_s": self.readout_wall_s
                    + sum(source.readout_wall_s for source in self.sources),
                    "complete": outcome.complete,
                    "extras": {},
                },
                paths=[path for source in self.sources for path in source.paths],
            )

    def snapshot(self) -> EnvironmentSnapshot:
        """Capture seven live A values with full settings identity, without reset.

        Return coil_bias followed by Q1_bias..Q6_bias, retaining native currents,
        not combined flux. Invalid current/unit or a missing driver raises
        ValueError. A closed source raises RuntimeError. Registry removal or
        renaming does not replace bound drivers or restore startup currents.
        """
        if self._closed:
            raise RuntimeError("six-fluxonium source is closed")
        values = []
        for dc, device in zip(self._dc_sources, self._dc_devices, strict=True):
            if device is None:
                raise ValueError(f"{dc.name}: missing startup device")
            info = device.get_info()
            if not isinstance(info, FakeDeviceInfo) or info.value_unit != dc.unit:
                raise ValueError(f"{dc.name}.unit: fixed source unit mismatch")
            values.append((dc.name, info.value))
        snapshot = EnvironmentSnapshot(
            "six-fluxonium-v1", self._identity, tuple(values)
        )
        self._currents(snapshot)
        return snapshot

    def _currents(self, snapshot: EnvironmentSnapshot) -> dict[str, float]:
        currents = dict(snapshot.values)
        if len(snapshot.values) != 7 or set(currents) != {
            dc.name for dc in self._dc_sources
        }:
            raise ValueError(
                "DC snapshot requires exactly seven matching current sources"
            )
        for dc in self._dc_sources:
            value = currents[dc.name]
            if not isfinite(value) or not dc.limits[0] <= value <= dc.limits[1]:
                raise ValueError(f"{dc.name}.value: nonfinite or outside source limits")
        return currents

    def _preflight_group(
        self,
        request: AcquireRequest,
        currents: dict[str, float],
        group: RoutedGroup,
    ) -> SingleFluxoniumPreflight:
        source = self.sources[group.cavity]
        bias = self.settings.qubits.ordered()[group.cavity].dc_bias
        flux = _reduced_flux(
            bias, currents[self.settings.dc_source.name], currents[bias.source.name]
        )
        settings = tuple(
            _single_settings(
                self._effective_settings, group.cavity, channel, group.gen_ch, flux
            )
            for channel in group.channels
        )
        source.settings = settings[0]
        snapshot = EnvironmentSnapshot(
            "single-fluxonium-v1",
            source.identity,
            ((bias.source.name, currents[bias.source.name]),),
        )
        count = len(group.points)
        subrequest = replace(
            request,
            loop_dims=(request.reps, count),
            avg_level=0,
            readouts=tuple(w for w in request.readouts if w.channel in group.channels),
            point_dims=(count,),
            point_names=("routed_point",),
            templates=group.templates,
        )
        return source.preflight(
            subrequest,
            snapshot,
            receivers=tuple(value.readout.adc for value in settings),
        )

    def execute(self, round_request: RoundRequest) -> IntegratedRound | RoundSignals:
        """Evolve each observed independent circuit and return all requested ADCs.

        Routing and all source/ADC/template layout checks finish before any
        density evolution or sampling. Each point starts in
        that circuit's Gibbs state. Shared-cavity ADCs use the same physical
        populations and jump histories; distinct cavities draw independent
        streams. Integrated histograms retain all marginal counts in linear
        space; deferred raw retains detached ADC inputs beyond source close.
        Decimated companion raw comes from the same traces at the adapter.
        Invalid RNG/version/index or closed sources raise ValueError/RuntimeError.
        Numeric/ADC failures propagate without partial results. A new request
        consumes pending begin_acquire values, or freezes losses itself when no
        begin was called. Reusing that request across rounds never samples drift.
        """
        if self._closed:
            raise RuntimeError("six-fluxonium source is closed")
        request, snapshot, index = (
            round_request.acquire,
            round_request.snapshot,
            round_request.index,
        )
        if type(index) is not int or index < 0:
            raise ValueError("round index must be a nonnegative integer")
        if request.rng_version != f"PCG64/SeedSequence-v2/numpy-{np.__version__}":
            raise ValueError("six-fluxonium RNG version mismatch")
        if self._request is not request:
            if not self._pending_acquire:
                self.begin_acquire(seed=request.root_seed)
            self._request = request
            self._pending_acquire = False
        currents = self._currents(snapshot)
        groups = route_request(self.settings, request)
        validated = tuple(
            self._preflight_group(request, currents, group) for group in groups
        )
        rounds = []
        traces: dict[int, NDArray[np.float64]] = {}
        for group, preflight in zip(groups, validated, strict=True):
            source = self.sources[group.cavity]
            source.settings = preflight.settings
            prepared = source.prepare_readouts(preflight, cancel=round_request.cancel)
            readout, channels, raw = (
                prepared.raw.readout,
                prepared.channels,
                prepared.raw,
            )
            # Cantor-pair the fixed-radix acquire/cavity identity with the
            # partition's first point, independent of request size. ADCs within
            # one group still share a single path draw.
            cavity_sequence = request.sequence * 6 + group.cavity
            first_point = int(group.points[0])
            diagonal = cavity_sequence + first_point
            sequence = diagonal * (diagonal + 1) // 2 + first_point
            started = monotonic()
            try:
                if request.mode == "integrated":
                    statistics = round_statistics(
                        readout,
                        channels,
                        reps=request.reps,
                        seed=request.root_seed,
                        acquire_sequence=sequence,
                        round_index=index,
                    )
                    rounds.append(
                        CavityRound(
                            group,
                            IntegratedRound(
                                statistics, raw.bind(statistics.raw_identity)
                            ),
                        )
                    )
                else:
                    result = sample_traces(
                        readout,
                        dict(raw.batches),
                        reps=request.reps,
                        identity=SamplingIdentity(
                            request.root_seed, (sequence, index, 0)
                        ),
                        rtol=raw.rtol,
                        atol=raw.atol,
                    )
                    for channel in result.channels:
                        shape = channel.iq.shape[1:]
                        if channel.channel not in traces:
                            traces[channel.channel] = np.empty(
                                (prod(request.point_dims), request.reps, *shape)
                            )
                        traces[channel.channel][group.points] = channel.iq.reshape(
                            len(group.points), request.reps, *shape
                        )
            finally:
                self.readout_wall_s += monotonic() - started
        return self._combine_round(round_request, tuple(rounds), traces)

    def _combine_round(
        self,
        round_request: RoundRequest,
        rounds: tuple[CavityRound, ...],
        traces: dict[int, NDArray[np.float64]],
    ) -> IntegratedRound | RoundSignals:
        request, index = round_request.acquire, round_request.index
        started = monotonic()
        try:
            if request.mode == "integrated":
                return combine_integrated(
                    request,
                    rounds,
                    tuple(
                        q.numerics.truncation.levels
                        for q in self.settings.qubits.ordered()
                    ),
                    index,
                )
            return RoundSignals(
                tuple(
                    ChannelSignals(
                        channel,
                        np.moveaxis(
                            values.reshape(
                                *request.point_dims, request.reps, *values.shape[2:]
                            ),
                            len(request.point_dims),
                            request.avg_level,
                        ).reshape(-1, *values.shape[2:]),
                    )
                    for channel, values in traces.items()
                )
            )
        finally:
            self.readout_wall_s += monotonic() - started

    def close(self) -> None:
        """Release six numeric backends and reject later work, leaving DC untouched.

        Idempotent. Call after acquisitions drain. Delivered raw sources retain
        their detached physical inputs and remain usable after close.
        """
        for source in self.sources:
            source.close()
        self._closed = True


def six_fluxonium(  # noqa: PLR0913  # tolerated: independent optional controls retain caller-selected workers, clock, truth log and routing
    configuration: SixFluxoniumSettings | str | Path,
    *,
    soccfg: QickConfig,
    workers: int = 1,
    clock: SimulationClock | None = None,
    truth_log: Path | None = None,
    history: RoutingHistory | None = None,
    families: Collection[str] | None = None,
) -> SimulationEnvironment:
    """Build seven DC FakeDevices, six independent models and a bound SoC.

    configuration is validated material or a YAML path relative to cwd.
    soccfg is the required caller-owned QICK compile profile built from independent
    board YAML. It must include the configured drive/readout channels; this builder
    does not infer board capabilities or channel counts from physical wiring. Startup
    applies each initial current once. No hardware or numerical warmup runs.
    The host owns devices after return and calls close_backend after work drains.
    YAML supplies the shared mux generator, six mux ADCs and separate pulse route.
    Each qubit's microwave generator binds those ADCs for transport admission;
    numeric routing keeps each control on its own circuit. Other IDs are unbound.
    Q1 seed selects the root RNG; pacing uses the maximum per-qubit time scale.
    The pulse ADC uses Q1 receiver calibration.
    clock optionally injects monotonic physical seconds; None creates one shared
    environment clock. Only the outer binding completes physical rounds.
    truth_log selects private JSONL output for enabled drift, defaulting to
    ~/.local/state/zcu-tools/simulation/six-fluxonium/<session>/truth.jsonl.
    Drift-off writes nothing. No truth enters public metadata.
    history is borrowed by all six model sources; None gives each its own history.
    clear() forces fresh solves without resetting sampling or clocks. families is
    the whitelist for every span, including projective continuations; None permits
    all families. Unknown names raise at build, unavailable domains at acquire.
    workers selects each cavity Engine's parallel width across timelines (default 1).
    Values below 1 raise ValueError; construction starts no worker pool.
    numerics.telemetry defaults to enabled and writes one bounded routing JSONL
    row per admitted acquire, including failures. directory=None selects project-root
    .sim_telemetry; relative paths resolve at construction. Writer failures warn
    once without changing results. Disabled policy suppresses writes.
    YAML, wiring, truth I/O and construction failures propagate with cleanup failures.
    """
    settings = (
        parse_six_fluxonium(configuration)
        if isinstance(configuration, SixFluxoniumSettings)
        else load_six_fluxonium(configuration)
    )
    devices = DeviceManager()
    source = None
    try:
        for dc in (
            settings.dc_source,
            *(q.dc_bias.source for q in settings.qubits.ordered()),
        ):
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
        source = SixFluxoniumSource(
            settings,
            devices,
            workers=workers,
            clock=sim_clock,
            truth_log=truth_log,
            history=history,
            families=families,
        )
        binding = SimulationBinding(
            source.snapshot,
            source.execute,
            backend_version="six-fluxonium-v1",
            seed=settings.qubits.Q1.numerics.seed,
            timing=SimulationTiming(
                time_scale=max(
                    q.numerics.time_scale for q in settings.qubits.ordered()
                ),
                clock=sim_clock,
            ),
            callbacks=SimulationCallbacks(
                on_begin=lambda: source.begin_acquire(seed=binding.root_seed),
                on_end=source.finish_acquire if source.telemetry is not None else None,
            ),
        )
        mux_channels = tuple(q.readout.adc.ro_ch for q in settings.qubits.ordered())
        routes = (
            *((settings.mux_gen_ch, ch) for ch in mux_channels),
            (settings.pulse_gen_ch, settings.pulse_ro_ch),
            *(
                (q.drive_line.gen_ch, ch)
                for q in settings.qubits.ordered()
                for ch in (*mux_channels, settings.pulse_ro_ch)
            ),
        )
        soc = SimulationSoc(soccfg, binding, routes=routes)
        return SimulationEnvironment(
            soc,
            soccfg,
            devices,
            "Six independent fluxonium cavities with six local DC sources and one coil",
            close_backend=source.close,
        )
    except Exception as error:
        errors = [error]
        if source is not None:
            try:
                source.close()
            except Exception as backend_error:
                logging.exception("Six-fluxonium backend cleanup failed")
                errors.append(backend_error)
        try:
            devices.close_all_devices()
        except Exception as device_error:
            logging.exception("Six-fluxonium device cleanup failed")
            errors.append(device_error)
        if len(errors) > 1:
            raise ExceptionGroup(
                "six-fluxonium startup and cleanup failed", errors
            ) from error
        raise
