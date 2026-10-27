"""Six single-model sources, seven DC currents and bounded ADC composition."""

import logging
from dataclasses import replace
from hashlib import sha256
from math import isfinite, prod
from pathlib import Path
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray

from zcu_tools.core.simulation import (
    AcquireRequest,
    CancellationQuery,
    ChannelSignals,
    EnvironmentSnapshot,
    RoundRequest,
    RoundSignals,
    SamplingIdentity,
)
from zcu_tools.device import DeviceManager, FakeDevice, FakeDeviceInfo
from zcu_tools.experiment.simulation import SimulationEnvironment
from zcu_tools.experiment.single_fluxonium import (
    SingleFluxoniumRuntimeOptions,
    SingleFluxoniumSource,
)
from zcu_tools.experiment.single_fluxonium.drift import CoherenceDrift
from zcu_tools.program.v2.mocksoc import make_mux_mock_soccfg
from zcu_tools.program.v2.simulation import (
    IntegratedRound,
    SimulationBinding,
    SimulationCallbacks,
    SimulationClock,
    SimulationSoc,
    SimulationTiming,
)
from zcu_tools.resources.single_fluxonium.schema import (
    DCBias,
    DCSource,
    SingleFluxoniumSettings,
)
from zcu_tools.resources.six_fluxonium import (
    SixFluxoniumSettings,
    load_six_fluxonium,
    parse_six_fluxonium,
)
from zcu_tools.resources.six_fluxonium.schema import CircuitBias
from zcu_tools.simulate.adc import (
    ChannelReadout,
    PreparedRawShots,
    ReadoutBatch,
    round_statistics,
    sample_traces,
)
from zcu_tools.simulate.evolution import Engine

from .results import CavityRound, combine_integrated
from .routing import RoutedGroup, route_request


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
    # Pulse ADC6 has Q1's receiver calibration; the selected cavity supplies
    # its physical resonator and incident-field calibration, never another ADC.
    adc = qubits[channel if channel < 6 else 0].readout.adc.model_copy(
        update={"ro_ch": channel}
    )
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
        drive_line=qubit.drive_line,
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
    settings.numerics provides environment-wide prepare/result/routing byte caps;
    each source receives one sixth, rounded down. ADC bytes apply per source.
    No joint Hilbert space or pulsed flux route is created. Call snapshot/execute
    only serially; close releases numeric resources but never caller devices.
    clock is an environment-owned monotonic clock in seconds, shared with every
    source and the outer binding; None creates one. truth_log selects private
    JSONL output for enabled qubits, or a fresh six-fluxonium session directory.
    begin_acquire freezes losses once for every point/round; execute also admits
    new standalone requests. Truth I/O failures propagate. No private replay is
    provided. The caller retains device and injected clock ownership.
    """

    def __init__(
        self,
        settings: SixFluxoniumSettings,
        devices: DeviceManager,
        *,
        clock: SimulationClock | None = None,
        truth_log: Path | None = None,
    ) -> None:
        self.settings = settings
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
                    index,
                    0,
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
                        workers=1,
                        cache_bytes=settings.numerics.prepare_cache_bytes // 6,
                        result_memo_bytes=settings.numerics.result_memo_bytes // 6,
                        routing_cache_bytes=settings.numerics.routing_cache_bytes // 6,
                    ),
                    readout_cache_bytes=settings.numerics.adc_cache_bytes,
                    clock=self._clock,
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
        self._request = None
        self._pending_acquire = True

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

    def _prepare_group(
        self,
        request: AcquireRequest,
        currents: dict[str, float],
        group: RoutedGroup,
        *,
        cancel: CancellationQuery | None = None,
    ) -> tuple[NDArray[np.float64], tuple[ChannelReadout, ...], PreparedRawShots]:
        source = self.sources[group.cavity]
        bias = self.settings.qubits.ordered()[group.cavity].dc_bias
        flux = _reduced_flux(
            bias, currents[self.settings.dc_source.name], currents[bias.source.name]
        )
        channels = []
        batches: dict[int, ReadoutBatch] = {}
        populations = None
        count = len(group.points)
        for channel in group.channels:
            source.settings = _single_settings(
                self._effective_settings, group.cavity, channel, group.gen_ch, flux
            )
            snapshot = EnvironmentSnapshot(
                "single-fluxonium-v1",
                source.identity,
                ((bias.source.name, currents[bias.source.name]),),
            )
            subrequest = replace(
                request,
                loop_dims=(request.reps, count),
                avg_level=0,
                readouts=tuple(w for w in request.readouts if w.channel == channel),
                point_dims=(count,),
                point_names=("routed_point",),
                templates=tuple(
                    replace(
                        template,
                        readouts=tuple(
                            w for w in template.readouts if w.channel == channel
                        ),
                    )
                    for template in group.templates
                ),
            )
            prepared = source.prepare(subrequest, snapshot, cancel=cancel)
            populations = prepared.populations
            channels.append(prepared.channel)
            batches.update(prepared.raw.batches)
        if populations is None:
            raise ValueError("cavity group requires an ADC")
        solver = source.settings.numerics.solver
        raw = PreparedRawShots(
            populations, batches, reps=request.reps, rtol=solver.rtol, atol=solver.atol
        )
        return populations, tuple(channels), raw

    def execute(self, round_request: RoundRequest) -> IntegratedRound | RoundSignals:
        """Evolve each observed independent circuit and return all requested ADCs.

        Routing errors fail before physical preparation. Each point starts in
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
        rounds = []
        traces: dict[int, NDArray[np.float64]] = {}
        for group in groups:
            populations, channels, raw = self._prepare_group(
                request, currents, group, cancel=round_request.cancel
            )
            # Cantor-pair the fixed-radix acquire/cavity identity with the
            # partition's first point, independent of request size. ADCs within
            # one group still share a single path draw.
            cavity_sequence = request.sequence * 6 + group.cavity
            first_point = int(group.points[0])
            diagonal = cavity_sequence + first_point
            sequence = diagonal * (diagonal + 1) // 2 + first_point
            if request.mode == "integrated":
                statistics = round_statistics(
                    populations,
                    channels,
                    reps=request.reps,
                    seed=request.root_seed,
                    acquire_sequence=sequence,
                    round_index=index,
                )
                rounds.append(
                    CavityRound(
                        group,
                        IntegratedRound(statistics, raw.bind(statistics.raw_identity)),
                    )
                )
            else:
                result = sample_traces(
                    populations,
                    dict(raw.batches),
                    reps=request.reps,
                    identity=SamplingIdentity(request.root_seed, (sequence, index, 0)),
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
        if request.mode == "integrated":
            return combine_integrated(
                request,
                tuple(rounds),
                tuple(
                    q.numerics.truncation.levels for q in self.settings.qubits.ordered()
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

    def close(self) -> None:
        """Release six numeric backends and reject later work, leaving DC untouched.

        Idempotent. Call after acquisitions drain. Delivered raw sources retain
        their detached physical inputs and remain usable after close.
        """
        for source in self.sources:
            source.close()
        self._closed = True


def six_fluxonium(
    configuration: SixFluxoniumSettings | str | Path,
    *,
    clock: SimulationClock | None = None,
    truth_log: Path | None = None,
) -> SimulationEnvironment:
    """Build seven DC FakeDevices, six independent models and a bound SoC.

    configuration is validated material or a YAML path relative to cwd. Startup
    applies each initial current once. No hardware or numerical warmup runs.
    The host owns devices after return and calls close_backend after work drains.
    Mux gen0 binds ADC0..5; pulse gen13 binds ADC6. Microwave gens1..6 bind those
    ADCs for transport admission; numeric routing keeps each control on its own
    circuit. Holes7..12 are unbound. Q1 seed selects the root RNG; pacing uses the
    maximum per-qubit floor. Pulse ADC6 uses Q1 receiver calibration.
    clock optionally injects monotonic physical seconds; None creates one shared
    environment clock. Only the outer binding completes physical rounds.
    truth_log selects private JSONL output for enabled drift, defaulting to
    ~/.local/state/zcu-tools/simulation/six-fluxonium/<session>/truth.jsonl.
    Drift-off writes nothing. No truth enters public metadata.
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
            settings, devices, clock=sim_clock, truth_log=truth_log
        )
        soccfg = make_mux_mock_soccfg(14, 7, gen_ch=0, ro_chs=tuple(range(6)))
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
                on_begin=lambda: source.begin_acquire(seed=binding.root_seed)
            ),
        )
        routes = (
            *((0, ch) for ch in range(6)),
            (13, 6),
            *((gen, ch) for gen in range(1, 7) for ch in range(7)),
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
