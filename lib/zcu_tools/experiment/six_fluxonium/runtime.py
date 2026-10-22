"""Six single-model sources, one global current and bounded ADC composition."""

import logging
from dataclasses import replace
from hashlib import sha256
from math import prod
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from zcu_tools.core.simulation import (
    AcquireRequest,
    ChannelSignals,
    EnvironmentSnapshot,
    RoundRequest,
    RoundSignals,
    SamplingIdentity,
)
from zcu_tools.device import DeviceManager, FakeDevice, FakeDeviceInfo
from zcu_tools.experiment.simulation import SimulationEnvironment
from zcu_tools.experiment.single_fluxonium import SingleFluxoniumSource
from zcu_tools.program.v2.mocksoc import make_mux_mock_soccfg
from zcu_tools.program.v2.simulation import (
    IntegratedRound,
    SimulationBinding,
    SimulationSoc,
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


def _single_settings(
    settings: SixFluxoniumSettings, cavity: int, channel: int, gen: int
) -> SingleFluxoniumSettings:
    qubits, dc = settings.qubits.ordered(), settings.dc_source
    qubit = qubits[cavity]
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
            offset_phi0=qubit.dc_bias.offset_phi0,
            source=DCSource(
                name=dc.name,
                unit=dc.unit,
                limits=dc.limits,
                initial=dc.initial,
                ramp_step=dc.ramp_step,
                flux_per_unit=qubit.dc_bias.flux_per_unit,
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
        numerics=qubit.numerics,
    )


class SixFluxoniumSource:
    """Serialized six-model backend with one shared live DC driver.

    settings: validated six-system material. devices: manager retaining the
    addressed global FakeDevice. Six SingleFluxoniumSources retain that exact
    driver, independent oscillator/evolution/ADC caches and physical parameters.
    settings.numerics provides environment-wide prepare/result/routing byte caps;
    each source receives one sixth, rounded down. ADC bytes apply per source.
    No joint Hilbert space or pulsed flux route is created. Call snapshot/execute
    only serially; close releases numeric resources but never caller devices.
    """

    def __init__(self, settings: SixFluxoniumSettings, devices: DeviceManager) -> None:
        self.settings = settings
        self.sources = tuple(
            SingleFluxoniumSource(
                _single_settings(settings, index, index, 0),
                devices,
                readout_kind="mux",
                engine=Engine(
                    workers=1,
                    cache_bytes=settings.numerics.prepare_cache_bytes // 6,
                    result_memo_bytes=settings.numerics.result_memo_bytes // 6,
                    routing_cache_bytes=settings.numerics.routing_cache_bytes // 6,
                ),
                readout_cache_bytes=settings.numerics.adc_cache_bytes,
            )
            for index in range(6)
        )
        self._closed = False
        self._identity = sha256(settings.model_dump_json().encode()).hexdigest()

    def snapshot(self) -> EnvironmentSnapshot:
        """Capture the one live A value with full settings identity, without reset.

        Invalid current/unit or a missing driver raises ValueError. A closed
        source raises RuntimeError. Each qubit converts this current using its
        own offset_phi0 and flux_per_unit during physical preparation.
        """
        if self._closed:
            raise RuntimeError("six-fluxonium source is closed")
        captured = self.sources[0].snapshot()
        return EnvironmentSnapshot("six-fluxonium-v1", self._identity, captured.values)

    def _prepare_group(
        self, request: AcquireRequest, snapshot: EnvironmentSnapshot, group: RoutedGroup
    ) -> tuple[NDArray[np.float64], tuple[ChannelReadout, ...], PreparedRawShots]:
        source = self.sources[group.cavity]
        channels = []
        batches: dict[int, ReadoutBatch] = {}
        populations = None
        count = len(group.points)
        for channel in group.channels:
            source.settings = _single_settings(
                self.settings, group.cavity, channel, group.gen_ch
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
            prepared = source.prepare(subrequest, snapshot)
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
        Numeric/ADC failures propagate without partial results.
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
        groups = route_request(self.settings, request)
        rounds = []
        traces: dict[int, NDArray[np.float64]] = {}
        for group in groups:
            populations, channels, raw = self._prepare_group(request, snapshot, group)
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
) -> SimulationEnvironment:
    """Build one global DC FakeDevice, six independent models and a bound SoC.

    configuration is validated material or a YAML path relative to cwd. Startup
    applies global initial current once. No hardware or numerical warmup runs.
    The host owns devices after return and calls close_backend after work drains.
    Mux gen0 binds ADC0..5; pulse gen13 binds ADC6. Microwave gens1..6 bind those
    ADCs for transport admission; numeric routing keeps each control on its own
    circuit. Holes7..12 are unbound. Q1 seed selects the root RNG; pacing uses the
    maximum per-qubit floor. Pulse ADC6 uses Q1 receiver calibration.
    YAML, wiring and construction failures propagate with cleanup failures.
    """
    settings = (
        parse_six_fluxonium(configuration)
        if isinstance(configuration, SixFluxoniumSettings)
        else load_six_fluxonium(configuration)
    )
    devices = DeviceManager()
    source = None
    try:
        dc = settings.dc_source
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
        source = SixFluxoniumSource(settings, devices)
        soccfg = make_mux_mock_soccfg(14, 7, gen_ch=0, ro_chs=tuple(range(6)))
        binding = SimulationBinding(
            source.snapshot,
            source.execute,
            backend_version="six-fluxonium-v1",
            seed=settings.qubits.Q1.numerics.seed,
            min_round_duration_s=max(
                q.numerics.min_round_duration_s for q in settings.qubits.ordered()
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
            "Six independent fluxonium cavities with one global DC source",
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
