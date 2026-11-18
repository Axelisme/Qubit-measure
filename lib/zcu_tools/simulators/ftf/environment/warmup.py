"""Small disposable numeric workloads covering common FTF compilation families.

Synthetic literals serve JIT compilation only; their outputs must not enter results.
This manifest is deliberately finite, not a promise to compile every route or
specialization. Update inputs and remeasure compile events when routing changes.
No YAML, device, acquisition binding, sampling or global RNG is involved.
"""

from collections.abc import Callable, Collection
from contextlib import closing
from dataclasses import replace
from functools import partial
from importlib.util import find_spec
from threading import Event, Lock
from types import TracebackType
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from zcu_tools.experiment.warmup import EnvironmentWarmup
from zcu_tools.program.v2.simulation.signals import StatisticsCopier
from zcu_tools.simulate.adc import (
    ConditionalReadout,
    PiecewiseLinearEnvelope,
    ReadoutBatch,
    ReadoutEvent,
    ReadoutModel,
    ReadoutPulse,
    Receiver,
    SampleGrid,
)
from zcu_tools.simulate.boundary import (
    AcquireRequest,
    ChannelStatistics,
    ReadoutWindow,
    RoundStatistics,
    SamplingIdentity,
    TimeBase,
)
from zcu_tools.simulate.evolution import (
    DriveTrack,
    Engine,
    EnvelopeSpec,
    FluxSchedule,
    ModelSource,
    SecularDrive,
    SecularModel,
    TimelineBatch,
)

# Each name is both a progress item and a coverage-manifest entry. Workloads use
# public numeric seams so private kernel layouts remain owned by evolution/ADC.
COVERAGE = (
    "shaped_carrier_family",
    "shaped_direct",
    "shaped_complex",
    "multitone_rwa",
    "multitone_constant",
    "multitone_direct",
    "multitone_gate",
    "single_tone_gate",
    "single_tone_gate_adaptive",
    "single_tone_gate_retained",
    "single_tone_rwa",
    "floquet_probe_windows",
    "dyson_gain_batch",
    "floquet_period",
    "floquet_low_carrier",
    "floquet_cf4",
    "t4",
    "readout_integrated",
    "readout_flat_batch",
    "readout_decimated",
    "readout_mux",
    "readout_sampled",
    "statistics_copy",
)

_CPU_ONLY = "cpu-only"
_lock = Lock()
_startup: dict[int | Literal["cpu-only"], EnvironmentWarmup] = {}
_compile_lock = Lock()
_cpu_prepared: set[str] = set()
_cpu_failures: dict[str, tuple[Exception, TracebackType | None]] = {}


def startup_warmup(
    dimension: int, *, families: Collection[str] | None
) -> EnvironmentWarmup:
    """Return lazy preparation shared by FTF bundles with matching startup work.

    dimension is the number of retained states (at least one, else ValueError).
    When pyopencl is installed and families permits dop853 (None permits all),
    a GPU item follows the CPU manifest. Completion means device detection
    finished and the program was built if the device and dimension were usable.
    An error after GPU detection succeeds fails the item. A CPU failure stops
    the manifest before the GPU item. The caller starts the returned controller
    after environment construction. CPU items and each GPU dimension run at
    most once per process.
    """
    if dimension < 1:
        raise ValueError("dimension must be at least one")
    prepare_gpu = (families is None or "dop853" in families) and find_spec(
        "pyopencl"
    ) is not None
    key = dimension if prepare_gpu else _CPU_ONLY
    with _lock:
        if key not in _startup:
            items: list[tuple[str, Callable[[], None]]] = [
                (name, partial(_prepare_cpu_once, name)) for name in COVERAGE
            ]
            if prepare_gpu:
                items.append(("dop853_opencl", partial(_prepare_opencl, dimension)))
            _startup[key] = EnvironmentWarmup(tuple(items))
        return _startup[key]


def _prepare_cpu_once(name: str) -> None:
    # Shared across dimension-specific controllers. A separate lock keeps
    # controller lookup free while a compiler is running.
    with _compile_lock:
        if name in _cpu_prepared:
            return
        if name in _cpu_failures:
            error, traceback = _cpu_failures[name]
            raise error.with_traceback(traceback)
        try:
            _prepare(name)
        except Exception as error:
            # Keep the original frames instead of accumulating each re-raise.
            _cpu_failures[name] = (error, error.__traceback__)
            raise
        _cpu_prepared.add(name)


def _prepare_opencl(dimension: int) -> None:
    # Keep the device probe single-flight even for different model dimensions.
    with _compile_lock:
        _warm_opencl(dimension)


def _warm_opencl(dimension: int) -> None:
    with closing(Engine(1, families=("dop853",))) as engine:
        if engine.detect_execution_device() == "cpu":
            # The live Engine will report why detection failed on eligible work.
            return
        with closing(_opencl_source(dimension)) as source:
            result = engine.evolve(source, (_opencl_batch(),), telemetry=True)
        routing = result.paths.routing["dop853"]
        if routing.opencl_detection_error is not None:
            raise RuntimeError(routing.opencl_detection_error)
        # No OpenCL rows without an error means auto did not admit this workload.


def _opencl_source(dimension: int) -> ModelSource:
    ladder = np.eye(dimension, k=1, dtype=complex)
    drive = ladder + ladder.T
    zeros = np.zeros((dimension, dimension), dtype=np.float64)
    quadrature = np.zeros_like(drive)
    model = SecularModel(
        (0,),
        "ftf-startup-opencl",
        np.linspace(0.0, 0.13 * (dimension - 1), dimension),
        np.eye(dimension, dtype=complex),
        (
            SecularDrive("drive", drive, quadrature),
            SecularDrive("aux", drive, quadrature),
        ),
        zeros,
        zeros,
    )
    density = np.zeros_like(drive)
    density[0, 0] = 1
    return ModelSource(density, lambda _: model)


def _opencl_batch() -> TimelineBatch:
    # Independent, distinct endpoints amortize dispatch even at small d.
    # The saved program depends on d, not on these synthetic controls.
    duration = 120.0
    times = np.linspace(0, duration, 129)
    envelope = EnvelopeSpec(
        "sampled", np.sin(np.pi * times / duration).astype(complex), times
    )
    gains = np.linspace(0.05, 0.15, 16)
    batch = _batch(np.full(len(gains), 0.13), gains, envelope, duration)
    (first,) = batch.tracks
    second = replace(first, port="aux", frequency_ghz=np.full(len(gains), 0.17))
    return replace(batch, tracks=(first, second), end_ns=first.duration_ns)


def _source() -> ModelSource:
    x = np.array([[0, 0.012, 0.002], [0.012, 0, 0.009], [0.002, 0.009, 0]], complex)
    model = SecularModel(
        (0,),
        "ftf-startup-numeric",
        np.array([0.0, 5.0, 7.0]),
        np.eye(3, dtype=complex),
        (SecularDrive("drive", x, np.zeros_like(x)),),
        np.array([[0, 1e-5, 1e-6], [0, 0, 1e-5], [0, 0, 0]]),
        np.zeros((3, 3)),
    )
    return ModelSource(np.diag([0.8, 0.15, 0.05]).astype(complex), lambda _: model)


def _batch(
    frequencies: NDArray[np.float64],
    gains: NDArray[np.float64],
    envelope: EnvelopeSpec,
    duration: float,
) -> TimelineBatch:
    count = len(frequencies)
    track = DriveTrack(
        "drive",
        np.zeros(count),
        np.full(count, duration),
        frequencies,
        np.full(count, 0.37),
        gains,
        (envelope,),
        np.zeros(count, dtype=np.int64),
        np.ones(count, dtype=bool),
    )
    return TimelineBatch(
        np.arange(count),
        (track,),
        FluxSchedule(
            np.arange(count + 1), np.zeros(count), np.zeros((count, 1), dtype=np.int64)
        ),
        np.full(count, duration + 2),
    )


def _prepare(name: str) -> None:
    if name == "statistics_copy":
        _statistics_copy()
        return
    if name.startswith("readout_"):
        _readout(name)
        return
    if name == "floquet_probe_windows":
        _floquet_probe_windows()
        return
    # Checked before the generic multitone prefix, which would also match it.
    if name == "multitone_gate":
        _joint_gate()
        return
    if name.startswith("multitone_"):
        _multitone(name)
        return
    _single_track(name)


def _single_track(name: str) -> None:
    duration = 20.3
    envelope = EnvelopeSpec("constant")
    frequencies = np.array([5.0])
    gains = np.array([0.2])
    if name == "shaped_carrier_family":
        frequencies = np.linspace(4.99, 5.01, 11)
        gains = np.full(11, 0.2)
        envelope = EnvelopeSpec(
            "sampled", np.array([0.0, 1.0], complex), np.array([0.0, duration])
        )
    elif name == "single_tone_rwa":
        duration = 200.0
        # Keep the ordinary window and restore an off-window edge as well.
        # Nonempty retained graphs pass a bool4 mask to the coefficient builder,
        # a distinct compiled signature from the default-window None variant.
        frequencies = np.array([5.0, 4.89])
        gains = np.array([0.5, 1.0])
    elif name in (
        "single_tone_gate",
        "single_tone_gate_adaptive",
        "single_tone_gate_retained",
    ):
        duration = 128.0
        times = np.linspace(0.0, duration, 129)
        envelope = EnvelopeSpec(
            "sampled", np.exp(-0.5 * ((times - 64) / 32) ** 2).astype(complex), times
        )
        # Restored masks select the bool4 coefficient specialization, not
        # the default-window None variant of ordinary single-tone gates.
        frequency, gain = {"single_tone_gate_retained": (4.89, 1.0)}.get(
            name, (5.0, 0.08)
        )
        frequencies = np.full(1024, frequency)
        gains = np.full(1024, gain)
    elif name == "shaped_direct":
        envelope = EnvelopeSpec(
            "sampled",
            np.array([0.0, 1.0j, 0.01], complex),
            np.array([0.0, 0.02, duration]),
        )
    elif name == "shaped_complex":
        envelope = EnvelopeSpec(
            "sampled",
            np.array([0.0, 1.0 + 0.1j], complex),
            np.array([0.0, duration]),
        )
    elif name in ("floquet_low_carrier", "floquet_cf4"):
        frequencies = np.array([0.05])
        duration = 100.3
    elif name == "dyson_gain_batch":
        frequencies = np.full(12, 0.01)
        gains = np.linspace(0.00001, 0.0001, 12)
        duration = 300.0
    batch = _batch(frequencies, gains, envelope, duration)
    if name == "t4":
        batch = TimelineBatch(batch.points, (), batch.flux, batch.end_ns)
    elif name == "dyson_gain_batch":
        # Distinct driven prefixes prevent shared-input binding. Both bound and
        # ordinary gain-batch paths must compile (their array ownership differs).
        prefix = replace(batch.tracks[0], duration_ns=np.full(len(gains), 2.0))
        probe = replace(batch.tracks[0], start_ns=np.full(len(gains), 4.0))
        batch = replace(
            batch, tracks=(prefix, probe), end_ns=np.full(len(gains), duration + 6)
        )
    source = _source()
    if name == "single_tone_gate_adaptive":
        model = source.model((0,))
        model = replace(model, rates_per_ns=model.rates_per_ns * 20_000)
        source = ModelSource(source.initial_density, lambda _: model)
    if name in ("floquet_low_carrier", "floquet_cf4"):
        # A remote spectator makes spectral construction cheaper than CF4;
        # the narrow spectrum independently exercises the CF4 builder.
        spectator = 0.13 if name == "floquet_cf4" else 7.0
        model = replace(
            source.model((0,)), energies_ghz=np.array([0.0, 0.05, spectator])
        )
        source = ModelSource(source.initial_density, lambda _: model)
    # Tight budgets keep the period workloads on their original exact families.
    _evolve(source, batch, budget=1e-4 if name.startswith("floquet_") else 1e-3)


def _multitone(name: str) -> None:
    duration = 128.0
    times = np.linspace(0.0, duration, 129)
    envelope = EnvelopeSpec(
        "sampled", np.exp(-(((times - 64) / 32) ** 2) / 2).astype(complex), times
    )
    if name == "multitone_constant":
        envelope = EnvelopeSpec("constant")
    elif name == "multitone_direct":
        envelope = EnvelopeSpec(
            "sampled",
            np.array([0.0, 1.0j, 0.01], complex),
            np.array([0.0, 0.02, duration]),
        )
    source = _source()
    model = source.model((0,))
    auxiliary = SecularDrive(
        "aux", model.drives[0].in_phase_ghz, model.drives[0].quadrature_ghz
    )
    model = replace(model, drives=(*model.drives, auxiliary))
    source = ModelSource(source.initial_density, lambda _: model)
    batch = _batch(np.array([5.0]), np.array([0.2]), envelope, duration)
    batch = replace(
        batch,
        tracks=(
            batch.tracks[0],
            replace(batch.tracks[0], port="aux", frequency_ghz=np.array([7.0])),
        ),
    )
    _evolve(source, batch)


def _evolve(source: ModelSource, batch: TimelineBatch, *, budget: float = 1e-3) -> None:
    engine = Engine(1, backend="cpu")
    try:
        engine.evolve(source, (batch,), budget=budget, cancel=Event())
    finally:
        engine.close()


def _floquet_probe_windows() -> None:
    # A small Stark/probe scan exercises Fourier certificates, including new
    # probe bins on a retained frame. Rejected guard products stay disposable.
    x = np.array([[0.0, 0.02], [0.02, 0.0]], complex)
    model = SecularModel(
        (0,),
        "ftf-startup-floquet",
        np.array([0.0, 0.13]),
        np.eye(2, dtype=complex),
        (
            SecularDrive("drive", x, np.zeros_like(x)),
            SecularDrive("probe", x, np.zeros_like(x)),
        ),
        np.array([[0.0, 0.001], [0.0, 0.0]]),
        np.zeros((2, 2)),
    )
    source = ModelSource(np.diag([0.8, 0.2]).astype(complex), lambda _: model)
    engine = Engine(1, backend="cpu")
    try:
        for shift in (0.0, 0.004):
            batch = _batch(
                np.linspace(0.106, 0.114, 65) + shift,
                np.full(65, 0.0001),
                EnvelopeSpec("constant"),
                16.0,
            )
            stark = replace(
                batch.tracks[0],
                frequency_ghz=np.full(65, 0.13),
                amplitude=np.full(65, 0.2),
            )
            engine.evolve(
                source,
                (
                    replace(
                        batch, tracks=(stark, replace(batch.tracks[0], port="probe"))
                    ),
                ),
                cancel=Event(),
            )
    finally:
        engine.close()


def _readout(name: str) -> None:
    model = ReadoutModel(
        identity="ftf-startup-readout",
        state_labels=("g", "e", "f"),
        frequencies_ghz=np.array([[6.001, 6.013], [6.002, 6.014], [6.003, 6.015]]),
        reference_frequency_ghz=6.0,
        kappa_internal_per_ns=np.array([0.03, 0.05]),
        kappa_external_per_ns=np.array([0.08, 0.06]),
        rates_per_ns=np.array([[0, 1e-5, 1e-6], [0, 0, 1e-5], [0, 0, 0]]),
        receiver=Receiver(1.0, 0.0, np.zeros(2), np.zeros((2, 2))),
    )
    pulses = (ReadoutPulse(0, 20, 0.1 + 0.02j, 6.0),)
    if name == "readout_mux":
        pulses += (ReadoutPulse(0, 20, 0.05j, 6.01),)
    elif name == "readout_sampled":
        pulses = (
            ReadoutPulse(
                0,
                20,
                0.1 + 0.02j,
                6.0,
                envelope=PiecewiseLinearEnvelope(
                    np.array([0.0, 5.0, 15.0, 20.0]),
                    np.array([0.0, 1.0, 1.0, 0.0], complex),
                ),
            ),
        )
    contents = (pulses,)
    if name == "readout_flat_batch":
        contents += ((replace(pulses[0], frequency_ghz=6.01),),)
    grids = ((SampleGrid(np.arange(8, dtype=float) * 4, 6.0),),) * len(contents)
    response = ConditionalReadout(model)
    for traces in (
        (False, True)
        if name in ("readout_mux", "readout_sampled")
        else (name == "readout_decimated",)
    ):
        response.responses(
            ReadoutBatch(
                model,
                tuple(
                    (ReadoutEvent(pulses, grids),)
                    for pulses, grids in zip(contents, grids, strict=True)
                ),
                np.arange(len(contents), dtype=np.int64),
                include_traces=traces,
            )
        )


def _statistics_copy() -> None:
    request = AcquireRequest(
        "integrated",
        (2, 4),
        1,
        (ReadoutWindow(0, 0, 8, 1, (0,)),),
        0,
        0,
        "startup",
        time_base=TimeBase.from_clocks((100.0,)),
        reps=4,
    )
    counts = np.array([[1, 3], [2, 2]], dtype=np.int64)
    sums = np.zeros((2, 1, 2), dtype=np.float64)
    result = RoundStatistics(
        counts,
        (
            ChannelStatistics(
                0,
                sums,
                np.zeros((2, 1, 2, 2), dtype=np.float64),
                np.eye(2, dtype=np.float64),
            ),
        ),
        SamplingIdentity(0, (0, 0, 1)),
    )
    copier = StatisticsCopier()
    # Borrowed results can be writable or readonly; Numba specializes on both.
    for writable in (True, False):
        counts.setflags(write=writable)
        sums.setflags(write=writable)
        copier.copy(request, result)


def _joint_gate() -> None:
    duration = 128.0
    envelope = EnvelopeSpec(
        "sampled",
        np.array([0.5, 1.0, 0.5], dtype=complex),
        np.array([0.0, 64.0, duration]),
    )
    batch = _batch(np.full(24, 5.0), np.full(24, 0.01), envelope, duration)
    companion = replace(
        batch.tracks[0], frequency_ghz=np.full(24, 2.0), phase_rad=np.full(24, 0.71)
    )
    engine = Engine(1, backend="cpu")
    try:
        engine.evolve(
            _source(),
            (replace(batch, tracks=(*batch.tracks, companion)),),
            cancel=Event(),
        )
    finally:
        engine.close()
