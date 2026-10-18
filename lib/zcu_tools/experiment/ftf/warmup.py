"""Small disposable numeric workloads covering common FTF compilation families.

This manifest is deliberately finite, not a promise to compile every route or
specialization. Update inputs and remeasure compile events when routing changes.
No YAML, device, acquisition binding, sampling or global RNG is involved.
"""

from dataclasses import replace
from functools import partial
from threading import Event, Lock

import numpy as np
from numpy.typing import NDArray

from zcu_tools.experiment.warmup import EnvironmentWarmup
from zcu_tools.simulate.adc import (
    ConditionalReadout,
    PiecewiseLinearEnvelope,
    ReadoutBatch,
    ReadoutModel,
    ReadoutPulse,
    Receiver,
    SampleGrid,
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
    "multitone_direct",
    "multitone_gate",
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
)

_lock = Lock()
_startup: EnvironmentWarmup | None = None


def startup_warmup() -> EnvironmentWarmup:
    """Return process-wide single-flight preparation shared by all FTF bundles.

    Creation is lazy; the caller starts it after building its environment. All
    work uses one numeric worker independent of production acquire parallelism.
    Completed/failed results are retained for the process lifetime; no retries
    or duplicate workers are created by opening another FTF environment.
    """
    global _startup
    with _lock:
        if _startup is None:
            items = tuple((name, partial(_prepare, name)) for name in COVERAGE)
            _startup = EnvironmentWarmup(items)
        return _startup


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
    if name in ("floquet_low_carrier", "floquet_cf4"):
        # A remote spectator makes spectral construction cheaper than CF4;
        # the narrow spectrum independently exercises the CF4 builder.
        spectator = 0.13 if name == "floquet_cf4" else 7.0
        model = replace(
            source.model((0,)), energies_ghz=np.array([0.0, 0.05, spectator])
        )
        source = ModelSource(source.initial_density, lambda _: model)
    _evolve(source, batch)


def _multitone(name: str) -> None:
    duration = 128.0
    times = np.linspace(0.0, duration, 129)
    envelope = EnvelopeSpec(
        "sampled", np.exp(-(((times - 64) / 32) ** 2) / 2).astype(complex), times
    )
    if name == "multitone_direct":
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


def _evolve(source: ModelSource, batch: TimelineBatch) -> None:
    engine = Engine(1, 0)
    try:
        engine.evolve(source, (batch,), cancel=Event())
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
    engine = Engine(1, 8_000_000)
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
                contents,
                grids,
                np.arange(len(contents), dtype=np.int64),
                include_traces=traces,
            )
        )


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
    engine = Engine(1, 0)
    try:
        engine.evolve(
            _source(),
            (replace(batch, tracks=(*batch.tracks, companion)),),
            cancel=Event(),
        )
    finally:
        engine.close()
