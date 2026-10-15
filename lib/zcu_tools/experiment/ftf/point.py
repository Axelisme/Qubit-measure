"""Explicit QICK-to-numeric FTF composition and author diagnostics."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from zcu_tools.core.simulation import EnvironmentSnapshot, PointTimeline, PulseEvent
from zcu_tools.resources.ftf import FTFSettings
from zcu_tools.simulate.adc import ReadoutModel, ReadoutPulse, Receiver, SampleGrid
from zcu_tools.simulate.evolution import (
    Basis,
    ControlPulse,
    ControlSequence,
    DriveSource,
    Envelope,
    NumericModel,
)
from zcu_tools.simulate.ftf import JointModel, Resonator
from zcu_tools.simulate.lines import FilteredFlux, FluxLine

from .numeric import joint_parameters
from .numeric import resonators as numeric_resonators


def _envelope(event: PulseEvent) -> Envelope:
    if event.envelope is None:
        return Envelope.flat(event.gain)
    times = np.array(event.envelope.sample_us) * 1000
    values = np.array(event.envelope.values, dtype=complex) * event.gain
    # Program owns linear playback with the final sample held through pulse end.
    return Envelope(lambda t: complex(np.interp(t, times, values)))


def _control(event: PulseEvent, name: str) -> ControlPulse:
    start = event.start_us * 1000
    frequency = event.frequency_mhz / 1000
    phase = np.deg2rad(event.phase_deg) + 2 * np.pi * frequency * (
        start - event.phase_reference_us * 1000
    )
    breakpoints = (
        ()
        if event.envelope is None
        else tuple(
            start + t * 1000
            for t in event.envelope.sample_us
            if 0 < t < event.duration_us
        )
    )
    return ControlPulse(
        name,
        start,
        start + event.duration_us * 1000,
        _envelope(event),
        frequency,
        float(phase),
        breakpoints_ns=breakpoints,
    )


def _dc_flux(
    settings: FTFSettings, snapshot: EnvironmentSnapshot
) -> NDArray[np.float64]:
    values = dict(snapshot.values)
    result = np.array(settings.dc_mapping.offset)
    for source in settings.devices:
        if (
            source.name not in values
            or not np.isfinite(values[source.name])
            or not source.limits[0] <= values[source.name] <= source.limits[1]
        ):
            raise ValueError(
                f"devices.{source.name}: missing/nonfinite/out-of-limits round value"
            )
    for column in settings.dc_mapping.columns:
        result += np.array(column.coefficients) * values[column.source]
    return result


def _controls(settings: FTFSettings, timeline: PointTimeline) -> ControlSequence:
    wiring = {
        source.gen_ch: source.name
        for source in (*settings.flux_sources, *settings.microwave_sources)
    }
    pulses = []
    for event in timeline.events:
        if event.gen_ch == settings.readout_source.gen_ch:
            continue
        if event.gen_ch not in wiring:
            raise ValueError(f"unbound FTF control generator {event.gen_ch}")
        pulses.append(_control(event, wiring[event.gen_ch]))
    return ControlSequence(tuple(pulses))


def _sources(settings: FTFSettings, joint: JointModel) -> tuple[DriveSource, ...]:
    zero = np.zeros_like(joint.hamiltonian(joint.dc_flux))
    sources = [DriveSource(source.name, zero) for source in settings.flux_sources]
    for source in settings.microwave_sources:
        coefficients = tuple(
            complex(real, imag) for real, imag in source.coefficients_ghz_per_gain
        )
        # Re[c*z] = Re(c)*Re(z)-Im(c)*Im(z); n itself is Hermitian.
        x = sum(
            (c.real * joint.charge_operator(i) for i, c in enumerate(coefficients)),
            zero.copy(),
        )
        y = sum(
            (-c.imag * joint.charge_operator(i) for i, c in enumerate(coefficients)),
            zero.copy(),
        )
        sources.append(DriveSource(source.name, x, y))
    return tuple(sources)


def _readout_event(settings: FTFSettings, timeline: PointTimeline) -> PulseEvent:
    pulses = [e for e in timeline.events if e.gen_ch == settings.readout_source.gen_ch]
    if len(pulses) != 1:
        raise ValueError("FTF supports one common terminal readout pulse period")
    pulse = pulses[0]
    if any(
        e.start_us + e.duration_us > pulse.start_us
        for e in timeline.events
        if e.gen_ch != pulse.gen_ch
    ):
        raise ValueError("active control during/after readout is unsupported")
    if (
        len(timeline.readouts) != 1
        or timeline.readouts[0].channel != settings.receiver.ro_ch
    ):
        raise ValueError("FTF supports its single bound receiver only")
    window = timeline.readouts[0]
    if (
        not window.reads
        or len(window.frequency_mhz) != window.reads
        or len(window.phase_deg) != window.reads
    ):
        raise ValueError("FTF requires per-execution ADC timing/demodulation")
    if any(ch != pulse.gen_ch for ch in window.generator_channels):
        raise ValueError("FTF read uses an unbound readout generator")
    return pulse


@dataclass(frozen=True)
class FTFPointPlan:
    """Author-only numeric composition; ordinary acquisition publishes I/Q only.

    joint: fixed round-DC circuit; evolution_model/controls: continuous numeric
    microwave/flux model and ns/GHz/radian controls. flux: actual causal source
    response before readout. readout_start_ns: common pulse origin. readout_flux:
    actual flux at that instant, including passive tail; readout_basis/model use
    that fixed value for the entire window (tail changes within it are ignored).
    resonators: two numeric line resonators. pulses: finite incident field.
    grids: per-read physical sample times and demodulation. receiver PSD converts
    to sample covariance using each ADC dt; all reads have this receiver's rate.
    Fields are private author diagnostics, not normal Run/MCP metadata.
    """

    joint: JointModel
    evolution_model: NumericModel
    controls: ControlSequence
    flux: FilteredFlux
    readout_start_ns: float
    readout_flux: NDArray[np.float64]
    readout_basis: Basis
    readout_model: ReadoutModel
    resonators: tuple[Resonator, Resonator]
    pulses: tuple[ReadoutPulse, ...]
    grids: tuple[SampleGrid, ...]

    @property
    def grid(self) -> SampleGrid:
        """Return the first read's ns/GHz/radian sample grid; plan has >=1 read."""
        return self.grids[0]


def plan_point(
    settings: FTFSettings, snapshot: EnvironmentSnapshot, timeline: PointTimeline
) -> FTFPointPlan:
    """Compose one detached execution into independent numeric tool inputs.

    settings is a validated immutable FTF resource snapshot, snapshot contains
    this round's actual device values, timeline supplies this execution's QICK
    us/MHz/degrees, including per-read ADC association/demodulation. Explicitly
    converts once to ns/GHz/radians and PSD/dt sample noise covariance. Builds
    fixed-DC joint V then Hro at the actual filtered flux at readout start.
    Only active gate/bias instructions during/after readout are rejected; passive
    tails are allowed and frozen at readout origin without settling thresholds.
    No evolution or ADC solve occurs here. Invalid/unsupported input raises.
    """
    pulse = _readout_event(settings, timeline)
    controls = _controls(settings, timeline)
    dc = _dc_flux(settings, snapshot)
    joint = JointModel(
        joint_parameters(settings), dc, temperature_k=settings.model.temperature_k
    )
    lines = tuple(
        FluxLine(s.name, s.coefficients_phi0_per_gain, s.delay_ns, s.tau_ns)
        for s in settings.flux_sources
    )
    flux = FilteredFlux(
        dc, controls, lines, rtol=settings.solver.rtol, atol=settings.solver.atol
    )
    start = pulse.start_us * 1000
    actual = flux.at(start)
    resonators = numeric_resonators(settings)
    model = joint.numeric_model(
        flux.at,
        _sources(settings, joint),
        end_ns=start,
        bath_segment_ns=settings.solver.bath_segment_ns,
        resonators=resonators,
    )
    _, vectors = joint.eigensystem(actual)
    basis = Basis("ftf-readout", joint.basis.labels, vectors)
    window = timeline.readouts[0]
    dt = 1000 / window.sample_rate_mhz
    receiver = Receiver(
        settings.receiver.gain,
        float(np.deg2rad(settings.receiver.phase_deg)),
        np.array(settings.receiver.offset_adc),
        np.array(settings.receiver.noise_psd_adc2_ns) / dt,
    )
    ro_model = ReadoutModel(
        "ftf-shared-line-v1",
        basis.labels,
        joint.readout_frequencies_ghz(actual, resonators),
        pulse.frequency_mhz / 1000,
        np.array([r.kappa_internal_per_ns for r in resonators]),
        np.array([r.kappa_external_per_ns for r in resonators]),
        joint.rates(actual),
        receiver,
    )
    envelope = _envelope(pulse)
    field_pulse = ReadoutPulse(
        start,
        start + pulse.duration_us * 1000,
        settings.readout_source.input_per_gain,
        pulse.frequency_mhz / 1000,
        float(np.deg2rad(pulse.phase_deg)),
        envelope.at,
    )
    grids = tuple(
        SampleGrid(
            np.asarray(
                trigger * 1000 + dt * np.arange(window.samples), dtype=np.float64
            ),
            frequency / 1000,
            float(np.deg2rad(phase)),
        )
        for trigger, frequency, phase in zip(
            window.trigger_us, window.frequency_mhz, window.phase_deg, strict=True
        )
    )
    return FTFPointPlan(
        joint,
        model,
        controls,
        flux,
        start,
        actual,
        basis,
        ro_model,
        resonators,
        (field_pulse,),
        grids,
    )
