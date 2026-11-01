"""Observed per-point routing partitions, without Cartesian state expansion."""

from dataclasses import dataclass, replace
from math import prod

import numpy as np
from numpy.typing import NDArray

from zcu_tools.core.simulation import (
    AcquireRequest,
    EventSlot,
    ReadoutSlot,
    TimeBase,
    TimelineTemplate,
)
from zcu_tools.experiment.readout_plan import ReadoutDrive, classify_readout
from zcu_tools.simulators.six_fluxonium.config import SixFluxoniumSettings


# Cover floating subtraction at nearest-cavity ties, in MHz.
_ROUTING_TIE_MHZ = 1e-9


@dataclass(frozen=True)
class RoutedGroup:
    """One observed routing pattern and cavity, with original point ownership.

    cavity: Q1..Q6 zero-based index. points: original flattened point indices.
    channels: ADCs observing this cavity in the pattern. templates: compact
    point-indexed controls and all readout windows for those ADCs. gen_ch: the
    common incident readout generator selected by YAML.
    """

    cavity: int
    points: NDArray[np.int64]
    channels: tuple[int, ...]
    templates: tuple[TimelineTemplate, ...]
    gen_ch: int


@dataclass(frozen=True)
class _TemplateRoutes:
    template: TimelineTemplate
    tones: tuple[EventSlot, ...]
    cavities: tuple[NDArray[np.int64], ...]
    terminal_masks: tuple[NDArray[np.bool_], ...]
    receivers: NDArray[np.int64]


def _nearest_cavity(
    frequencies_mhz: NDArray[np.float64], bare_mhz: NDArray[np.float64]
) -> NDArray[np.int64]:
    if not np.isfinite(frequencies_mhz).all():
        raise ValueError("readout routing requires finite frequencies")
    distances = np.abs(frequencies_mhz[..., None] - bare_mhz)
    best = distances.min(axis=-1, keepdims=True)
    # Subtraction tolerance in MHz, not a physical acceptance band. Never
    # choose a lowest-index cavity when roundoff makes a midpoint ambiguous.
    if np.any(np.sum(np.abs(distances - best) <= _ROUTING_TIE_MHZ, axis=-1) != 1):
        raise ValueError("ambiguous equidistant readout tone: nearest-cavity tie")
    return distances.argmin(axis=-1).astype(np.int64)


def _event_rows(
    event: EventSlot, rows: NDArray[np.int64], active: NDArray[np.bool_]
) -> EventSlot:
    return replace(
        event,
        start_ticks=event.start_ticks[rows],
        duration_ticks=event.duration_ticks[rows],
        frequency_mhz=event.frequency_mhz[rows],
        phase_deg=event.phase_deg[rows],
        gain=event.gain[rows],
        envelope_index=event.envelope_index[rows],
        active=active[rows],
    )


def _window_rows(window: ReadoutSlot, rows: NDArray[np.int64]) -> ReadoutSlot:
    return replace(
        window,
        trigger_ticks=window.trigger_ticks[rows],
        frequency_mhz=window.frequency_mhz[rows],
        phase_deg=window.phase_deg[rows],
        generator_channels=window.generator_channels[rows],
    )


def _route_template(
    template: TimelineTemplate,
    channels: tuple[int, ...],
    gen: int,
    bare: NDArray[np.float64],
    time_base: TimeBase,
    drive_channels: tuple[int, ...],
) -> _TemplateRoutes:
    partition = classify_readout(
        template, ReadoutDrive(gen, 1, "mux"), {}, time_base=time_base
    )
    if any(partition.cavity_drives):
        raise ValueError("six fluxonium does not support untriggered cavity drives")
    for event in template.events:
        if event.gen_ch not in (gen, *drive_channels):
            raise ValueError(f"unbound six-fluxonium generator {event.gen_ch}")
    receivers = []
    for channel in channels:
        window = next(w for w in template.readouts if w.channel == channel)
        selected = _nearest_cavity(window.frequency_mhz, bare)
        if np.any(selected != selected[:, :1]):
            raise ValueError(
                "one ADC's reads must observe the same cavity at each point"
            )
        if np.any(window.generator_channels != gen):
            raise ValueError("ADC requires its bound pulse or mux generator")
        receivers.append(selected[:, 0])
    tone_routes = []
    # Validate every active tone, including tones not observed by an ADC.
    tones = tuple(event for event in template.events if event.gen_ch == gen)
    for event in tones:
        routed = np.full(len(template.points), -1, dtype=np.int64)
        routed[event.active] = _nearest_cavity(event.frequency_mhz[event.active], bare)
        tone_routes.append(routed)
    return _TemplateRoutes(
        template,
        tones,
        tuple(tone_routes),
        tuple(event.active for event in partition.terminal_slots),
        np.column_stack(receivers),
    )


def _project_template(
    routes: _TemplateRoutes,
    compact: NDArray[np.int64],
    cavity: int,
    observed: tuple[int, ...],
    drive_gen_ch: int,
) -> tuple[TimelineTemplate, ...]:
    template, tones = routes.template, routes.tones
    selected_rows = np.flatnonzero(compact[template.points] >= 0)
    if not len(selected_rows):
        return ()
    active = tuple(
        e.active & (a == cavity) for e, a in zip(tones, routes.cavities, strict=True)
    )
    counts = np.sum(active, axis=0)
    terminal_counts = np.sum(
        [
            mask & terminal
            for mask, terminal in zip(active, routes.terminal_masks, strict=True)
        ],
        axis=0,
    )
    if np.any(terminal_counts[selected_rows] == 0):
        raise ValueError("observed cavity requires an incident terminal tone")
    templates = []
    # Different counts are structural templates, not zero-amplitude tones.
    for count in np.unique(counts[selected_rows]):
        rows = selected_rows[counts[selected_rows] == count]
        events = tuple(
            _event_rows(e, rows, e.active)
            for e in template.events
            if e.gen_ch == drive_gen_ch
        )
        events += tuple(
            _event_rows(e, rows, mask)
            for e, mask in zip(tones, active, strict=True)
            if np.any(mask[rows])
        )
        templates.append(
            replace(
                template,
                points=compact[template.points[rows]],
                events=events,
                readouts=tuple(
                    _window_rows(w, rows)
                    for w in template.readouts
                    if w.channel in observed
                ),
                duration_ticks=template.duration_ticks[rows],
            )
        )
    return tuple(templates)


def route_request(
    settings: SixFluxoniumSettings, request: AcquireRequest
) -> tuple[RoutedGroup, ...]:
    """Detach independent-cavity templates from the complete admitted request.

    Each tone selects its nearest bare cavity at each point, rejecting distances
    tied within 1e-9 MHz (subtraction tolerance). Each ADC's demod frequency
    selects the observed cavity; all reads of that ADC at a point must select
    the same cavity. Only observed routing patterns are enumerated.
    Triggered nonterminal blocks follow their incident tones into each source.
    Untriggered cavity drives, mixed pulse/mux routes, foreign generators, ties
    and absent terminal tones fail before any evolution or sampling. Microwave
    controls remain with their own circuit; they never drive another qubit.
    Nonempty charge requirements raise ValueError before projection or evolution:
    this environment does not yet support physical charge admission.
    """
    if request.charge_requirements:
        raise ValueError("six-fluxonium charge drive requirements are not supported")
    bare = np.array(
        [q.readout.resonator.frequency_ghz * 1000 for q in settings.qubits.ordered()]
    )
    channels = tuple(w.channel for w in request.readouts)
    if not channels or not request.templates:
        raise ValueError("six fluxonium requires readout templates")
    pulse = channels == (settings.pulse_ro_ch,)
    qubits = settings.qubits.ordered()
    mux_channels = tuple(q.readout.adc.ro_ch for q in qubits)
    if not pulse and any(ch not in mux_channels for ch in channels):
        raise ValueError("pulse and mux ADC routes must be acquired separately")
    gen = settings.pulse_gen_ch if pulse else settings.mux_gen_ch
    routing = np.empty((prod(request.point_dims), len(channels)), dtype=np.int64)
    templates = tuple(
        _route_template(t, channels, gen, bare, request.time_base, tuple(q.drive_line.gen_ch for q in qubits))
        for t in request.templates
    )
    for template in templates:
        routing[template.template.points] = template.receivers
    groups = []
    for pattern in np.unique(routing, axis=0):
        points = np.flatnonzero(np.all(routing == pattern, axis=1))
        compact = np.full(len(routing), -1, dtype=np.int64)
        compact[points] = np.arange(len(points))
        for cavity in np.unique(pattern):
            observed = tuple(
                ch
                for ch, selected in zip(channels, pattern, strict=True)
                if selected == cavity
            )
            projected = tuple(
                t
                for template in templates
                for t in _project_template(template, compact, int(cavity), observed, qubits[int(cavity)].drive_line.gen_ch)
            )
            groups.append(RoutedGroup(int(cavity), points, observed, projected, gen))
    return tuple(groups)
