"""Wire serializers for fluxdep EventBus push to remote clients.

Each entry in :data:`EVENT_SERIALIZERS` maps a fluxdep ``Payload`` *type* to a
function producing the JSON-friendly ``payload`` dict that goes on the wire, and
the wire event name. The fluxdep EventBus subscribes by payload type (not an
event enum), so the registry is keyed by type; ``payload.EVENT.value`` gives the
wire name.

Live numpy arrays / SpectrumEntry objects are never sent — a change push carries
only scalar identifiers (e.g. the spectrum name) plus a ``requery`` hint listing
the RPC method(s) the client may call to obtain current state.
"""

from __future__ import annotations

from typing import Callable, Mapping, Optional

from zcu_tools.fluxdep_gui.event_bus import (
    ActiveSpectrumChangedPayload,
    Payload,
    ProjectChangedPayload,
    SelectionChangedPayload,
    SpectrumAddedPayload,
    SpectrumChangedPayload,
    SpectrumRemovedPayload,
)

# Wire payload type: a JSON-friendly mapping (never None for fluxdep — every
# event carries something useful, but the type stays Optional for parity).
WirePayload = Optional[Mapping[str, object]]
Serializer = Callable[[Payload], WirePayload]


# ---------------------------------------------------------------------------
# Per-event serializers
# ---------------------------------------------------------------------------


def _ser_spectrum_added(payload: Payload) -> WirePayload:
    assert isinstance(payload, SpectrumAddedPayload)
    return {"name": payload.name, "requery": ["spectrum.list"]}


def _ser_spectrum_removed(payload: Payload) -> WirePayload:
    assert isinstance(payload, SpectrumRemovedPayload)
    return {"name": payload.name, "requery": ["spectrum.list"]}


def _ser_spectrum_changed(payload: Payload) -> WirePayload:
    assert isinstance(payload, SpectrumChangedPayload)
    return {"name": payload.name, "requery": ["spectrum.list"]}


def _ser_active_spectrum_changed(payload: Payload) -> WirePayload:
    assert isinstance(payload, ActiveSpectrumChangedPayload)
    return {"name": payload.name}


def _ser_selection_changed(payload: Payload) -> WirePayload:
    assert isinstance(payload, SelectionChangedPayload)
    del payload
    return {"requery": ["selection.pointcloud"]}


def _ser_project_changed(payload: Payload) -> WirePayload:
    assert isinstance(payload, ProjectChangedPayload)
    del payload
    return {"requery": ["project.info"]}


# ---------------------------------------------------------------------------
# Registry — keyed by payload type (the fluxdep EventBus subscribe key).
# ---------------------------------------------------------------------------


EVENT_SERIALIZERS: dict[type[Payload], Serializer] = {
    SpectrumAddedPayload: _ser_spectrum_added,
    SpectrumRemovedPayload: _ser_spectrum_removed,
    SpectrumChangedPayload: _ser_spectrum_changed,
    ActiveSpectrumChangedPayload: _ser_active_spectrum_changed,
    SelectionChangedPayload: _ser_selection_changed,
    ProjectChangedPayload: _ser_project_changed,
}


def wire_event_name(payload_type: type[Payload]) -> str:
    """Map a fluxdep ``Payload`` type to its lowercase wire event name."""
    return payload_type.EVENT.value
