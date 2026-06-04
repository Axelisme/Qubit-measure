"""Wire serializers for dispersive EventBus push to remote clients.

Each entry in :data:`EVENT_SERIALIZERS` maps a dispersive ``Payload`` *type* to a
function producing the JSON-friendly ``payload`` dict and the wire event name. The
EventBus subscribes by payload type; ``payload.EVENT.value`` gives the wire name.

Live numpy arrays are never sent — a change push carries only scalar identifiers
plus a ``requery`` hint listing the RPC method(s) the client may call for current
state.
"""

from __future__ import annotations

from typing import Callable, Mapping, Optional

from zcu_tools.gui.app.dispersive.event_bus import (
    DispFitChangedPayload,
    FitInputsLoadedPayload,
    OnetoneLoadedPayload,
    Payload,
    PreprocessChangedPayload,
    ProjectChangedPayload,
)

WirePayload = Optional[Mapping[str, object]]
Serializer = Callable[[Payload], WirePayload]


def _ser_project_changed(payload: Payload) -> WirePayload:
    assert isinstance(payload, ProjectChangedPayload)
    del payload
    return {"requery": ["project.info"]}


def _ser_fit_inputs_loaded(payload: Payload) -> WirePayload:
    assert isinstance(payload, FitInputsLoadedPayload)
    return {"has_inputs": payload.has_inputs, "requery": ["fit_inputs.info"]}


def _ser_onetone_loaded(payload: Payload) -> WirePayload:
    assert isinstance(payload, OnetoneLoadedPayload)
    return {"name": payload.name, "requery": ["state.check"]}


def _ser_preprocess_changed(payload: Payload) -> WirePayload:
    assert isinstance(payload, PreprocessChangedPayload)
    del payload
    return {"requery": ["preprocess.status"]}


def _ser_disp_fit_changed(payload: Payload) -> WirePayload:
    assert isinstance(payload, DispFitChangedPayload)
    return {"has_result": payload.has_result, "requery": ["fit.result"]}


EVENT_SERIALIZERS: dict[type[Payload], Serializer] = {
    ProjectChangedPayload: _ser_project_changed,
    FitInputsLoadedPayload: _ser_fit_inputs_loaded,
    OnetoneLoadedPayload: _ser_onetone_loaded,
    PreprocessChangedPayload: _ser_preprocess_changed,
    DispFitChangedPayload: _ser_disp_fit_changed,
}


def wire_event_name(payload_type: type[Payload]) -> str:
    """Map a dispersive ``Payload`` type to its lowercase wire event name."""
    return payload_type.EVENT.value
