"""Wire serializers for ``GuiEvent`` push to remote clients.

Each entry in :data:`EVENT_SERIALIZERS` maps a ``GuiEvent`` to a function
producing the JSON-friendly ``payload`` dict that goes on the wire. Live
Python references (``MetaDict``, ``ModuleLibrary``, ``SocHandle``,
``DeviceSetupSnapshot``, ``BaseDeviceInfo``, ``FluxoniumPredictor``) are
**never** sent — instead the wire payload carries a ``requery`` hint
listing the RPC method(s) the client should call to obtain current state.

Adding a new event:
  1. Confirm the payload dataclass exists in ``gui/event_bus.py``.
  2. Add a serializer returning either a JSON-able dict or ``None`` to
     suppress the push (None means "do not forward").
  3. Register in :data:`EVENT_SERIALIZERS`.
  4. Document the wire shape in this module's docstring above and in
     ``services/remote/AI_NOTE.md``.
"""

from __future__ import annotations

from typing import Callable, Mapping, Optional

from zcu_tools.gui.app.main.event_bus import (
    ContextSwitchedPayload,
    DeviceChangedPayload,
    DeviceSetupFinishedPayload,
    DeviceSetupStartedPayload,
    GuiEvent,
    MdChangedPayload,
    MlChangedPayload,
    Payload,
    PredictorChangedPayload,
    RunFinishedPayload,
    RunStartedPayload,
    SocChangedPayload,
    TabAddedPayload,
    TabClosedPayload,
    TabContentChangedPayload,
    TabInteractionChangedPayload,
)

# Wire payload type: a JSON-friendly mapping or ``None`` to drop the event.
WirePayload = Optional[Mapping[str, object]]
Serializer = Callable[[Payload], WirePayload]


# ---------------------------------------------------------------------------
# Per-event serializers
# ---------------------------------------------------------------------------


def _ser_tab_added(payload: Payload) -> WirePayload:
    assert isinstance(payload, TabAddedPayload)
    return {"tab_id": payload.tab_id, "adapter_name": payload.adapter_name}


def _ser_tab_closed(payload: Payload) -> WirePayload:
    assert isinstance(payload, TabClosedPayload)
    return {"tab_id": payload.tab_id}


def _ser_tab_content_changed(payload: Payload) -> WirePayload:
    assert isinstance(payload, TabContentChangedPayload)
    return {"tab_id": payload.tab_id, "requery": ["tab.snapshot"]}


def _ser_tab_interaction_changed(payload: Payload) -> WirePayload:
    assert isinstance(payload, TabInteractionChangedPayload)
    return {"tab_id": payload.tab_id, "requery": ["tab.snapshot"]}


def _ser_run_started(payload: Payload) -> WirePayload:
    assert isinstance(payload, RunStartedPayload)
    return {"tab_id": payload.tab_id}


def _ser_run_finished(payload: Payload) -> WirePayload:
    assert isinstance(payload, RunFinishedPayload)
    # outcome ∈ finished/failed/cancelled lets the agent tell success from
    # failure from cancellation; error_message only on failure.
    out: dict[str, object] = {
        "tab_id": payload.tab_id,
        "outcome": payload.outcome,
        "requery": ["tab.snapshot"],
    }
    if payload.error_message is not None:
        out["error_message"] = payload.error_message
    return out


def _ser_predictor_changed(payload: Payload) -> WirePayload:
    assert isinstance(payload, PredictorChangedPayload)
    del payload
    # No scalar field worth shipping; client should requery if needed.
    return {}


def _ser_device_changed(payload: Payload) -> WirePayload:
    assert isinstance(payload, DeviceChangedPayload)
    return {"name": payload.name, "requery": ["device.list"]}


def _ser_device_setup_started(payload: Payload) -> WirePayload:
    assert isinstance(payload, DeviceSetupStartedPayload)
    # Live progress is polled via operation.progress (by operation_id), not pushed.
    return {"name": payload.name}


def _ser_device_setup_finished(payload: Payload) -> WirePayload:
    assert isinstance(payload, DeviceSetupFinishedPayload)
    out: dict[str, object] = {"name": payload.name, "outcome": payload.outcome}
    if payload.error_message is not None:
        out["error_message"] = payload.error_message
    return out


def _ser_context_switched(payload: Payload) -> WirePayload:
    assert isinstance(payload, ContextSwitchedPayload)
    # MetaDict / ModuleLibrary are live; let the client requery.
    del payload
    return {"requery": ["context.active"]}


def _ser_md_changed(payload: Payload) -> WirePayload:
    assert isinstance(payload, MdChangedPayload)
    del payload
    return {"requery": ["context.get_md_attr"]}


def _ser_ml_changed(payload: Payload) -> WirePayload:
    assert isinstance(payload, MlChangedPayload)
    del payload
    return {"requery": ["context.get_ml"]}


def _ser_soc_changed(payload: Payload) -> WirePayload:
    assert isinstance(payload, SocChangedPayload)
    return {"connected": payload.soc is not None}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


EVENT_SERIALIZERS: dict[GuiEvent, Serializer] = {
    GuiEvent.TAB_ADDED: _ser_tab_added,
    GuiEvent.TAB_CLOSED: _ser_tab_closed,
    GuiEvent.TAB_CONTENT_CHANGED: _ser_tab_content_changed,
    GuiEvent.TAB_INTERACTION_CHANGED: _ser_tab_interaction_changed,
    GuiEvent.RUN_STARTED: _ser_run_started,
    GuiEvent.RUN_FINISHED: _ser_run_finished,
    GuiEvent.PREDICTOR_CHANGED: _ser_predictor_changed,
    GuiEvent.DEVICE_CHANGED: _ser_device_changed,
    GuiEvent.DEVICE_SETUP_STARTED: _ser_device_setup_started,
    GuiEvent.DEVICE_SETUP_FINISHED: _ser_device_setup_finished,
    GuiEvent.CONTEXT_SWITCHED: _ser_context_switched,
    GuiEvent.MD_CHANGED: _ser_md_changed,
    GuiEvent.ML_CHANGED: _ser_ml_changed,
    GuiEvent.SOC_CHANGED: _ser_soc_changed,
}


def wire_event_name(event: GuiEvent) -> str:
    """Map ``GuiEvent`` to its lowercase wire name.

    ``GuiEvent`` already extends ``str`` with the lowercase value (e.g.
    ``"tab_added"``), but this indirection keeps the wire schema explicit
    and lets future renames stay coherent.
    """
    return event.value
