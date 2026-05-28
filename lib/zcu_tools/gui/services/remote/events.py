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

from zcu_tools.gui.event_bus import (
    ContextSwitchedPayload,
    DeviceChangedPayload,
    DeviceSetupChangedPayload,
    GuiEvent,
    MdChangedPayload,
    MlChangedPayload,
    Payload,
    PredictorChangedPayload,
    RunLockChangedPayload,
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


def _ser_run_lock_changed(payload: Payload) -> WirePayload:
    assert isinstance(payload, RunLockChangedPayload)
    out: dict[str, object] = {"running_tab_id": payload.running_tab_id}
    # Terminal emits carry an outcome (finished/failed/cancelled); the run-start
    # emit leaves it None. Surface the outcome so a single event stream lets the
    # agent distinguish success / failure / cancellation.
    if payload.outcome is not None:
        out["tab_id"] = payload.tab_id
        out["outcome"] = payload.outcome
        if payload.error_message is not None:
            out["error_message"] = payload.error_message
        out["requery"] = ["tab.snapshot"]
    return out


def _ser_predictor_changed(payload: Payload) -> WirePayload:
    assert isinstance(payload, PredictorChangedPayload)
    del payload
    # No scalar field worth shipping; client should requery if needed.
    return {}


def _ser_device_changed(payload: Payload) -> WirePayload:
    assert isinstance(payload, DeviceChangedPayload)
    return {"name": payload.name, "requery": ["device.list"]}


def _ser_device_setup_changed(payload: Payload) -> WirePayload:
    assert isinstance(payload, DeviceSetupChangedPayload)
    # DeviceSetupSnapshot holds live Qt-thread state; never push.
    del payload
    return {"requery": ["device.active_setup"]}


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
    GuiEvent.RUN_LOCK_CHANGED: _ser_run_lock_changed,
    GuiEvent.PREDICTOR_CHANGED: _ser_predictor_changed,
    GuiEvent.DEVICE_CHANGED: _ser_device_changed,
    GuiEvent.DEVICE_SETUP_CHANGED: _ser_device_setup_changed,
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
