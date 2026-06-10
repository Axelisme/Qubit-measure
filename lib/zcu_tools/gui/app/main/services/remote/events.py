"""Wire serializers for measure-gui EventBus push to remote clients.

Each entry in :data:`EVENT_SERIALIZERS` maps a ``Payload`` *type* to a function
producing the JSON-friendly ``payload`` dict that goes on the wire. Live
Python references (``MetaDict``, ``ModuleLibrary``, ``SocHandle``,
``DeviceSetupSnapshot``, ``BaseDeviceInfo``, ``FluxoniumPredictor``) are
**never** sent — instead the wire payload carries a ``requery`` hint
listing the RPC method(s) the client should call to obtain current state. The
measure-gui EventBus subscribes by payload type (not an event enum), so the
registry is keyed by type; ``payload.EVENT.value`` gives the wire name.

Adding a new event:
  1. Confirm the payload dataclass exists (tab events in
     ``app/main/events/tab.py``, run events in ``app/main/events/run.py``;
     session-core payloads in ``gui/session/events.py``).
  2. Add a serializer returning either a JSON-able dict or ``None`` to
     suppress the push (None means "do not forward").
  3. Register in :data:`EVENT_SERIALIZERS`.
  4. Document the wire shape in this module's docstring above and in
     ``services/remote/AI_NOTE.md``.
"""

from __future__ import annotations

from typing import Callable, Mapping, Optional

from zcu_tools.gui.app.main.events.run import RunFinishedPayload, RunStartedPayload
from zcu_tools.gui.app.main.events.tab import (
    TabAddedPayload,
    TabClosedPayload,
    TabContentChangedPayload,
    TabInteractionChangedPayload,
)
from zcu_tools.gui.event_bus import BasePayload
from zcu_tools.gui.session.events import (
    ContextSwitchedPayload,
    DeviceChangedPayload,
    DeviceSetupFinishedPayload,
    DeviceSetupStartedPayload,
    MdChangedPayload,
    MlChangedPayload,
    PredictorChangedPayload,
    SocChangedPayload,
)

# Wire payload type: a JSON-friendly mapping or ``None`` to drop the event.
WirePayload = Optional[Mapping[str, object]]
Serializer = Callable[[BasePayload], WirePayload]


# ---------------------------------------------------------------------------
# Per-event serializers
# ---------------------------------------------------------------------------


def _ser_tab_added(payload: BasePayload) -> WirePayload:
    assert isinstance(payload, TabAddedPayload)
    return {"tab_id": payload.tab_id, "adapter_name": payload.adapter_name}


def _ser_tab_closed(payload: BasePayload) -> WirePayload:
    assert isinstance(payload, TabClosedPayload)
    return {"tab_id": payload.tab_id}


def _ser_tab_content_changed(payload: BasePayload) -> WirePayload:
    assert isinstance(payload, TabContentChangedPayload)
    return {"tab_id": payload.tab_id, "requery": ["tab.snapshot"]}


def _ser_tab_interaction_changed(payload: BasePayload) -> WirePayload:
    assert isinstance(payload, TabInteractionChangedPayload)
    return {"tab_id": payload.tab_id, "requery": ["tab.snapshot"]}


def _ser_run_started(payload: BasePayload) -> WirePayload:
    assert isinstance(payload, RunStartedPayload)
    return {"tab_id": payload.tab_id}


def _ser_run_finished(payload: BasePayload) -> WirePayload:
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


def _ser_predictor_changed(payload: BasePayload) -> WirePayload:
    assert isinstance(payload, PredictorChangedPayload)
    del payload
    # No scalar field worth shipping; client should requery if needed.
    return {}


def _ser_device_changed(payload: BasePayload) -> WirePayload:
    assert isinstance(payload, DeviceChangedPayload)
    return {"name": payload.name, "requery": ["device.list"]}


def _ser_device_setup_started(payload: BasePayload) -> WirePayload:
    assert isinstance(payload, DeviceSetupStartedPayload)
    # Live progress is polled via operation.progress (by operation_id), not pushed.
    return {"name": payload.name}


def _ser_device_setup_finished(payload: BasePayload) -> WirePayload:
    assert isinstance(payload, DeviceSetupFinishedPayload)
    out: dict[str, object] = {"name": payload.name, "outcome": payload.outcome}
    if payload.error_message is not None:
        out["error_message"] = payload.error_message
    return out


def _ser_context_switched(payload: BasePayload) -> WirePayload:
    assert isinstance(payload, ContextSwitchedPayload)
    # MetaDict / ModuleLibrary are live; let the client requery.
    del payload
    return {"requery": ["context.active"]}


def _ser_md_changed(payload: BasePayload) -> WirePayload:
    assert isinstance(payload, MdChangedPayload)
    del payload
    return {"requery": ["context.get_md_attr"]}


def _ser_ml_changed(payload: BasePayload) -> WirePayload:
    assert isinstance(payload, MlChangedPayload)
    del payload
    return {"requery": ["context.get_ml"]}


def _ser_soc_changed(payload: BasePayload) -> WirePayload:
    assert isinstance(payload, SocChangedPayload)
    return {"connected": payload.soc is not None}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


EVENT_SERIALIZERS: dict[type[BasePayload], Serializer] = {
    TabAddedPayload: _ser_tab_added,
    TabClosedPayload: _ser_tab_closed,
    TabContentChangedPayload: _ser_tab_content_changed,
    TabInteractionChangedPayload: _ser_tab_interaction_changed,
    RunStartedPayload: _ser_run_started,
    RunFinishedPayload: _ser_run_finished,
    PredictorChangedPayload: _ser_predictor_changed,
    DeviceChangedPayload: _ser_device_changed,
    DeviceSetupStartedPayload: _ser_device_setup_started,
    DeviceSetupFinishedPayload: _ser_device_setup_finished,
    ContextSwitchedPayload: _ser_context_switched,
    MdChangedPayload: _ser_md_changed,
    MlChangedPayload: _ser_ml_changed,
    SocChangedPayload: _ser_soc_changed,
}


def wire_event_name(payload_type: type[BasePayload]) -> str:
    """Map a ``Payload`` type to its lowercase wire event name.

    The enum value is the wire name (e.g. ``"tab_added"``); routing through
    ``payload_type.EVENT`` keeps the wire schema explicit and lets future
    renames stay coherent.
    """
    return payload_type.EVENT.value
