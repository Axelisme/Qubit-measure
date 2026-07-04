"""Wire serializers for autofluxdep EventBus push to remote clients.

Each entry in :data:`EVENT_SERIALIZERS` maps an autofluxdep payload *type* to a
function producing the JSON-friendly ``payload`` dict that goes on the wire. The
autofluxdep EventBus (``BaseEventBus``) subscribes by payload type (not an event
enum), so the registry is keyed by type; ``payload.EVENT.value`` gives the wire
name.

Live numpy Results / Node objects are never sent — a change push carries only
scalar identifiers (the node name / flux index / message) plus a ``requery`` hint
listing the read method(s) the client may call to obtain current state.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from zcu_tools.gui.app.autofluxdep.events.run import (
    NodeEnteredPayload,
    PointDonePayload,
    RunContinuedPayload,
    RunFailedPayload,
    RunFinishedPayload,
    RunPausedPayload,
    RunPauseRequestedPayload,
    RunStartedPayload,
    RunStoppedPayload,
)
from zcu_tools.gui.app.autofluxdep.events.workflow import (
    FluxChangedPayload,
    WorkflowChangedPayload,
)
from zcu_tools.gui.event_bus import BasePayload

# Wire payload type: a JSON-friendly mapping (the read tools carry the real data;
# events only nudge the client to requery).
WirePayload = Mapping[str, object] | None
Serializer = Callable[[BasePayload], WirePayload]

# After a run-lifecycle change the run results + readiness flags both move, so the
# client should refresh both.
_RUN_REQUERY = ["state.check", "result.summary"]


# ---------------------------------------------------------------------------
# Per-event serializers — workflow edits + run lifecycle.
# ---------------------------------------------------------------------------


def _ser_workflow_changed(payload: BasePayload) -> WirePayload:
    assert isinstance(payload, WorkflowChangedPayload)
    return {"name": payload.name, "requery": ["workflow.list", "state.check"]}


def _ser_flux_changed(payload: BasePayload) -> WirePayload:
    assert isinstance(payload, FluxChangedPayload)
    return {"count": payload.count, "requery": ["state.check"]}


def _ser_run_started(payload: BasePayload) -> WirePayload:
    assert isinstance(payload, RunStartedPayload)
    del payload
    return {"requery": _RUN_REQUERY}


def _ser_run_pause_requested(payload: BasePayload) -> WirePayload:
    assert isinstance(payload, RunPauseRequestedPayload)
    del payload
    return {"requery": ["state.check"]}


def _ser_run_paused(payload: BasePayload) -> WirePayload:
    assert isinstance(payload, RunPausedPayload)
    return {"next_flux_idx": payload.next_flux_idx, "requery": _RUN_REQUERY}


def _ser_run_continued(payload: BasePayload) -> WirePayload:
    assert isinstance(payload, RunContinuedPayload)
    return {"next_flux_idx": payload.next_flux_idx, "requery": _RUN_REQUERY}


def _ser_node_entered(payload: BasePayload) -> WirePayload:
    assert isinstance(payload, NodeEnteredPayload)
    return {"name": payload.name, "idx": payload.idx, "requery": ["result.summary"]}


def _ser_point_done(payload: BasePayload) -> WirePayload:
    assert isinstance(payload, PointDonePayload)
    return {"idx": payload.idx, "requery": ["result.summary"]}


def _ser_run_finished(payload: BasePayload) -> WirePayload:
    assert isinstance(payload, RunFinishedPayload)
    del payload
    return {"requery": _RUN_REQUERY}


def _ser_run_stopped(payload: BasePayload) -> WirePayload:
    assert isinstance(payload, RunStoppedPayload)
    del payload
    return {"requery": _RUN_REQUERY}


def _ser_run_failed(payload: BasePayload) -> WirePayload:
    assert isinstance(payload, RunFailedPayload)
    return {"message": payload.message, "requery": _RUN_REQUERY}


# ---------------------------------------------------------------------------
# Registry — keyed by payload type (the autofluxdep EventBus subscribe key).
# ---------------------------------------------------------------------------


EVENT_SERIALIZERS: dict[type[BasePayload], Serializer] = {
    WorkflowChangedPayload: _ser_workflow_changed,
    FluxChangedPayload: _ser_flux_changed,
    RunStartedPayload: _ser_run_started,
    RunPauseRequestedPayload: _ser_run_pause_requested,
    RunPausedPayload: _ser_run_paused,
    RunContinuedPayload: _ser_run_continued,
    NodeEnteredPayload: _ser_node_entered,
    PointDonePayload: _ser_point_done,
    RunFinishedPayload: _ser_run_finished,
    RunStoppedPayload: _ser_run_stopped,
    RunFailedPayload: _ser_run_failed,
}


def wire_event_name(payload_type: type[BasePayload]) -> str:
    """Map an autofluxdep payload type to its lowercase wire event name."""
    return payload_type.EVENT.value
