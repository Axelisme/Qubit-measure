"""Tests for the autofluxdep remote wire-version constants + event serializers.

The wire envelope / field-validation primitives are exercised by the shared
``tests/.../remote`` suite; here we only pin autofluxdep's own version constants
and smoke each event serializer (every payload type has one, and each produces a
JSON-friendly dict carrying a ``requery`` hint).
"""

from __future__ import annotations

import json

from zcu_tools.gui.app.autofluxdep.events.run import (
    NodeEnteredPayload,
    PointDonePayload,
    RunFailedPayload,
    RunFinishedPayload,
    RunStartedPayload,
    RunStoppedPayload,
)
from zcu_tools.gui.app.autofluxdep.events.workflow import (
    FluxChangedPayload,
    WorkflowChangedPayload,
)
from zcu_tools.gui.app.autofluxdep.services.remote.events import (
    EVENT_SERIALIZERS,
    wire_event_name,
)
from zcu_tools.gui.app.autofluxdep.services.remote.wire_version import (
    GUI_VERSION,
    WIRE_VERSION,
)


def test_versions_track_workflow_enabled_contract():
    assert WIRE_VERSION == 3
    assert GUI_VERSION == 3


def test_every_payload_type_has_a_serializer():
    # The serializer registry must cover every workflow + run payload the bus
    # emits, so no live event is silently dropped on the wire.
    expected = {
        WorkflowChangedPayload,
        FluxChangedPayload,
        RunStartedPayload,
        NodeEnteredPayload,
        PointDonePayload,
        RunFinishedPayload,
        RunStoppedPayload,
        RunFailedPayload,
    }
    assert set(EVENT_SERIALIZERS) == expected


def test_serializers_emit_json_friendly_requery_hints():
    samples = {
        WorkflowChangedPayload(name="qubit_freq"): "workflow_changed",
        FluxChangedPayload(count=3): "flux_changed",
        RunStartedPayload(): "run_started",
        NodeEnteredPayload(name="t1", idx=0): "node_entered",
        PointDonePayload(idx=1): "point_done",
        RunFinishedPayload(): "run_finished",
        RunStoppedPayload(): "run_stopped",
        RunFailedPayload(message="boom"): "run_failed",
    }
    for payload, wire_name in samples.items():
        serializer = EVENT_SERIALIZERS[type(payload)]
        wire = serializer(payload)
        assert wire is not None
        # JSON-serialisable (no numpy / objects leak onto the wire).
        json.dumps(wire)
        assert "requery" in wire
        assert wire_event_name(type(payload)) == wire_name
