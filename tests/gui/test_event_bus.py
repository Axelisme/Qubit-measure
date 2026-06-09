from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.app.main.event_bus import EventBus
from zcu_tools.gui.session.events import (
    MdChangedPayload,
    MlChangedPayload,
)
from zcu_tools.meta_tool import MetaDict, ModuleLibrary


def test_event_bus_dispatches_by_payload_type() -> None:
    """A payload reaches only subscribers of its concrete type."""
    bus = EventBus()
    md_received: list[MdChangedPayload] = []
    ml_received: list[MlChangedPayload] = []
    bus.subscribe(MdChangedPayload, md_received.append)
    bus.subscribe(MlChangedPayload, ml_received.append)

    bus.emit(MdChangedPayload(md=MetaDict()))

    assert len(md_received) == 1
    assert ml_received == []


def test_event_bus_swallows_subscriber_exceptions() -> None:
    """A raising subscriber does not propagate out of emit, and a later
    subscriber still runs."""
    bus = EventBus()
    later = MagicMock()
    bus.subscribe(
        MdChangedPayload, MagicMock(side_effect=RuntimeError("subscriber failed"))
    )
    bus.subscribe(MdChangedPayload, later)

    # No exception escapes emit.
    bus.emit(MdChangedPayload(md=MetaDict()))

    assert later.called


def test_event_bus_logs_all_subscriber_exceptions(
    caplog: pytest.LogCaptureFixture,
) -> None:
    bus = EventBus()
    first = MagicMock(side_effect=RuntimeError("first failed"))
    second = MagicMock(side_effect=ValueError("second failed"))
    third = MagicMock()
    bus.subscribe(MdChangedPayload, first)
    bus.subscribe(MdChangedPayload, second)
    bus.subscribe(MdChangedPayload, third)

    # emit does not raise even though two subscribers do.
    bus.emit(MdChangedPayload(md=MetaDict()))

    assert first.called
    assert second.called
    assert third.called
    messages = [record.getMessage() for record in caplog.records]
    assert sum("EventBus subscriber for" in message for message in messages) == 2


def test_event_bus_unsubscribe() -> None:
    """An unsubscribed callback no longer receives payloads."""
    bus = EventBus()
    received: list[MlChangedPayload] = []
    cb = received.append
    bus.subscribe(MlChangedPayload, cb)
    bus.unsubscribe(MlChangedPayload, cb)

    bus.emit(MlChangedPayload(ml=ModuleLibrary()))

    assert received == []
