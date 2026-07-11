from __future__ import annotations

from threading import Thread
from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.event_bus import EventMeta, EventOrigin, EventSubscriptions
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


def test_event_bus_subscription_handle_unsubscribes_lambda() -> None:
    """A subscribe() handle can remove callbacks that have no external name."""
    bus = EventBus()
    received: list[str] = []
    handle = bus.subscribe(MlChangedPayload, lambda _p: received.append("hit"))

    bus.emit(MlChangedPayload(ml=ModuleLibrary()))
    handle.unsubscribe()
    bus.emit(MlChangedPayload(ml=ModuleLibrary()))

    assert received == ["hit"]


def test_event_bus_subscription_handle_is_idempotent() -> None:
    bus = EventBus()
    received: list[MlChangedPayload] = []
    handle = bus.subscribe(MlChangedPayload, received.append)

    handle.unsubscribe()
    handle.unsubscribe()
    bus.emit(MlChangedPayload(ml=ModuleLibrary()))

    assert received == []


def test_event_subscription_group_unsubscribes_all_repeatedly() -> None:
    bus = EventBus()
    group = EventSubscriptions()
    md_received: list[MdChangedPayload] = []
    ml_received: list[MlChangedPayload] = []
    group.subscribe(bus, MdChangedPayload, md_received.append)
    group.subscribe(bus, MlChangedPayload, ml_received.append)

    group.unsubscribe_all("ignored signal payload")
    group.unsubscribe_all()
    bus.emit(MdChangedPayload(md=MetaDict()))
    bus.emit(MlChangedPayload(ml=ModuleLibrary()))

    assert md_received == []
    assert ml_received == []


def test_event_meta_sequence_is_process_wide_and_strictly_increasing() -> None:
    first_bus = EventBus()
    second_bus = EventBus()
    received: list[EventMeta] = []
    first_bus.subscribe_with_meta(
        MdChangedPayload, lambda _p, meta: received.append(meta)
    )
    second_bus.subscribe_with_meta(
        MdChangedPayload, lambda _p, meta: received.append(meta)
    )

    first_bus.emit(MdChangedPayload(md=MetaDict()))
    second_bus.emit(MdChangedPayload(md=MetaDict()))
    first_bus.emit(MdChangedPayload(md=MetaDict()))

    assert [meta.seq for meta in received] == sorted(meta.seq for meta in received)
    assert len({meta.seq for meta in received}) == 3


def test_event_origin_defaults_and_nested_scopes_restore() -> None:
    bus = EventBus()
    received: list[EventMeta] = []
    bus.subscribe_with_meta(MdChangedPayload, lambda _p, meta: received.append(meta))
    outer = EventOrigin(kind="agent", client_id="client-a")
    inner = EventOrigin(kind="system", operation_id="restore")

    bus.emit(MdChangedPayload(md=MetaDict()))
    with bus.origin(outer):
        bus.emit(MdChangedPayload(md=MetaDict()))
        with bus.origin(inner):
            bus.emit(MdChangedPayload(md=MetaDict()))
        bus.emit(MdChangedPayload(md=MetaDict()))
    bus.emit(MdChangedPayload(md=MetaDict()))

    assert [meta.origin for meta in received] == [
        EventOrigin(kind="user"),
        outer,
        inner,
        outer,
        EventOrigin(kind="user"),
    ]


def test_event_origin_does_not_propagate_to_new_thread() -> None:
    bus = EventBus()
    received: list[EventMeta] = []
    bus.subscribe_with_meta(MdChangedPayload, lambda _p, meta: received.append(meta))

    with bus.origin(EventOrigin(kind="agent", client_id="client-a")):
        thread = Thread(target=lambda: bus.emit(MdChangedPayload(md=MetaDict())))
        thread.start()
        thread.join()

    assert received == [EventMeta(seq=received[0].seq, origin=EventOrigin(kind="user"))]


def test_legacy_and_meta_subscribers_coexist_with_one_stamp() -> None:
    bus = EventBus()
    legacy_received: list[MdChangedPayload] = []
    meta_received: list[tuple[MdChangedPayload, EventMeta]] = []
    second_meta: list[EventMeta] = []
    bus.subscribe(MdChangedPayload, legacy_received.append)
    bus.subscribe_with_meta(
        MdChangedPayload, lambda payload, meta: meta_received.append((payload, meta))
    )
    bus.subscribe_with_meta(
        MdChangedPayload, lambda _payload, meta: second_meta.append(meta)
    )
    payload = MdChangedPayload(md=MetaDict())

    bus.emit(payload)

    assert legacy_received == [payload]
    assert meta_received == [(payload, second_meta[0])]
    assert meta_received[0][1] is second_meta[0]


def test_legacy_and_meta_subscriber_exceptions_are_isolated() -> None:
    bus = EventBus()
    later_legacy = MagicMock()
    later_meta = MagicMock()
    bus.subscribe(MdChangedPayload, MagicMock(side_effect=RuntimeError("legacy")))
    bus.subscribe(MdChangedPayload, later_legacy)
    bus.subscribe_with_meta(
        MdChangedPayload, MagicMock(side_effect=RuntimeError("meta"))
    )
    bus.subscribe_with_meta(MdChangedPayload, later_meta)

    bus.emit(MdChangedPayload(md=MetaDict()))

    assert later_legacy.called
    assert later_meta.called


def test_meta_subscription_handles_and_group_cleanup_are_idempotent() -> None:
    bus = EventBus()
    group = EventSubscriptions()
    received: list[EventMeta] = []
    standalone = bus.subscribe_with_meta(
        MdChangedPayload, lambda _payload, meta: received.append(meta)
    )
    group.subscribe_with_meta(
        bus, MdChangedPayload, lambda _payload, meta: received.append(meta)
    )

    standalone.unsubscribe()
    standalone.unsubscribe()
    group.unsubscribe_all()
    group.unsubscribe_all()
    bus.emit(MdChangedPayload(md=MetaDict()))

    assert received == []
