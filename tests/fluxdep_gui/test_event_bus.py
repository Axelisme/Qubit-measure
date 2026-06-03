"""Tests for the fluxdep-gui EventBus."""

from __future__ import annotations

from zcu_tools.fluxdep_gui.event_bus import (
    EventBus,
    FluxDepEvent,
    SpectrumAddedPayload,
    SpectrumChangedPayload,
)


def test_subscribe_and_emit_delivers_payload():
    bus = EventBus()
    seen = []
    bus.subscribe(SpectrumAddedPayload, lambda p: seen.append(p.name))
    bus.emit(SpectrumAddedPayload(name="a"))
    assert seen == ["a"]


def test_emit_routes_by_payload_type_only():
    bus = EventBus()
    added, changed = [], []
    bus.subscribe(SpectrumAddedPayload, lambda p: added.append(p.name))
    bus.subscribe(SpectrumChangedPayload, lambda p: changed.append(p.name))
    bus.emit(SpectrumChangedPayload(name="x"))
    assert added == [] and changed == ["x"]


def test_unsubscribe_stops_delivery():
    bus = EventBus()
    seen = []
    cb = lambda p: seen.append(p.name)  # noqa: E731
    bus.subscribe(SpectrumAddedPayload, cb)
    bus.unsubscribe(SpectrumAddedPayload, cb)
    bus.emit(SpectrumAddedPayload(name="a"))
    assert seen == []


def test_one_raising_subscriber_does_not_break_others():
    bus = EventBus()
    seen = []

    def bad(_p):
        raise RuntimeError("boom")

    bus.subscribe(SpectrumAddedPayload, bad)
    bus.subscribe(SpectrumAddedPayload, lambda p: seen.append(p.name))
    bus.emit(SpectrumAddedPayload(name="a"))  # must not raise
    assert seen == ["a"]


def test_payload_event_tag_is_fixed():
    assert SpectrumAddedPayload.EVENT is FluxDepEvent.SPECTRUM_ADDED
    assert SpectrumChangedPayload.EVENT is FluxDepEvent.SPECTRUM_CHANGED
