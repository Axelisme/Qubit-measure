"""Tests for the dispersive-fit-gui EventBus."""

from __future__ import annotations

from zcu_tools.gui.app.dispersive.event_bus import (
    DispersiveEvent,
    DispFitChangedPayload,
    EventBus,
    OnetoneLoadedPayload,
    PreprocessChangedPayload,
)


def test_subscribe_and_emit_delivers_payload():
    bus = EventBus()
    seen = []
    bus.subscribe(OnetoneLoadedPayload, lambda p: seen.append(p.name))
    bus.emit(OnetoneLoadedPayload(name="r1"))
    assert seen == ["r1"]


def test_emit_routes_by_payload_type_only():
    bus = EventBus()
    loaded, preprocessed = [], []
    bus.subscribe(OnetoneLoadedPayload, lambda p: loaded.append(p.name))
    bus.subscribe(PreprocessChangedPayload, lambda _p: preprocessed.append(True))
    bus.emit(PreprocessChangedPayload())
    assert loaded == [] and preprocessed == [True]


def test_unsubscribe_stops_delivery():
    bus = EventBus()
    seen = []
    cb = lambda p: seen.append(p.name)  # noqa: E731
    bus.subscribe(OnetoneLoadedPayload, cb)
    bus.unsubscribe(OnetoneLoadedPayload, cb)
    bus.emit(OnetoneLoadedPayload(name="r1"))
    assert seen == []


def test_one_raising_subscriber_does_not_break_others():
    bus = EventBus()
    seen = []

    def boom(_p):
        raise RuntimeError("bad subscriber")

    bus.subscribe(DispFitChangedPayload, boom)
    bus.subscribe(DispFitChangedPayload, lambda p: seen.append(p.has_result))
    bus.emit(DispFitChangedPayload(has_result=True))
    assert seen == [True]


def test_payload_carries_its_event_tag():
    assert OnetoneLoadedPayload(name="r1").EVENT is DispersiveEvent.ONETONE_LOADED
    assert PreprocessChangedPayload().EVENT is DispersiveEvent.PREPROCESS_CHANGED
    assert DispFitChangedPayload().EVENT is DispersiveEvent.DISP_FIT_CHANGED
