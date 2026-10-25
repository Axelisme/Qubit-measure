"""Lifetime callbacks retire before reverse-order, retryable cleanup."""

import pytest
from qtpy.QtCore import QCoreApplication, QEvent
from qtpy.QtWidgets import QPushButton
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.lifetime import LifetimeScope
from zcu_tools.gui.session.events import SocChangedPayload


def test_close_invalidates_callbacks_before_reentrant_cleanup_and_is_idempotent():
    scope = LifetimeScope()
    calls = []
    callback = scope.callback(lambda value: calls.append(value))
    callback("live")
    scope.add_cleanup(lambda: calls.append("first"))
    scope.add_cleanup(lambda: callback("late"))
    scope.add_cleanup(lambda: calls.append("last"))
    scope.close()
    scope.close()
    callback("retired")
    assert calls == ["live", "last", "first"]
    with pytest.raises(RuntimeError, match="closed"):
        scope.add_cleanup(lambda: None)


def test_broadcast_already_in_flight_does_not_reenter_a_retired_owner():
    bus = BaseEventBus()
    scope = LifetimeScope()
    calls = []
    first = bus.subscribe(SocChangedPayload, lambda _payload: scope.close())
    scope.add_cleanup(
        bus.subscribe(
            SocChangedPayload, scope.callback(lambda _payload: calls.append("late"))
        ).unsubscribe
    )
    try:
        bus.emit(SocChangedPayload(soc=None, soccfg=None))
        assert calls == []
    finally:
        first.unsubscribe()
        scope.close()


def test_external_qt_parent_deletion_invalidates_an_owner_callback(qapp):
    owner = QPushButton()
    scope = LifetimeScope(owner)
    calls = []
    callback = scope.callback(lambda: calls.append("called"))
    owner.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    callback()
    assert calls == []
    assert scope.closed


def test_scoped_event_filter_stops_delivery_before_removal(qapp):
    button = QPushButton()
    scope = LifetimeScope(button)
    calls = []

    def observed(_watched, event):
        if event is not None and event.type() == QEvent.Type.User:
            calls.append("event")
        return False

    scope.install_event_filter(button, observed)
    QCoreApplication.sendEvent(button, QEvent(QEvent.Type.User))
    scope.close()
    QCoreApplication.sendEvent(button, QEvent(QEvent.Type.User))
    assert calls == ["event"]
    button.deleteLater()


def test_qt_connection_adapts_unused_signal_arguments_and_retires(qapp):
    scope = LifetimeScope()
    button = QPushButton()
    calls = []
    scope.connect(button.clicked, lambda: calls.append("click"))
    button.click()
    scope.close()
    button.click()
    assert calls == ["click"]
    button.deleteLater()


def test_failed_close_keeps_callbacks_inactive_and_retries_only_unfinished_cleanup():
    scope = LifetimeScope()
    calls = []
    callback = scope.callback(lambda: calls.append("event"))

    def detach():
        calls.append("detach")
        if calls.count("detach") == 1:
            raise RuntimeError("detach failed")

    scope.add_cleanup(lambda: calls.append("earliest"))
    scope.add_cleanup(detach)
    scope.add_cleanup(lambda: calls.append("latest"))
    with pytest.raises(RuntimeError, match="detach failed"):
        scope.close()
    callback()
    scope.close()
    assert calls == ["latest", "detach", "detach", "earliest"]
