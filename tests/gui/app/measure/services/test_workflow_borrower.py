"""Borrower disposal detaches callbacks before the GUI deletes its children."""

import pytest
from zcu_tools.gui.app.measure.services import app_services
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.session.events import SocChangedPayload


def test_dispose_stops_owned_timer_and_subscriptions_once():
    borrower_type = app_services.WorkflowBorrower
    bus = BaseEventBus()
    calls = []
    borrower = borrower_type()
    subscription = bus.subscribe(SocChangedPayload, lambda payload: calls.append("event"))
    borrower.add_cleanup(subscription.unsubscribe)
    borrower.add_cleanup(lambda: calls.append("timer stopped"))
    bus.emit(SocChangedPayload(soc=None, soccfg=None))
    borrower.dispose()
    borrower.dispose()
    bus.emit(SocChangedPayload(soc=None, soccfg=None))
    assert calls == ["event", "timer stopped"]
    with pytest.raises(RuntimeError, match="disposed"):
        borrower.add_cleanup(lambda: None)


def test_failed_dispose_retains_cleanup_for_retry_before_children_can_be_deleted():
    borrower_type = app_services.WorkflowBorrower
    borrower = borrower_type()
    calls = []

    def detach():
        calls.append("detach")
        if len(calls) == 1:
            raise RuntimeError("detach failed")

    borrower.add_cleanup(detach)
    with pytest.raises(RuntimeError, match="detach failed"):
        borrower.dispose()
    assert not borrower.disposed
    borrower.dispose()
    assert borrower.disposed
    assert calls == ["detach", "detach"]
