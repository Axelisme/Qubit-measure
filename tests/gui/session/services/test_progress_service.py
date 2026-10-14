"""ProgressService: containers keyed by operation, owner attach, rotation."""

from __future__ import annotations

import pytest
from zcu_tools.gui.session.ports import (
    ProgressEvent,
    ProgressEventKind,
)
from zcu_tools.gui.session.services.progress import ProgressService

from tests.gui._progress_fakes import DirectProgressTransport


@pytest.fixture
def transport():
    return DirectProgressTransport()


def _create(transport, op, handle, label="", total=None):
    transport.emit(
        ProgressEvent(op, handle, ProgressEventKind.CREATE, label=label, total=total)
    )


def _update(transport, op, handle, n, label=""):
    transport.emit(
        ProgressEvent(op, handle, ProgressEventKind.UPDATE, label=label, n=n)
    )


def test_create_update_close_round_trip(transport):
    svc = ProgressService(transport)
    svc.make_factory(1, owner_id="owner")
    _create(transport, 1, 0, total=10)
    _update(transport, 1, 0, 4)
    ((_, model),) = svc.bars_for_owner("owner")
    assert model.n == 4
    transport.emit(ProgressEvent(1, 0, ProgressEventKind.CLOSE))
    assert svc.bars_for_owner("owner") == ()


def test_update_before_create_is_tolerated(transport):
    svc = ProgressService(transport)
    svc.make_factory(1, owner_id="owner")
    _update(transport, 1, 0, 7, label="late")  # no CREATE first
    ((_, model),) = svc.bars_for_owner("owner")
    assert model.n == 7
    assert model.label == "late"


def test_multiple_handles_in_one_operation_are_independent(transport):
    svc = ProgressService(transport)
    svc.make_factory(1, owner_id="owner")
    _create(transport, 1, 0, total=10)
    _create(transport, 1, 1, total=5)
    _update(transport, 1, 0, 3)
    _update(transport, 1, 1, 2)
    bars = dict(svc.bars_for_owner("owner"))
    assert bars[0].n == 3
    assert bars[1].n == 2
    # closing one leaves the other
    transport.emit(ProgressEvent(1, 0, ProgressEventKind.CLOSE))
    bars = dict(svc.bars_for_owner("owner"))
    assert set(bars) == {1}


def test_event_for_unknown_operation_is_ignored(transport):
    svc = ProgressService(transport)
    # No make_factory for op=99 → no container; event must not raise.
    _create(transport, 99, 0, total=10)
    assert svc.bars_for_owner("owner") == ()


def test_attach_listener_fires_and_disposes(transport):
    svc = ProgressService(transport)
    calls = []
    dispose = svc.attach_by_owner("owner", lambda: calls.append(1))
    svc.make_factory(1, owner_id="owner")  # make_factory notifies the owner
    _create(transport, 1, 0, total=10)
    _update(transport, 1, 0, 5)
    assert len(calls) >= 2  # at least make_factory + one event
    before = len(calls)
    dispose()
    _update(transport, 1, 0, 6)
    assert len(calls) == before  # no further notifications after dispose


def test_listener_exception_is_logged_and_next_listener_still_runs(caplog, transport):
    svc = ProgressService(transport)
    calls: list[str] = []

    def broken() -> None:
        calls.append("broken")
        raise RuntimeError("listener boom")

    def healthy() -> None:
        calls.append("healthy")

    svc.attach_by_owner("owner", broken)
    svc.attach_by_owner("owner", healthy)

    with caplog.at_level("ERROR"):
        svc.make_factory(1, owner_id="owner")

    assert calls == ["broken", "healthy"]
    assert "progress listener failed" in caplog.text

    caplog.clear()
    _update(transport, 1, 0, 1)

    assert calls == ["broken", "healthy", "healthy"]
    assert "progress listener failed" not in caplog.text


def test_attach_before_any_operation_returns_empty(transport):
    svc = ProgressService(transport)
    svc.attach_by_owner("owner", lambda: None)
    assert svc.bars_for_owner("owner") == ()  # nothing live yet, no crash


def test_owner_follows_operation_rotation(transport):
    """A View attaches once by owner; successive operations are followed."""
    svc = ProgressService(transport)
    seen = []
    svc.attach_by_owner("owner", lambda: seen.append(svc.bars_for_owner("owner")))

    # operation #1
    svc.make_factory(5, owner_id="owner")
    _create(transport, 5, 0, total=10)
    _update(transport, 5, 0, 3)
    assert dict(svc.bars_for_owner("owner"))[0].n == 3
    svc.discard_operation(5)
    assert svc.bars_for_owner("owner") == ()

    # operation #2 for the SAME owner — attach (done once) still tracks it
    svc.make_factory(8, owner_id="owner")
    _create(transport, 8, 0, total=20)
    _update(transport, 8, 0, 11)
    assert dict(svc.bars_for_owner("owner"))[0].n == 11


def test_discard_clears_and_notifies(transport):
    svc = ProgressService(transport)
    calls = []
    svc.attach_by_owner("owner", lambda: calls.append(1))
    svc.make_factory(1, owner_id="owner")
    _create(transport, 1, 0, total=10)
    before = len(calls)
    svc.discard_operation(1)
    assert svc.bars_for_owner("owner") == ()
    assert len(calls) > before  # discard notified the owner


def test_discard_keeps_newer_owner_mapping(transport):
    """If a newer op already replaced the owner, discarding the old op must not
    wipe the live mapping."""
    svc = ProgressService(transport)
    svc.make_factory(1, owner_id="owner")
    svc.make_factory(2, owner_id="owner")  # owner now points at op 2
    svc.discard_operation(1)  # stale discard
    # op 2 still live for owner
    _create(transport, 2, 0, total=10)
    _update(transport, 2, 0, 4)
    assert dict(svc.bars_for_owner("owner"))[0].n == 4
