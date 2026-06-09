"""ProgressService: containers keyed by operation, owner attach, rotation."""

from __future__ import annotations

from zcu_tools.gui.app.main.services.progress import ProgressService
from zcu_tools.gui.session.ports import (
    ProgressEvent,
    ProgressEventKind,
)

from ._progress_fakes import DirectProgressTransport


def _create(svc, op, handle, label="", total=None):
    svc._on_event(
        ProgressEvent(op, handle, ProgressEventKind.CREATE, label=label, total=total)
    )


def _update(svc, op, handle, n, label=""):
    svc._on_event(ProgressEvent(op, handle, ProgressEventKind.UPDATE, label=label, n=n))


def test_create_update_close_round_trip():
    svc = ProgressService(DirectProgressTransport())
    svc.make_factory(1, owner_id="owner")
    _create(svc, 1, 0, total=10)
    _update(svc, 1, 0, 4)
    ((_, model),) = svc.bars_for_owner("owner")
    assert model.n == 4
    svc._on_event(ProgressEvent(1, 0, ProgressEventKind.CLOSE))
    assert svc.bars_for_owner("owner") == ()


def test_update_before_create_is_tolerated():
    svc = ProgressService(DirectProgressTransport())
    svc.make_factory(1, owner_id="owner")
    _update(svc, 1, 0, 7, label="late")  # no CREATE first
    ((_, model),) = svc.bars_for_owner("owner")
    assert model.n == 7
    assert model.label == "late"


def test_multiple_handles_in_one_operation_are_independent():
    svc = ProgressService(DirectProgressTransport())
    svc.make_factory(1, owner_id="owner")
    _create(svc, 1, 0, total=10)
    _create(svc, 1, 1, total=5)
    _update(svc, 1, 0, 3)
    _update(svc, 1, 1, 2)
    bars = dict(svc.bars_for_owner("owner"))
    assert bars[0].n == 3
    assert bars[1].n == 2
    # closing one leaves the other
    svc._on_event(ProgressEvent(1, 0, ProgressEventKind.CLOSE))
    bars = dict(svc.bars_for_owner("owner"))
    assert set(bars) == {1}


def test_event_for_unknown_operation_is_ignored():
    svc = ProgressService(DirectProgressTransport())
    # No make_factory for op=99 → no container; event must not raise.
    _create(svc, 99, 0, total=10)
    assert svc.bars_for_owner("owner") == ()


def test_attach_listener_fires_and_disposes():
    svc = ProgressService(DirectProgressTransport())
    calls = []
    dispose = svc.attach_by_owner("owner", lambda: calls.append(1))
    svc.make_factory(1, owner_id="owner")  # make_factory notifies the owner
    _create(svc, 1, 0, total=10)
    _update(svc, 1, 0, 5)
    assert len(calls) >= 2  # at least make_factory + one event
    before = len(calls)
    dispose()
    _update(svc, 1, 0, 6)
    assert len(calls) == before  # no further notifications after dispose


def test_attach_before_any_operation_returns_empty():
    svc = ProgressService(DirectProgressTransport())
    svc.attach_by_owner("owner", lambda: None)
    assert svc.bars_for_owner("owner") == ()  # nothing live yet, no crash


def test_owner_follows_operation_rotation():
    """A View attaches once by owner; successive operations are followed."""
    svc = ProgressService(DirectProgressTransport())
    seen = []
    svc.attach_by_owner("owner", lambda: seen.append(svc.bars_for_owner("owner")))

    # operation #1
    svc.make_factory(5, owner_id="owner")
    _create(svc, 5, 0, total=10)
    _update(svc, 5, 0, 3)
    assert dict(svc.bars_for_owner("owner"))[0].n == 3
    svc.discard_operation(5)
    assert svc.bars_for_owner("owner") == ()

    # operation #2 for the SAME owner — attach (done once) still tracks it
    svc.make_factory(8, owner_id="owner")
    _create(svc, 8, 0, total=20)
    _update(svc, 8, 0, 11)
    assert dict(svc.bars_for_owner("owner"))[0].n == 11


def test_discard_clears_and_notifies():
    svc = ProgressService(DirectProgressTransport())
    calls = []
    svc.attach_by_owner("owner", lambda: calls.append(1))
    svc.make_factory(1, owner_id="owner")
    _create(svc, 1, 0, total=10)
    before = len(calls)
    svc.discard_operation(1)
    assert svc.bars_for_owner("owner") == ()
    assert len(calls) > before  # discard notified the owner


def test_discard_keeps_newer_owner_mapping():
    """If a newer op already replaced the owner, discarding the old op must not
    wipe the live mapping."""
    svc = ProgressService(DirectProgressTransport())
    svc.make_factory(1, owner_id="owner")
    svc.make_factory(2, owner_id="owner")  # owner now points at op 2
    svc.discard_operation(1)  # stale discard
    # op 2 still live for owner
    _create(svc, 2, 0, total=10)
    _update(svc, 2, 0, 4)
    assert dict(svc.bars_for_owner("owner"))[0].n == 4
