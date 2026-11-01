"""Explicit and lexical operation leases share resource cleanup, not ambient scopes."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace

import pytest
from zcu_tools.gui.event_bus import BaseEventBus, EventOrigin
from zcu_tools.gui.session.operation_handles import OperationHandles, OperationOutcome
from zcu_tools.gui.session.operation_runner import (
    ExclusionRequest,
    OperationRunner,
    OperationSpec,
)
from zcu_tools.gui.session.ports import OperationConflictError

from tests.gui.session.operation_runner_support import FakeBg, FakeGate, make_runner


def _lease_spec() -> OperationSpec:
    """Owner policy for a handle-only synchronous operation."""
    return OperationSpec(
        exclusion=None,
        owner_id="inline-owner",
        wants_progress=False,
        cancel_hook=None,
        work=None,
        run_in_pool=False,
        on_terminal=None,
    )


def test_explicit_lease_retains_resources_without_retaining_ambient_origin() -> None:
    bus = BaseEventBus()
    runner, gate, handles, progress, _bg = make_runner(bus=bus)
    opened: list[int] = []
    spec = replace(
        _lease_spec(),
        exclusion=ExclusionRequest(
            "device_setup", "device-a", "inline setup", "device-a"
        ),
        owner_id="device-a",
        on_opened=opened.append,
    )
    captured = EventOrigin(kind="agent", client_id="lease-client")
    with bus.origin(captured):
        lease = runner.open_lease(spec)
        assert bus.current_origin == captured
    assert bus.current_origin == EventOrigin(kind="user")
    assert handles.live_count() == 1
    assert handles.known_outcome(lease.token) is None
    assert opened == [lease.token]
    assert gate.ensure_calls == [("device_setup", "device-a")]
    assert gate.register_calls == [
        (lease.token, "device_setup", "device-a", "agent", "inline setup", "device-a")
    ]
    assert gate.release_calls == []
    assert progress.discard_calls == []
    with bus.origin(lease.origin):
        assert bus.current_origin == EventOrigin(
            kind="agent", client_id="lease-client", operation_id=str(lease.token)
        )
    lease.settle(OperationOutcome("finished"))
    assert handles.known_outcome(lease.token) == OperationOutcome("finished")
    assert handles.live_count() == 0
    assert gate.release_calls == [lease.token]
    assert bus.current_origin == EventOrigin(kind="user")


def test_inline_finishes_and_restores_origin_without_background_submission() -> None:
    bus = BaseEventBus()
    runner, gate, handles, _progress, _bg = make_runner(
        bus=bus, bg=FakeBg(fail_submit=True)
    )
    spec = replace(
        _lease_spec(),
        exclusion=ExclusionRequest("soc_connect", "soc", "connect inline"),
        owner_id="soc",
    )
    with bus.origin(EventOrigin(kind="agent", client_id="inline-client")):
        with runner.inline(spec) as lease:
            assert handles.known_outcome(lease.token) is None
            assert bus.current_origin == EventOrigin(
                kind="agent", client_id="inline-client", operation_id=str(lease.token)
            )
            assert gate.release_calls == []
        assert bus.current_origin == EventOrigin(
            kind="agent", client_id="inline-client"
        )
    assert handles.known_outcome(lease.token) == OperationOutcome("finished")
    assert handles.live_count() == 0
    assert gate.release_calls == [lease.token]
    assert bus.current_origin == EventOrigin(kind="user")


@pytest.mark.parametrize(
    "failure", [ValueError("body failed"), KeyboardInterrupt("stop")]
)
def test_inline_body_failure_settles_and_propagates_original(
    failure: BaseException,
) -> None:
    bus = BaseEventBus()
    runner, gate, handles, _progress, _bg = make_runner(bus=bus)
    spec = replace(
        _lease_spec(), exclusion=ExclusionRequest("soc_connect", "soc", "failing body")
    )
    with (
        pytest.raises(type(failure)) as caught,
        runner.inline(spec) as lease,
    ):
        raise failure
    assert caught.value is failure
    assert handles.known_outcome(lease.token) == OperationOutcome(
        "failed", repr(failure)
    )
    assert handles.live_count() == 0
    assert gate.release_calls == [lease.token]
    assert bus.current_origin == EventOrigin(kind="user")


@pytest.mark.parametrize("inline", [False, True])
@pytest.mark.parametrize("callback", ["work", "on_terminal"])
def test_sync_spec_rejects_async_policy_before_opening(
    inline: bool, callback: str
) -> None:
    runner, gate, handles, progress, _bg = make_runner()
    invoked: list[str] = []
    spec = replace(
        _lease_spec(),
        work=(lambda _factory: invoked.append("work")) if callback == "work" else None,
        on_terminal=(lambda _result, _settle: invoked.append("on_terminal"))
        if callback == "on_terminal"
        else None,
    )
    with (
        pytest.raises(ValueError, match="work.*on_terminal"),
        runner.inline(spec) if inline else nullcontext(runner.open_lease(spec)),
    ):
        invoked.append("body")
    assert invoked == []
    assert handles.live_count() == 0
    assert gate.ensure_calls == gate.register_calls == gate.release_calls == []
    assert progress.factories == {}


@pytest.mark.parametrize("stage", ["on_opened", "register", "progress"])
@pytest.mark.parametrize(
    "failure", [OSError("startup failed"), KeyboardInterrupt("stop")]
)
def test_explicit_lease_startup_failure_unwinds_opened_resources(
    stage: str, failure: BaseException, monkeypatch: pytest.MonkeyPatch
) -> None:
    bus = BaseEventBus()
    runner, gate, handles, progress, _bg = make_runner(bus=bus)
    opened: list[int] = []

    def on_opened(token: int) -> None:
        opened.append(token)
        if stage == "on_opened":
            raise failure

    def reject(*_args: object, **_kwargs: object) -> None:
        raise failure

    if stage == "register":
        monkeypatch.setattr(gate, "register", reject)
    elif stage == "progress":
        monkeypatch.setattr(progress, "make_factory", reject)
    spec = replace(
        _lease_spec(),
        exclusion=ExclusionRequest("soc_connect", "soc", "startup"),
        wants_progress=True,
        on_opened=on_opened,
    )
    with pytest.raises(type(failure)) as caught:
        runner.open_lease(spec)
    assert caught.value is failure
    token = opened[0]
    assert handles.known_outcome(token) == OperationOutcome("failed", repr(failure))
    assert handles.live_count() == 0
    assert progress.discard_calls == gate.release_calls == [token]
    assert bus.current_origin == EventOrigin(kind="user")


def test_inline_cleanup_failures_are_isolated_and_preserve_original_error(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    bus = BaseEventBus()
    steps: list[tuple[str, EventOrigin]] = []

    class FailingHandles(OperationHandles):
        def settle(self, token: int, outcome: OperationOutcome) -> None:
            steps.append(("handle", bus.current_origin))
            raise RuntimeError("settle failed")

    runner, gate, _handles, progress, _bg = make_runner(
        bus=bus, handles=FailingHandles()
    )

    def discard(_token: int) -> None:
        steps.append(("progress", bus.current_origin))
        raise RuntimeError("discard failed")

    def release(_token: int) -> None:
        steps.append(("gate", bus.current_origin))
        raise RuntimeError("release failed")

    monkeypatch.setattr(progress, "discard_operation", discard)
    monkeypatch.setattr(gate, "release", release)
    spec = replace(
        _lease_spec(),
        exclusion=ExclusionRequest("soc_connect", "soc", "cleanup"),
        wants_progress=True,
    )
    failure = ValueError("original body failure")
    with (
        pytest.raises(ValueError, match="original body failure") as caught,
        runner.inline(spec) as lease,
    ):
        raise failure
    assert caught.value is failure
    origin = EventOrigin(kind="user", operation_id=str(lease.token))
    assert steps == [("progress", origin), ("handle", origin), ("gate", origin)]
    assert lease.settled
    assert "operation progress discard failed" in caplog.text
    assert "operation handle settle failed" in caplog.text
    assert "operation gate release failed" in caplog.text
    assert bus.current_origin == EventOrigin(kind="user")


@pytest.mark.parametrize("facet", ["gate", "progress"])
def test_explicit_lease_missing_facet_rejects_before_handle_creation(
    facet: str,
) -> None:
    handles = OperationHandles()
    runner = OperationRunner(None, handles, None, FakeBg(), BaseEventBus())
    opened: list[int] = []
    spec = replace(
        _lease_spec(),
        exclusion=ExclusionRequest("soc_connect", "soc", "required gate")
        if facet == "gate"
        else None,
        wants_progress=facet == "progress",
        on_opened=opened.append,
    )
    with pytest.raises(RuntimeError, match=facet):
        runner.open_lease(spec)
    assert handles.live_count() == 0
    assert opened == []


def test_explicit_lease_conflict_does_not_open_handle() -> None:
    runner, gate, handles, progress, _bg = make_runner(gate=FakeGate(conflict=True))
    spec = replace(
        _lease_spec(), exclusion=ExclusionRequest("soc_connect", "soc", "conflict")
    )
    with pytest.raises(OperationConflictError, match="conflict"):
        runner.open_lease(spec)
    assert handles.live_count() == 0
    assert gate.register_calls == gate.release_calls == progress.discard_calls == []


def test_explicit_lease_without_facets_is_not_cancellable() -> None:
    handles = OperationHandles()
    runner = OperationRunner(
        None, handles, None, FakeBg(fail_submit=True), BaseEventBus()
    )
    lease = runner.open_lease(_lease_spec())
    assert not handles.has_cancel_hook(lease.token)
    assert lease.progress_factory is None
    handles.cancel(lease.token)
    assert handles.known_outcome(lease.token) is None
    lease.settle(OperationOutcome("finished"))
    assert handles.known_outcome(lease.token) == OperationOutcome("finished")


def test_explicit_lease_allocates_progress_until_settlement() -> None:
    runner, _gate, handles, progress, _bg = make_runner()
    lease = runner.open_lease(replace(_lease_spec(), wants_progress=True))
    assert progress.factories[lease.token] is lease.progress_factory
    assert progress.discard_calls == []
    assert handles.known_outcome(lease.token) is None
    lease.settle(OperationOutcome("finished"))
    assert progress.discard_calls == [lease.token]
    assert handles.known_outcome(lease.token) == OperationOutcome("finished")


@pytest.mark.parametrize("cancelled", [False, True])
def test_inline_cancel_request_leaves_outcome_to_domain(
    cancelled: bool, caplog: pytest.LogCaptureFixture
) -> None:
    runner, gate, handles, _progress, _bg = make_runner()
    stops: list[str] = []
    spec = replace(
        _lease_spec(),
        exclusion=ExclusionRequest("device_setup", "device", "cancel inline"),
        cancel_hook=lambda: stops.append("stop"),
    )
    with runner.inline(spec) as lease:
        handles.cancel(lease.token)
        assert stops == ["stop"]
        assert handles.known_outcome(lease.token) is None
        assert gate.release_calls == []
        if cancelled:
            lease.settle(OperationOutcome("cancelled"))
    assert handles.known_outcome(lease.token) == OperationOutcome(
        "cancelled" if cancelled else "finished"
    )
    assert gate.release_calls == [lease.token]
    assert "settle called more than once" not in caplog.text
