"""Tests for OperationRunner — kind-agnostic lifecycle mechanism (ADR-0026 §1).

Verifies the mechanism contract with fake gate/handles/progress/bg:
- exclusion present / absent (ensure+register/release called or skipped)
- wants_progress present / absent (make_factory/discard called or skipped,
  and minted factory is passed to work)
- submit-fail → settle(failed) does full unwind (discard+settle+release) then raises
- on_terminal receives correct BgResult (ok/result vs error/NO_RESULT) + usable settle
- settle call-once (policy calling twice logs and only applies once)
- ensure_can_start raises → no create, no settle, propagates directly
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.background import NO_RESULT
from zcu_tools.gui.session.operation_handles import OperationHandles, OperationOutcome
from zcu_tools.gui.session.operation_runner import (
    BgResult,
    ExclusionRequest,
    OperationRunner,
    OperationSpec,
    SettleFn,
)
from zcu_tools.gui.session.ports import OperationConflictError

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeGate:
    """Minimal ExclusionGate: records calls, optionally raises on ensure."""

    def __init__(self, *, conflict: bool = False) -> None:
        self._conflict = conflict
        self.ensure_calls: list[tuple[str, str | None]] = []
        self.register_calls: list[tuple[int, str, str, str | None]] = []
        self.release_calls: list[int] = []

    def ensure_can_start(self, kind: str, *, resource_id: str | None = None) -> None:
        self.ensure_calls.append((kind, resource_id))
        if self._conflict:
            raise OperationConflictError("conflict")

    def register(
        self, token: int, kind: str, *, owner_id: str, resource_id: str | None = None
    ) -> None:
        self.register_calls.append((token, kind, owner_id, resource_id))

    def release(self, token: int) -> None:
        self.release_calls.append(token)

    def is_device_mutating(self, name: str) -> bool:
        return False


class _FakeBg:
    """Synchronous background executor stub.

    When ``fail_submit`` is True, submit() raises immediately (simulating a
    synchronous submit failure before any work starts).
    Otherwise, submit captures the callbacks; call ``deliver_result`` or
    ``deliver_error`` to drive the on_done / on_error path.
    """

    def __init__(self, *, fail_submit: bool = False) -> None:
        self._fail_submit = fail_submit
        self._on_done: Callable[[Any], None] | None = None
        self._on_error: Callable[[Exception], None] | None = None

    def submit(
        self,
        work: Callable[[], Any],
        *,
        run_in_pool: bool,
        on_done: Callable[[Any], None],
        on_error: Callable[[Exception], None],
    ) -> None:
        if self._fail_submit:
            raise RuntimeError("submit boom")
        # Keep work and callbacks so tests can drive them.
        self._work = work
        self._on_done = on_done
        self._on_error = on_error

    def deliver_result(self) -> None:
        """Execute the captured work thunk and pass its return value to on_done."""
        assert self._work is not None and self._on_done is not None
        result = self._work()
        self._on_done(result)

    def deliver_error(self, exc: Exception) -> None:
        assert self._on_error is not None
        self._on_error(exc)


class _FakeProgress:
    """Tracks make_factory and discard_operation calls."""

    def __init__(self, *, fail_discard: bool = False) -> None:
        self._fail_discard = fail_discard
        self.factories: dict[int, MagicMock] = {}
        self.discard_calls: list[int] = []

    def make_factory(self, operation_id: int, owner_id: str) -> MagicMock:
        factory = MagicMock(name=f"factory-{operation_id}")
        self.factories[operation_id] = factory
        return factory

    def discard_operation(self, operation_id: int) -> None:
        self.discard_calls.append(operation_id)
        if self._fail_discard:
            raise RuntimeError("discard boom")


def _make_runner(
    *,
    gate: _FakeGate | None = None,
    handles: OperationHandles | None = None,
    progress: _FakeProgress | None = None,
    bg: _FakeBg | None = None,
) -> tuple[OperationRunner, _FakeGate, OperationHandles, _FakeProgress, _FakeBg]:
    g = gate or _FakeGate()
    h = handles or OperationHandles()
    p = progress or _FakeProgress()
    b = bg or _FakeBg()
    runner = OperationRunner(g, h, p, b)  # type: ignore[arg-type]
    return runner, g, h, p, b


def _noop_spec(
    *,
    gate: _FakeGate,
    wants_progress: bool = False,
    exclusion: ExclusionRequest | None = None,
    fail_submit: bool = False,
) -> OperationSpec:
    """Minimal spec: work returns sentinel, on_terminal captures bg result."""
    terminal_results: list[BgResult] = []

    def on_terminal(bg: BgResult, settle: SettleFn) -> None:
        terminal_results.append(bg)
        settle(OperationOutcome("finished"))

    return OperationSpec(
        exclusion=exclusion,
        owner_id="owner",
        wants_progress=wants_progress,
        cancel_hook=None,
        work=lambda factory: object(),
        run_in_pool=False,
        on_terminal=on_terminal,
    )


# ---------------------------------------------------------------------------
# ensure + register / release calls with exclusion
# ---------------------------------------------------------------------------


def test_exclusion_present_calls_ensure_register_release():
    runner, gate, handles, progress, bg = _make_runner()
    excl = ExclusionRequest(kind="run", owner_id="tab1", resource_id=None)

    collected_settle: list[SettleFn] = []

    def on_terminal(bgr: BgResult, settle: SettleFn) -> None:
        collected_settle.append(settle)
        settle(OperationOutcome("finished"))

    spec = OperationSpec(
        exclusion=excl,
        owner_id="tab1",
        wants_progress=False,
        cancel_hook=None,
        work=lambda _f: None,
        run_in_pool=False,
        on_terminal=on_terminal,
    )
    token = runner.begin(spec)
    bg.deliver_result()

    assert len(gate.ensure_calls) == 1
    assert gate.ensure_calls[0] == ("run", None)
    assert len(gate.register_calls) == 1
    assert gate.register_calls[0] == (token, "run", "tab1", None)
    assert gate.release_calls == [token]


def test_exclusion_absent_skips_ensure_register_release():
    runner, gate, handles, progress, bg = _make_runner()

    def on_terminal(bgr: BgResult, settle: SettleFn) -> None:
        settle(OperationOutcome("finished"))

    spec = OperationSpec(
        exclusion=None,
        owner_id="tab1",
        wants_progress=False,
        cancel_hook=None,
        work=lambda _f: None,
        run_in_pool=False,
        on_terminal=on_terminal,
    )
    runner.begin(spec)
    bg.deliver_result()

    assert gate.ensure_calls == []
    assert gate.register_calls == []
    assert gate.release_calls == []


def test_exclusion_with_resource_id_passes_resource_id_to_gate():
    runner, gate, handles, progress, bg = _make_runner()
    excl = ExclusionRequest(kind="device_setup", owner_id="dev1", resource_id="dev1")

    def on_terminal(bgr: BgResult, settle: SettleFn) -> None:
        settle(OperationOutcome("finished"))

    spec = OperationSpec(
        exclusion=excl,
        owner_id="dev1",
        wants_progress=False,
        cancel_hook=None,
        work=lambda _f: None,
        run_in_pool=False,
        on_terminal=on_terminal,
    )
    token = runner.begin(spec)
    bg.deliver_result()

    assert gate.ensure_calls[0] == ("device_setup", "dev1")
    assert gate.register_calls[0] == (token, "device_setup", "dev1", "dev1")
    assert gate.release_calls == [token]


# ---------------------------------------------------------------------------
# wants_progress: make_factory / discard and factory injection into work
# ---------------------------------------------------------------------------


def test_wants_progress_calls_make_factory_and_discard():
    runner, gate, handles, progress, bg = _make_runner()

    received_factory: list[Any] = []

    def on_terminal(bgr: BgResult, settle: SettleFn) -> None:
        settle(OperationOutcome("finished"))

    spec = OperationSpec(
        exclusion=None,
        owner_id="tab1",
        wants_progress=True,
        cancel_hook=None,
        work=lambda factory: received_factory.append(factory),
        run_in_pool=False,
        on_terminal=on_terminal,
    )
    token = runner.begin(spec)
    bg.deliver_result()

    # make_factory was called for this token
    assert token in progress.factories
    # The factory was passed into work (captured above)
    assert len(received_factory) == 1
    assert received_factory[0] is progress.factories[token]
    # discard was called on terminal
    assert progress.discard_calls == [token]


def test_wants_progress_false_skips_make_factory_and_discard():
    runner, gate, handles, progress, bg = _make_runner()

    received_factory: list[Any] = []

    def on_terminal(bgr: BgResult, settle: SettleFn) -> None:
        settle(OperationOutcome("finished"))

    spec = OperationSpec(
        exclusion=None,
        owner_id="tab1",
        wants_progress=False,
        cancel_hook=None,
        work=lambda factory: received_factory.append(factory),
        run_in_pool=False,
        on_terminal=on_terminal,
    )
    runner.begin(spec)
    bg.deliver_result()

    assert progress.factories == {}
    # factory passed to work is None
    assert received_factory == [None]
    assert progress.discard_calls == []


# ---------------------------------------------------------------------------
# submit-fail → settle(failed) unwinds (discard+settle+release) then re-raises
# ---------------------------------------------------------------------------


def test_submit_fail_calls_settle_failed_and_unwinds():
    bg = _FakeBg(fail_submit=True)
    runner, gate, handles, progress, _ = _make_runner(bg=bg)
    excl = ExclusionRequest(kind="run", owner_id="tab1")

    terminal_calls: list[BgResult] = []

    def on_terminal(bgr: BgResult, settle: SettleFn) -> None:
        terminal_calls.append(bgr)
        settle(OperationOutcome("finished"))

    spec = OperationSpec(
        exclusion=excl,
        owner_id="tab1",
        wants_progress=True,
        cancel_hook=None,
        work=lambda _f: None,
        run_in_pool=False,
        on_terminal=on_terminal,
    )

    with pytest.raises(RuntimeError, match="submit boom"):
        runner.begin(spec)

    # on_terminal is NOT called (submit-fail settle is runner-internal)
    assert terminal_calls == []
    # progress discarded, gate released (full unwind)
    assert len(progress.discard_calls) == 1
    assert len(gate.release_calls) == 1
    # The handle was settled failed
    settled = handles.poll(gate.register_calls[0][0])
    assert settled is not None and settled.status == "failed"


# ---------------------------------------------------------------------------
# on_terminal receives correct BgResult
# ---------------------------------------------------------------------------


def test_on_terminal_receives_ok_result_on_done():
    runner, gate, handles, progress, bg = _make_runner()

    received: list[BgResult] = []

    def on_terminal(bgr: BgResult, settle: SettleFn) -> None:
        received.append(bgr)
        settle(OperationOutcome("finished"))

    sentinel = object()
    spec = OperationSpec(
        exclusion=None,
        owner_id="tab1",
        wants_progress=False,
        cancel_hook=None,
        work=lambda _f: sentinel,
        run_in_pool=False,
        on_terminal=on_terminal,
    )
    runner.begin(spec)
    bg.deliver_result()

    assert len(received) == 1
    assert received[0].ok is True
    assert received[0].result is sentinel
    assert received[0].error is None


def test_on_terminal_receives_error_result_on_error():
    runner, gate, handles, progress, bg = _make_runner()

    received: list[BgResult] = []

    def on_terminal(bgr: BgResult, settle: SettleFn) -> None:
        received.append(bgr)
        settle(OperationOutcome("failed", str(bgr.error)))

    exc = RuntimeError("boom")
    spec = OperationSpec(
        exclusion=None,
        owner_id="tab1",
        wants_progress=False,
        cancel_hook=None,
        work=lambda _f: (_ for _ in ()).throw(exc),  # raises
        run_in_pool=False,
        on_terminal=on_terminal,
    )
    runner.begin(spec)
    bg.deliver_error(exc)

    assert len(received) == 1
    assert received[0].ok is False
    assert received[0].result is NO_RESULT
    assert received[0].error is exc


def test_terminal_callback_exception_settles_failed_and_releases(caplog):
    runner, gate, handles, progress, bg = _make_runner()
    excl = ExclusionRequest(kind="run", owner_id="tab1")

    def on_terminal(_bgr: BgResult, _settle: SettleFn) -> None:
        raise RuntimeError("terminal boom")

    spec = OperationSpec(
        exclusion=excl,
        owner_id="tab1",
        wants_progress=True,
        cancel_hook=None,
        work=lambda _f: None,
        run_in_pool=False,
        on_terminal=on_terminal,
    )
    token = runner.begin(spec)

    with caplog.at_level("ERROR"):
        bg.deliver_result()

    settled = handles.poll(token)
    assert settled is not None
    assert settled.status == "failed"
    assert settled.error == "terminal boom"
    assert progress.discard_calls == [token]
    assert gate.release_calls == [token]
    assert "operation terminal callback failed" in caplog.text


def test_progress_discard_exception_still_settles_and_releases(caplog):
    progress = _FakeProgress(fail_discard=True)
    runner, gate, handles, _, bg = _make_runner(progress=progress)
    excl = ExclusionRequest(kind="run", owner_id="tab1")

    def on_terminal(_bgr: BgResult, settle: SettleFn) -> None:
        settle(OperationOutcome("finished"))

    spec = OperationSpec(
        exclusion=excl,
        owner_id="tab1",
        wants_progress=True,
        cancel_hook=None,
        work=lambda _f: None,
        run_in_pool=False,
        on_terminal=on_terminal,
    )
    token = runner.begin(spec)

    with caplog.at_level("ERROR"):
        bg.deliver_result()

    settled = handles.poll(token)
    assert settled is not None and settled.status == "finished"
    assert progress.discard_calls == [token]
    assert gate.release_calls == [token]
    assert "operation progress discard failed" in caplog.text


# ---------------------------------------------------------------------------
# settle call-once guard
# ---------------------------------------------------------------------------


def test_settle_call_once_second_call_is_logged_noop(caplog):
    runner, gate, handles, progress, bg = _make_runner(gate=_FakeGate())
    excl = ExclusionRequest(kind="run", owner_id="tab1")

    settle_fn: list[SettleFn] = []

    def on_terminal(bgr: BgResult, settle: SettleFn) -> None:
        settle_fn.append(settle)
        settle(OperationOutcome("finished"))
        # Second call — should be a no-op (not double-settle, not double-release)
        settle(OperationOutcome("finished"))

    spec = OperationSpec(
        exclusion=excl,
        owner_id="tab1",
        wants_progress=True,
        cancel_hook=None,
        work=lambda _f: None,
        run_in_pool=False,
        on_terminal=on_terminal,
    )
    token = runner.begin(spec)
    with caplog.at_level("ERROR"):
        bg.deliver_result()

    # release called exactly once despite two settle() calls in on_terminal
    assert gate.release_calls.count(token) == 1
    # discard called exactly once
    assert progress.discard_calls.count(token) == 1
    # handle settled exactly once (OperationHandles.settle is idempotent, but
    # the call-once guard prevents the second attempt reaching the handles layer)
    settled = handles.poll(token)
    assert settled is not None and settled.status == "finished"
    assert "operation settle called more than once" in caplog.text
    assert f"token={token}" in caplog.text
    assert "owner=tab1" in caplog.text


# ---------------------------------------------------------------------------
# ensure_can_start raises → no create, no settle, propagates directly
# ---------------------------------------------------------------------------


def test_ensure_raises_does_not_create_handle_or_call_settle():
    gate = _FakeGate(conflict=True)
    runner, _, handles, progress, bg = _make_runner(gate=gate)
    excl = ExclusionRequest(kind="run", owner_id="tab1")

    terminal_calls: list[BgResult] = []

    def on_terminal(bgr: BgResult, settle: SettleFn) -> None:
        terminal_calls.append(bgr)

    spec = OperationSpec(
        exclusion=excl,
        owner_id="tab1",
        wants_progress=True,
        cancel_hook=None,
        work=lambda _f: None,
        run_in_pool=False,
        on_terminal=on_terminal,
    )

    with pytest.raises(OperationConflictError):
        runner.begin(spec)

    # No handle was minted (no token created)
    assert handles.live_count() == 0
    # No settle, no register, no discard, no release
    assert gate.register_calls == []
    assert gate.release_calls == []
    assert progress.discard_calls == []
    assert progress.factories == {}
    assert terminal_calls == []
