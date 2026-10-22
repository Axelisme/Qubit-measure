"""Cooperative async scheduling shared by recipe and workflow execution workers.

The caller owns the worker thread and passes one coroutine to run(). Only that
thread runs asyncio. Blocking operations run in the loop's executor. Controls
are marshalled onto the loop with call_soon_threadsafe and acknowledged before
returning. No control cancels the authored asyncio Task.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine, Mapping
from concurrent.futures import Future
from dataclasses import dataclass
from threading import Event, RLock, get_ident
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel

from zcu_tools.core.script_protocol import Answer, Question

from .scope import Scope, ScriptCancelled, parse_cancel_scope
from .stop_signal import CancelResult

logger = logging.getLogger(__name__)

type ScriptStatus = Literal[
    "ready", "running", "paused", "completed", "cancelled", "failed"
]
type CancelScope = str


@dataclass(frozen=True)
class CallContext:
    """One blocking invocation's cooperative controls.

    scope is this operation\'s explicit child boundary. It may be born stopped; fn must
    still start and return its partial result. epoch identifies the invocation
    attempt, for workflow owners that discard interrupted work. Do not clear or
    release this token; the host owns its lifetime.
    """

    scope: Scope
    epoch: int


@dataclass(frozen=True)
class CallResult[T]:
    """One executor completion with an immutable control snapshot.

    result is fn's unchanged payload. epoch is the attempt's starting epoch.
    partial_result snapshots cancellation of the operation or its completed
    descendant scopes, not the payload's own data completeness. stopped snapshots
    only the operation token's own stop bit. Both facts and pause have one lock
    ordering; pause after this snapshot cannot retroactively mark completion.
    """

    result: T
    epoch: int
    partial_result: bool
    stopped: bool


@dataclass(frozen=True)
class PendingQuestion:
    """The currently answerable question, detached from asyncio storage.

    request_id is unique per ask and must accompany every answer. question is
    the authored evidence model; answer_type is its declared Answer model.
    Answering twice or after cancellation fails rather than resuming a new ask.
    """

    request_id: str
    question: Question[Answer]
    answer_type: type[Answer]


class ScriptHost:
    """Run one authored coroutine on a caller-owned worker, without task cancel.

    scope is the caller-created workflow or recipe root with its catalog name.
    The host creates and releases operation children under the explicit receiver.
    on_change runs on the worker after pending-question/status changes; it must
    not block or reenter run(). It may notify an owner's condition/event.
    Constructing the host starts neither a thread nor an event loop. One host
    owns one script scope and cannot be run twice. Operation failures propagate
    through await into the author. Unhandled exceptions propagate from run and
    remain in error; script cancellation takes precedence in terminal status.
    """

    def __init__(
        self, scope: Scope, *, on_change: Callable[[], None] | None = None
    ) -> None:
        self._on_change = on_change
        self._root = scope
        self._active: Scope | None = None
        self._epoch = 0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._owner: int | None = None
        self._lock = RLock()
        self._done = Event()
        self._status: ScriptStatus = "ready"
        self._error: BaseException | None = None
        self._pending: PendingQuestion | None = None
        self._answer: asyncio.Future[Answer | None] | None = None
        self._gate: asyncio.Event | None = None
        self._paused = False
        self._root_ack: CancelResult | None = None

    @property
    def status(self) -> ScriptStatus:
        """Read ready/running/paused or the final completed/cancelled/failed state."""
        with self._lock:
            return self._status

    @property
    def error(self) -> BaseException | None:
        """Read the unhandled author/operation failure, even after script cancel."""
        with self._lock:
            return self._error

    @property
    def pending_question(self) -> PendingQuestion | None:
        """Read the answerable request, or None before/after the explicit ask."""
        with self._lock:
            return self._pending

    @property
    def root(self) -> Scope:
        """The caller-supplied root, identical to the workflow author's env."""
        return self._root

    @property
    def live_paths(self) -> list[str]:
        """Complete paths of every live leaf under the root."""
        return self._root.live_paths

    @property
    def active_path(self) -> str:
        """Read the labeled live scope path, starting at the script root."""
        return self._root.active_path

    @property
    def epoch(self) -> int:
        """Read the pause epoch; only acknowledged pause commands increment it."""
        with self._lock:
            return self._epoch

    def run[T](self, script: Coroutine[object, object, T]) -> T:
        """Drive script to completion on this calling worker and drain executors.

        Pass the authored coroutine, not a Task. Calling within a running loop
        or reusing this host raises RuntimeError. The return is the author's
        value, even after cooperative cancel; owners suppress cancelled reports.
        Finally blocks run on this worker before run returns or raises.
        """
        with self._lock:
            if self._status != "ready":
                script.close()
                raise RuntimeError("ScriptHost.run may only be called once")
            self._owner = get_ident()
            self._status = "running"
        try:
            with asyncio.Runner() as runner:
                with self._lock:
                    self._loop = runner.get_loop()
                    self._gate = asyncio.Event()
                    if not self._paused or self._root.cancelled:
                        self._gate.set()
                return runner.run(self._execute(script))
        except ScriptCancelled:
            raise
        except BaseException as error:
            with self._lock:
                self._error = error
            raise
        finally:
            with self._lock:
                self._loop = None
                self._status = (
                    "cancelled"
                    if self._root.cancelled
                    else "failed"
                    if self._error is not None
                    else "completed"
                )
            try:
                self._changed()
            finally:
                self._done.set()

    async def call[T](
        self,
        fn: Callable[[CallContext], T],
        *,
        scope: Scope | None = None,
        name: str | None = None,
        on_completed: Callable[[CallResult[T]], None] | None = None,
    ) -> CallResult[T]:
        """Await cooperative executor work unless the root has been cancelled.

        Call only from this host's authored coroutine. fn must cooperatively
        drain before returning and preserve genuine failures. Returned partial
        values belong to the operation owner, not the scheduler. Every call gets
        a fresh child token under scope (root if omitted), detached on completion.
        name explicitly labels that operation; no callable reflection occurs.
        on_completed is a framework recording hook, not author control: after
        fn returns, the owner loop invokes it once with the atomic CallResult,
        before releasing the operation or raising ScriptCancelled for root stop.
        It cannot await or reenter the host. fn failures skip the hook; fn/hook
        failures propagate unchanged and take precedence over cancellation.
        Root stop before admission raises without invoking fn or the hook.
        """
        loop = self._require_loop()
        await self.wait_until_resumed()
        if self._root.cancelled:
            raise ScriptCancelled()
        parent = self._root if scope is None else scope
        ancestor = parent
        while ancestor.parent is not None:
            ancestor = ancestor.parent
        if ancestor is not self._root:
            raise ValueError("Call scope must belong to this host")
        token = parent.branch("operation", name=name)
        self._active = token
        context = CallContext(token, self._epoch)

        def invoke() -> CallResult[T]:
            value = fn(context)
            with self._lock:
                return CallResult(
                    value, context.epoch, token.stop.partial_result, token.cancelled
                )

        try:
            result = await loop.run_in_executor(None, invoke)
            if on_completed is not None:
                on_completed(result)
            if self._root.cancelled:
                raise ScriptCancelled()
            return result
        finally:
            self._active = None
            token.release()

    async def ask[A: Answer](self, question: Question[A]) -> A:
        """Await a request-id-bound answer, raising ScriptCancelled on root stop.

        No timeout or default answer exists. A stopped script does not publish a
        question. Invalid answers leave the same request pending. Asking never
        authorizes writes. Nested/concurrent asks raise RuntimeError.
        """
        loop = self._require_loop()
        await self.wait_until_resumed()
        if self._root.cancelled:
            raise ScriptCancelled()
        if self._pending is not None:
            raise RuntimeError("A question is already pending")
        expected = question.answer_type()
        future: asyncio.Future[Answer | None] = loop.create_future()
        with self._lock:
            self._answer = future
            self._pending = PendingQuestion(uuid4().hex, question, expected)
            self._status = "paused"
        self._changed()
        try:
            answer = await future
            if answer is None or self._root.cancelled:
                raise ScriptCancelled()
            return question.parse_answer(answer)
        finally:
            with self._lock:
                self._pending = None
                self._answer = None
                self._status = "running"
            self._changed()

    def answer(self, request_id: str, answer: Mapping[str, object] | Answer) -> None:
        """Validate and deliver this exact pending request on the worker loop.

        Wrong/late/duplicate IDs raise ValueError; no active loop raises
        RuntimeError. Invalid fields raise ValidationError without consuming the
        request. An Answer instance must match the declared model (subclasses
        allowed). Return acknowledges delivery, not subsequent author completion.
        """

        def deliver() -> None:
            pending, future = self._pending, self._answer
            if (
                pending is None
                or pending.request_id != request_id
                or future is None
                or future.done()
            ):
                raise ValueError("No pending question for this request id")
            if isinstance(answer, BaseModel) and not isinstance(
                answer, pending.answer_type
            ):
                raise TypeError("Answer must match the question's declared model")
            parsed = pending.answer_type.model_validate(answer)
            pending.question.validate_answer(parsed)
            future.set_result(parsed)

        self._command(deliver)

    async def wait_until_resumed(self) -> None:
        """Wait on the worker pause gate; script cancellation also opens it.

        Workflow env calls this before delivering a naturally completed result
        or retrying an interrupted request. The host never retries fn itself.
        """
        self._require_loop()
        if self._gate is None:
            raise RuntimeError("Host gate is not initialized")
        await self._gate.wait()

    def pause(self) -> None:
        """Acknowledge operation-only stop, epoch increment, and a closed gate.

        The iteration token stays clean. In-flight call still returns one
        CallResult with its starting epoch; workflow env owns discard/retry
        policy. New async work waits for Resume; synchronous commit does not.
        No author task is cancelled.
        """

        def apply() -> None:
            with self._lock:
                self._paused = True
                self._epoch += 1
                if self._active is not None:
                    self._active.stop.set()
                if self._gate is not None and not self._root.cancelled:
                    self._gate.clear()

        self._command(apply, allow_idle=True)

    def resume(self) -> None:
        """Acknowledge opening the gate without changing epoch or scope tokens."""

        def apply() -> None:
            self._paused = False
            if self._gate is not None:
                self._gate.set()

        self._command(apply, allow_idle=True)

    def cancel(
        self, scope: str | None = None, *, expected: int | None = None
    ) -> CancelResult:
        """Acknowledge accepted cancellation, independently of author drain.

        Child selectors match on the loop. Root requests latch the bit and the
        typed root-path acknowledgement under this host's lock, then only post a
        loop notification. The first caller never waits for that notification;
        repeated requests return the same acknowledgement without posting again.
        None selects the root; expected optionally guards its birth number.
        run aliases the current operation. Root notification opens the gate and
        releases Ask; the next async boundary raises ScriptCancelled. Other
        targets leave parents running. Invalid syntax raises ValueError and
        inactive/stale targets refuse. Terminal hosts still refuse further stops.
        Completion and finally drain remain the responsibility of run()/close().
        """
        selector = self._root.kind if scope is None else scope
        label, _sequence = parse_cancel_scope(selector, expected)
        if scope is None or (
            label != "run" and label in {self._root.kind, self._root.path}
        ):
            return self._cancel_root(scope, expected=expected)

        def stop() -> CancelResult:
            if self._done.is_set():
                return CancelResult(False, "scope not active", self.active_path)
            return self._root.cancel(selector, expected=expected)

        return self._command(stop, allow_idle=True)

    def _cancel_root(
        self, selector: str | None, *, expected: int | None
    ) -> CancelResult:
        with self._lock:
            if self._done.is_set():
                return CancelResult(False, "scope not active", self.active_path)
            if selector is None:
                if expected is not None and expected != self._root.sequence:
                    return CancelResult(False, "sequence mismatch", self.active_path)
                self._root.stop.set()
            else:
                result = self._root.cancel(selector, expected=expected)
                if not result.cancelled:
                    return result
            if self._root_ack is None:
                self._root_ack = CancelResult(
                    True, "scope cancellation requested", self._root.path
                )
                if self._loop is not None:
                    self._loop.call_soon_threadsafe(self._wake_root)
            return self._root_ack

    def _wake_root(self) -> None:
        """Notify asyncio resources on their owner without reentering the author."""
        with self._lock:
            if self._gate is not None:
                self._gate.set()
            if self._answer is not None and not self._answer.done():
                self._answer.set_result(None)

    def close(self) -> None:
        """Request script stop and cooperatively drain before returning off-worker.

        On the author worker this only requests stop, avoiding self-deadlock.
        Off-worker it waits for run/finally/executor shutdown; operations must
        cooperate. Before run it latches stop without starting a worker. Repeated
        close and terminal close are harmless. No forced deadline or Task.cancel.
        """
        self.cancel()
        with self._lock:
            running = self._owner is not None and not self._done.is_set()
        if running and get_ident() != self._owner:
            self._done.wait()

    def control[T](self, fn: Callable[[], T]) -> T:
        """Run one memory-only control on the owner loop and acknowledge its result.

        fn must take only microseconds: no await, blocking, I/O or author-code
        reentry. Owner-thread calls execute inline; foreign threads enqueue with
        call_soon_threadsafe and wait for the result. Callback exceptions return
        unchanged to the caller. Before run or after loop shutdown this raises
        RuntimeError. This does not wait for an external operation to drain.
        """
        return self._command(fn)

    async def _execute[T](self, script: Coroutine[object, object, T]) -> T:
        try:
            return await script
        finally:
            # Stop accepting commands before the loop's final drain. Previously
            # accepted callbacks are queued before run_until_complete's stop.
            with self._lock:
                self._loop = None

    def _changed(self) -> None:
        if self._on_change is not None:
            self._on_change()

    def _require_loop(self) -> asyncio.AbstractEventLoop:
        loop = asyncio.get_running_loop()
        if loop is not self._loop or get_ident() != self._owner:
            raise RuntimeError("Host await must run on its caller-owned worker loop")
        return loop

    def _command[T](self, command: Callable[[], T], *, allow_idle: bool = False) -> T:
        with self._lock:
            loop = self._loop
            if loop is None:
                if allow_idle:
                    return command()
                raise RuntimeError("Script host has no active loop")
            if get_ident() == self._owner:
                return command()
            ack: Future[T] = Future()

            def apply() -> None:
                try:
                    value = command()
                except BaseException as error:
                    # The control caller owns this failure, not the author task.
                    logger.exception("Script control callback failed")
                    ack.set_exception(error)
                else:
                    ack.set_result(value)

            loop.call_soon_threadsafe(apply)
        return ack.result()
