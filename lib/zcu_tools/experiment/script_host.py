"""Cooperative async scheduling shared by recipe and workflow execution workers.

The caller owns the worker thread and passes one coroutine to run(). Only that
thread runs asyncio. Blocking operations run in the loop's executor. Controls
are marshalled onto the loop with call_soon_threadsafe and acknowledged before
returning. No control cancels the authored asyncio Task.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine, Mapping
from concurrent.futures import Future
from dataclasses import dataclass
from threading import Event, RLock, get_ident
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel

from zcu_tools.core.script_protocol import Answer, Question

from .stop_signal import StopSignal

type ScriptStatus = Literal["ready", "running", "paused", "completed", "cancelled", "failed"]
type CancelScope = Literal["script", "run"]


@dataclass(frozen=True)
class CallContext:
    """One blocking invocation's cooperative controls.

    stop_signal is this operation's child scope. It may be born stopped; fn must
    still start and return its partial result. epoch identifies the invocation
    attempt, for workflow owners that discard interrupted work. Do not clear or
    release this token; the host owns its lifetime.
    """

    stop_signal: StopSignal
    epoch: int


@dataclass(frozen=True)
class CallResult[T]:
    """One executor completion with an immutable control snapshot.

    result is fn's unchanged payload. epoch is the attempt's starting epoch.
    partial_result snapshots the operation stop bit as fn completes, not the
    payload's own data completeness. Snapshot and pause have one lock ordering;
    pause after this snapshot cannot retroactively mark natural completion.
    """

    result: T
    epoch: int
    partial_result: bool


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

    on_change runs on the worker after pending-question/status changes; it must
    not block or reenter run(). It may notify an owner's condition/event.
    Constructing the host starts neither a thread nor an event loop. One host
    owns one script scope and cannot be run twice. Operation failures propagate
    through await into the author. Unhandled exceptions propagate from run and
    remain in error; script cancellation takes precedence in terminal status.
    """

    def __init__(self, *, on_change: Callable[[], None] | None = None) -> None:
        self._on_change = on_change
        self._root = StopSignal()
        self._iteration = self._root.child()
        self._active: StopSignal | None = None
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
                    if not self._paused or self._root.is_set():
                        self._gate.set()
                return runner.run(script)
        except BaseException as error:
            with self._lock:
                self._error = error
            raise
        finally:
            with self._lock:
                self._loop = None
                self._status = (
                    "cancelled" if self._root.is_set()
                    else "failed" if self._error is not None else "completed"
                )
                self._iteration.release()
            self._changed()
            self._done.set()

    async def call[T](self, fn: Callable[[CallContext], T]) -> CallResult[T]:
        """Await fn(ctx) in an executor, always starting even with a stopped token.

        Call only from this host's authored coroutine. fn must cooperatively
        drain before returning and preserve genuine failures. Returned partial
        values belong to the operation owner, not the scheduler. Every call gets
        a fresh child token which is detached when that invocation completes.
        """
        loop = self._require_loop()
        await self.wait_until_resumed()
        token = self._iteration.child()
        self._active = token
        context = CallContext(token, self._epoch)

        def invoke() -> CallResult[T]:
            value = fn(context)
            with self._lock:
                return CallResult(value, context.epoch, token.is_set())

        try:
            return await loop.run_in_executor(None, invoke)
        finally:
            self._active = None
            token.release()

    async def ask[A: Answer](self, question: Question[A]) -> A | None:
        """Await a request-id-bound validated answer, or None after script cancel.

        No timeout or default answer exists. A stopped script does not publish a
        question. Invalid answers leave the same request pending. Asking never
        authorizes writes. Nested/concurrent asks raise RuntimeError.
        """
        loop = self._require_loop()
        if self._root.is_set():
            return None
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
            return None if answer is None else expected.model_validate(answer)
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
            if pending is None or pending.request_id != request_id or future is None or future.done():
                raise ValueError("No pending question for this request id")
            if isinstance(answer, BaseModel) and not isinstance(answer, pending.answer_type):
                raise TypeError("Answer must match the question's declared model")
            parsed = pending.answer_type.model_validate(answer)
            future.set_result(parsed)
        self._command(deliver)

    async def commit(self) -> None:
        """Wait for the pause gate, then detach and replace the iteration scope.

        Recipe callers do not use this method. Script cancellation opens the
        gate, so cleanup can commit without waiting for a subsequent Resume.
        """
        await self.wait_until_resumed()
        self._iteration.release()
        self._iteration = self._root.child()

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
        policy. New calls and commit wait for Resume. No author task is cancelled.
        """
        def apply() -> None:
            with self._lock:
                self._paused = True
                self._epoch += 1
                if self._active is not None:
                    self._active.set()
                if self._gate is not None and not self._root.is_set():
                    self._gate.clear()
        self._command(apply, allow_idle=True)

    def resume(self) -> None:
        """Acknowledge opening the gate without changing epoch or scope tokens."""
        def apply() -> None:
            self._paused = False
            if self._gate is not None:
                self._gate.set()
        self._command(apply, allow_idle=True)

    def cancel(self, *, scope: CancelScope = "script") -> None:
        """Acknowledge cooperative script or current-operation stop on the loop.

        Script sets the root, releases a pending ask with None, and causes later
        operations to inherit stop. Run sets only the active operation token;
        subsequent calls are unaffected. No active run is a no-op. This method
        never waits for operation drain and never cancels the author Task.
        Before run, script cancellation is retained; after terminal it is a no-op.
        """
        if scope not in ("script", "run"):
            raise ValueError("scope must be script or run")
        def stop() -> None:
            if scope == "script":
                self._root.set()
                if self._gate is not None:
                    self._gate.set()
                if self._answer is not None and not self._answer.done():
                    self._answer.set_result(None)
            elif self._active is not None:
                self._active.set()
        self._command(stop, allow_idle=True)

    def close(self) -> None:
        """Request script stop and cooperatively drain before returning off-worker.

        On the author worker this only requests stop, avoiding self-deadlock.
        Off-worker it waits for run/finally/executor shutdown; operations must
        cooperate. Before run it latches stop without starting a worker. Repeated
        close and terminal close are harmless. No forced deadline or Task.cancel.
        """
        self.cancel()
        with self._lock:
            running = self._loop is not None
        if running and get_ident() != self._owner:
            self._done.wait()

    def _changed(self) -> None:
        if self._on_change is not None:
            self._on_change()

    def _require_loop(self) -> asyncio.AbstractEventLoop:
        loop = asyncio.get_running_loop()
        if loop is not self._loop or get_ident() != self._owner:
            raise RuntimeError("Host await must run on its caller-owned worker loop")
        return loop

    def _command(self, command: Callable[[], None], *, allow_idle: bool = False) -> None:
        with self._lock:
            loop = self._loop
            if loop is None:
                if not allow_idle:
                    raise RuntimeError("Script host has no active loop")
                if not self._done.is_set():
                    command()
                return
            if get_ident() == self._owner:
                command()
                return
            ack: Future[None] = Future()
            def apply() -> None:
                try:
                    command()
                except Exception as error:
                    ack.set_exception(error)
                else:
                    ack.set_result(None)
            loop.call_soon_threadsafe(apply)
        ack.result()
