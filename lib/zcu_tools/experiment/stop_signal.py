"""Cancellation and partial-result errors shared by experiment run callers."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Literal, TypeAlias

ErrorStatus: TypeAlias = Literal["interrupted", "failed"]
ScopeLabel: TypeAlias = Literal[
    "script", "run", "iteration", "operation", "schedule", "acquire"
]


@dataclass(frozen=True)
class CancelResult:
    """Acknowledged scope request, not a promise that work has drained.

    cancelled means the owner accepted the request, not that work drained.
    Token owners atomically match/set the target. Workflow bare script Stop
    instead records and latches intent before signaling a still-live host;
    acceptance survives host exit before owner finalization. reason is scope
    cancellation requested, scope not active, sequence mismatch, or the recipe
    admission refusal no run in progress. active_path is the path at the check.
    """

    cancelled: bool
    reason: str
    active_path: str


def parse_cancel_scope(
    scope: object, expected: object = None
) -> tuple[ScopeLabel, int | None]:
    """Validate a label or label#N and return its label and expected sequence.

    N and expected must be positive integers; bool is not an integer here.
    Conflicting sequences, unknown labels and malformed segments raise ValueError.
    A root's sequence is checked by the resolver, not by this syntax parser.
    """
    if expected is not None and (
        not isinstance(expected, int) or isinstance(expected, bool) or expected <= 0
    ):
        raise ValueError("expected must be a positive integer")
    if not isinstance(scope, str):
        raise ValueError("scope must be a label or label#N")
    label, separator, suffix = scope.partition("#")
    if separator:
        if not suffix.isascii() or not suffix.isdecimal() or int(suffix) <= 0:
            raise ValueError("scope sequence must be a positive integer")
        sequence = int(suffix)
        if expected is not None and expected != sequence:
            raise ValueError("scope sequence conflicts with expected")
        expected = sequence
    match label:
        case "script" | "run" | "iteration" | "operation" | "schedule" | "acquire":
            return label, expected
        case _:
            raise ValueError(f"Unknown stop scope label: {label!r}")


class ScheduleOutcomeError(RuntimeError):
    """Exception wrapper for a Schedule failure that returned partial data."""

    def __init__(
        self,
        status: ErrorStatus,
        reason: str,
        exception: BaseException | None,
    ) -> None:
        super().__init__(reason)
        self.status = status
        self.reason = reason
        self.exception = exception


class StopSignal:
    """Thread-safe scope tree separating cancellation from completion facts.

    label names the scope's control boundary. A supplied event is borrowed as
    this token's stop bit; descendants own independent events. Completed child
    facts travel upward on release, without stopping the parent, and stop at
    operation boundaries. Constructing a token does not attach it to a tree.
    """

    def __init__(
        self, event: threading.Event | None = None, *, label: ScopeLabel = "operation"
    ) -> None:
        if label not in (
            "script",
            "run",
            "iteration",
            "operation",
            "schedule",
            "acquire",
        ):
            raise ValueError(f"Unknown stop scope label: {label!r}")
        self._label: ScopeLabel = label
        self._sequence = 0
        self._completed_partial = False
        self._event = event if event is not None else threading.Event()
        self._error: ScheduleOutcomeError | None = None
        self._lock = threading.RLock()
        self._parent: StopSignal | None = None
        self._children: dict[StopSignal, None] = {}
        self._child_sequences: dict[ScopeLabel, int] = {}

    @property
    def label(self) -> ScopeLabel:
        """Read script/run/iteration/operation/schedule/acquire scope identity."""
        return self._label

    @property
    def sequence(self) -> int:
        """Read the one-based per-parent/per-label birth number, or 0 for a root."""
        return self._sequence

    @property
    def active_child(self) -> StopSignal | None:
        """Read the newest attached child, or None after all children release."""
        with self._lock:
            return next(reversed(self._children), None)

    @property
    def active_path(self) -> str:
        """Project this subtree as label/label#sequence along active_child.

        The starting token has no suffix. Releasing the newest child restores
        the previous live sibling; released scopes never appear in this path.
        """
        with self._lock:
            segments = [self.label]
            child = self.active_child
            while child is not None:
                segments.append(f"{child.label}#{child.sequence}")
                child = child.active_child
            return "/".join(segments)

    def cancel(self, scope: str, *, expected: int | None = None) -> CancelResult:
        """Atomically match the active path and set one scope and its descendants.

        scope is a label or label#N. run aliases operation unless this tree has
        a run root. Missing targets and stale sequences return a refusal with
        the current path; malformed input raises ValueError. Roots have no
        sequence. Child creation/release shares this check's tree lock.
        """
        label, sequence = parse_cancel_scope(scope, expected)
        if label == "run" and self.label != "run":
            label = "operation"
        with self._lock:
            path = self.active_path
            token: StopSignal | None = self
            while token is not None and token.label != label:
                token = token.active_child
            if token is None:
                return CancelResult(False, "scope not active", path)
            if sequence is not None and (
                token.sequence == 0 or token.sequence != sequence
            ):
                return CancelResult(False, "sequence mismatch", path)
            token.set()
            return CancelResult(True, "scope cancellation requested", path)

    @property
    def partial_result(self) -> bool:
        """Read this token's stop bit or a released child's partial completion.

        Child completion does not set this token's event. Active child facts
        are delivered only when that child releases.
        """
        with self._lock:
            return self.is_set() or self._completed_partial

    def is_set(self) -> bool:
        return self._event.is_set()

    def set(self) -> None:
        """Set this scope and all attached descendants, never its parent."""
        with self._lock:
            self._event.set()
            for child in self._children:
                child.set()

    def child(self, *, label: ScopeLabel = "operation") -> StopSignal:
        """Attach a scope inheriting this scope's stop/error, including at birth.

        label selects the control boundary and its per-label birth counter.
        Stops travel down immediately. Errors and partial completion travel up
        only on release, stopping at operation boundaries. Call release after
        the scope drains so completed work is not retained by the tree.
        """
        with self._lock:
            child = StopSignal(label=label)
            sequence = self._child_sequences.get(label, 0) + 1
            self._child_sequences[label] = sequence
            child._sequence = sequence
            child._lock = self._lock
            child._parent = self
            child._error = self._error
            if self.is_set():
                child.set()
            self._children[child] = None
            return child

    def release(self) -> None:
        """Deliver completion facts once, then detach without stopping the parent.

        Preserve the parent's first error and accumulate partial completion.
        An operation's caller consumes its facts, so releasing an operation
        never hands them to its iteration. Repeated release is a no-op.
        Future parent stops cannot reach a detached scope.
        """
        with self._lock:
            parent = self._parent
            if parent is not None:
                if self.label != "operation":
                    if parent._error is None:
                        parent._error = self._error
                    parent._completed_partial |= self.partial_result
                parent._children.pop(self)
                self._parent = None

    def clear_stop(self) -> None:
        with self._lock:
            self._error = None
        self._event.clear()

    @property
    def event(self) -> threading.Event:
        return self._event

    @property
    def error(self) -> ScheduleOutcomeError | None:
        with self._lock:
            return self._error

    def set_error(
        self,
        status: ErrorStatus,
        reason: str,
        exception: BaseException | None,
    ) -> None:
        """Record this scope's first error and stop it and its descendants.

        Ancestor and sibling stop/error state is unchanged. The stored error
        retains status, reason and exception for raise_if_error().
        """
        error = ScheduleOutcomeError(status, reason, exception)
        with self._lock:
            if self._error is None:
                self._error = error
            self.set()

    def raise_if_error(self) -> None:
        error = self.error
        if error is None:
            return
        if error.exception is not None:
            raise error from error.exception
        raise error
