"""Explicit experiment control scopes; stop bits remain owned by StopSignal."""

from __future__ import annotations

from threading import RLock
from types import TracebackType
from typing import Literal, Self

from .stop_signal import CancelResult, StopSignal

type RootKind = Literal["workflow", "recipe", "run"]
type BranchKind = Literal["iteration", "operation", "schedule", "acquire"]
_RESERVED = {
    "workflow",
    "recipe",
    "run",
    "iteration",
    "operation",
    "schedule",
    "acquire",
}


class ScriptCancelled(Exception):
    """A workflow or recipe root was cancelled; authors finish through finally."""


def parse_cancel_scope(
    scope: object, expected: object = None
) -> tuple[str, int | None]:
    """Parse a kind, kind#N, operation:name, or full path selector.

    expected is an optional positive birth sequence, never bool. Conflicting or
    malformed sequences raise ValueError. Full paths are validated by exact
    matching against the live tree, without interpreting catalog names.
    """
    if expected is not None and (
        not isinstance(expected, int) or isinstance(expected, bool) or expected <= 0
    ):
        raise ValueError("expected must be a positive integer")
    if not isinstance(scope, str) or not scope:
        raise ValueError("scope must be a nonempty selector")
    if "/" in scope:
        return scope, expected
    label, separator, suffix = scope.partition("#")
    if separator:
        if not suffix.isascii() or not suffix.isdecimal() or int(suffix) <= 0:
            raise ValueError("scope sequence must be a positive integer")
        sequence = int(suffix)
        if expected is not None and expected != sequence:
            raise ValueError("scope sequence conflicts with expected")
        expected = sequence
    if ":" in label and not label.startswith("operation:"):
        raise ValueError("Only operation:name is a named scope selector")
    if not label or label == "operation:":
        raise ValueError("scope label must not be empty")
    return label, expected


def _validate_name(name: str | None) -> None:
    if name is not None and (not name or any(char in name for char in "#/:")):
        raise ValueError("Scope name must be nonempty and must not contain #, / or :")


class Scope:
    """One explicit control boundary with child lifetimes and subtree selectors.

    kind is workflow, recipe or run for a root; name is the caller's catalog
    name (run normally has none). A non-None name must be nonempty and contain
    no #, / or :. Invalid kinds/names raise ValueError. stop optionally borrows
    a caller-owned token.
    Children inherit stop bits at birth. Cancellation travels down, while error
    and partial completion travel up on release and stop at operation children.
    Construction/child attach immediately; with or release closes the boundary.
    No current/latest child is registered. All tree operations are thread-safe.
    """

    def __init__(
        self, kind: RootKind, name: str | None = None, stop: StopSignal | None = None
    ) -> None:
        if kind not in {"workflow", "recipe", "run"}:
            raise ValueError("Root scope kind must be workflow, recipe or run")
        _validate_name(name)
        self._kind: str = kind
        self._name = name
        self._sequence = 0
        self._stop = stop if stop is not None else StopSignal()
        self._parent: Scope | None = None
        self._children: dict[Scope, None] = {}
        self._sequences: dict[str, int] = {}
        self._lock = RLock()
        self._closed = False

    @property
    def kind(self) -> str:
        """The root kind, built-in branch kind, or authored child label."""
        return self._kind

    @property
    def name(self) -> str | None:
        """The explicit catalog/operation/schedule name, or None when unnamed."""
        return self._name

    @property
    def sequence(self) -> int:
        """One-based birth number per parent/kind; roots use zero."""
        return self._sequence

    @property
    def stop(self) -> StopSignal:
        """The downward bit/error capability for low-level cooperative APIs."""
        return self._stop

    @property
    def cancelled(self) -> bool:
        """Whether this scope's own downward stop bit is set."""
        return self.stop.is_set()

    @property
    def path(self) -> str:
        """This boundary's full identity, without any descendant projection."""
        segment = self.kind + (f":{self.name}" if self.name is not None else "")
        if self.sequence:
            segment += f"#{self.sequence}"
        return f"{self._parent.path}/{segment}" if self._parent is not None else segment

    @property
    def active_path(self) -> str:
        """Full path down the unique chain, stopping at the first live fork."""
        with self._lock:
            node = self
            while len(node._children) == 1:
                node = next(iter(node._children))
            return node.path

    @property
    def live_paths(self) -> list[str]:
        """Full paths of every live leaf, in child birth order (self if empty)."""
        with self._lock:
            return [node.path for node in self._walk() if not node._children]

    def child(self, label: str) -> Scope:
        """Attach an authored label, rejecting reserved kinds and #, / or :.

        Empty labels raise ValueError. A released parent raises RuntimeError.
        Use the result in with, or explicitly release after work drains.
        """
        if not label or any(char in label for char in "#/:") or label in _RESERVED:
            raise ValueError("Invalid or reserved author scope label")
        return self._attach(label, None)

    def branch(self, kind: BranchKind, *, name: str | None = None) -> Scope:
        """Attach a framework iteration/operation/schedule/acquire boundary.

        name explicitly labels an operation or schedule; no callable reflection
        occurs. Other kinds reject names. Release after descendants drain.
        """
        if kind not in {"iteration", "operation", "schedule", "acquire"}:
            raise ValueError("Unknown framework scope kind")
        if name is not None and kind not in {"operation", "schedule"}:
            raise ValueError("Only operation and schedule branches accept names")
        _validate_name(name)
        return self._attach(kind, name)

    def _attach(self, kind: str, name: str | None) -> Scope:
        with self._lock:
            if self._closed:
                raise RuntimeError("Cannot add children to a released scope")
            # The public constructor accepts roots; specialize its shared state here.
            child = Scope("run", stop=self.stop.child(boundary=kind == "operation"))
            child._kind = kind
            child._name = name
            child._parent = self
            child._lock = self._lock
            child._sequence = self._sequences.get(kind, 0) + 1
            self._sequences[kind] = child.sequence
            self._children[child] = None
            return child

    def _walk(self) -> list[Scope]:
        nodes: list[Scope] = [self]
        for child in self._children:
            nodes.extend(child._walk())
        return nodes

    def cancel(self, scope: str, *, expected: int | None = None) -> CancelResult:
        """Atomically select and stop a live subtree, never its parent.

        Select by kind, kind#N, operation:name, or exact full leaf path.
        run aliases operation outside a run root. Multiple matches refuse with
        reason ambiguous and candidate full paths. Missing/stale selectors
        refuse without stopping anything. Invalid syntax raises ValueError.
        """
        label, sequence = parse_cancel_scope(scope, expected)
        if label == "run" and self.kind != "run":
            label = "operation"
        with self._lock:
            path = self.active_path
            if self._closed:
                return CancelResult(False, "scope not active", path)
            nodes = self._walk()
            if "/" in label:
                matches = [
                    node for node in nodes if node.path == label and not node._children
                ]
            else:
                matches = [
                    node
                    for node in nodes
                    if node.kind == label
                    or (node.kind == "operation" and f"operation:{node.name}" == label)
                ]
            if not matches:
                return CancelResult(False, "scope not active", path)
            if sequence is not None:
                matches = [node for node in matches if node.sequence == sequence]
                if not matches:
                    return CancelResult(False, "sequence mismatch", path)
            if len(matches) > 1:
                return CancelResult(
                    False, "ambiguous", path, [node.path for node in matches]
                )
            matches[0].stop.set()
            return CancelResult(True, "scope cancellation requested", path)

    def release(self) -> None:
        """Deliver facts once and detach after children drain; repeat is a no-op.

        Closing a boundary with live children raises RuntimeError. A released
        boundary rejects new children and cannot be reentered.
        """
        with self._lock:
            if self._closed:
                return
            if self._children:
                raise RuntimeError("Release child scopes before their parent")
            self.stop.release()
            if self._parent is not None:
                self._parent._children.pop(self)
            self._closed = True

    def __enter__(self) -> Self:
        if self._closed:
            raise RuntimeError("Cannot enter a released scope")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()
