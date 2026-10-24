"""Explicit experiment control scopes; stop bits remain owned by StopSignal."""

from __future__ import annotations

from collections.abc import Callable, Generator
from datetime import datetime
from pathlib import Path
from threading import RLock
from types import TracebackType
from typing import TYPE_CHECKING, Literal, Protocol, Self

from pydantic import BaseModel

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.device import BaseDeviceInfo

from .operations import Completed, Failed, MissingCapability, RunFunction, RunIdentity
from .records import RunRecord

if TYPE_CHECKING:
    from .script_host import CallContext

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
    root_kind, separator, root_name = scope.partition(":")
    if separator and root_kind in {"workflow", "recipe", "run"}:
        _validate_name(root_name)
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


def _validate_record(record: object) -> None:
    if not isinstance(record, BaseModel):
        raise TypeError("Scope record must be a BaseModel")


class ScopeRuntime(Protocol):
    """Host capabilities shared by a root and its explicitly selected children.

    Implementations own operation execution and record persistence. Each method
    receives its exact receiver scope, never an implicit current boundary.
    """

    def directory(self, scope: Scope) -> Path:
        """Create and return the scope's mirrored data directory."""
        ...

    def read_identity(self) -> RunIdentity:
        """Read detached startup provenance, or raise MissingCapability."""
        ...

    def read_device_cfg(self, name: str) -> BaseDeviceInfo:
        """Read one detached full device settings model."""
        ...

    async def execute_call[T](
        self, scope: Scope, fn: Callable[[CallContext], T], *, name: str | None = None
    ) -> Completed[T] | Failed:
        """Execute blocking work under scope and return attempt facts."""
        ...

    async def execute_run[Cfg: ConfigBase, Result](
        self,
        scope: Scope,
        experiment: RunFunction[Cfg, Result],
        cfg: Cfg,
        *,
        name: str | None = None,
    ) -> Completed[RunRecord[Cfg, Result]] | Failed:
        """Execute a core callable with an owned cfg and borrowed RunContext."""
        ...

    async def execute_setup(
        self, scope: Scope, name: str, cfg: BaseDeviceInfo
    ) -> Completed[None] | Failed:
        """Apply a captured complete settings model cooperatively."""
        ...

    async def execute_wait(
        self, scope: Scope, target: datetime
    ) -> Completed[None] | Failed:
        """Wait cooperatively for an aware absolute deadline."""
        ...

    def execute_commit(self, scope: Scope, record: BaseModel) -> None:
        """Persist a record snapshot once for the receiver scope."""
        ...


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
        self,
        kind: RootKind,
        name: str | None = None,
        stop: StopSignal | None = None,
        *,
        runtime: ScopeRuntime | None = None,
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
        self._runtime = runtime
        self._committed = False

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
            child._runtime = self._runtime
            child._kind = kind
            child._name = name
            child._parent = self
            child._lock = self._lock
            child._sequence = self._sequences.get(kind, 0) + 1
            self._sequences[kind] = child.sequence
            self._children[child] = None
            return child

    @property
    def parent(self) -> Scope | None:
        """The explicit parent boundary, or None for the root."""
        return self._parent

    @property
    def iteration_sequence(self) -> int:
        """Birth number of the nearest iteration ancestor; zero outside iterations."""
        if self.kind == "iteration":
            return self.sequence
        return self.parent.iteration_sequence if self.parent is not None else 0

    def _capabilities(self) -> ScopeRuntime:
        if self._runtime is None:
            raise MissingCapability("scope runtime")
        return self._runtime

    @property
    def dir(self) -> Path:
        """Create the mirrored scope directory on first use; require a bound host."""
        return self._capabilities().directory(self)

    @property
    def identity(self) -> RunIdentity:
        """Detached root startup identity, inherited by all child scopes."""
        return self._capabilities().read_identity()

    def get_device_cfg(self, name: str) -> BaseDeviceInfo:
        """Live-read detached full device settings; unknown name/I/O errors propagate."""
        return self._capabilities().read_device_cfg(name)

    async def call[T](
        self, fn: Callable[[CallContext], T], *, name: str | None = None
    ) -> Completed[T] | Failed:
        """Run fn(CallContext) off-loop under this scope, with optional operation name.

        Acquire failure returns Failed; unexpected errors propagate. Descendant
        cancellation delivers partial values, while root cancellation raises
        ScriptCancelled after work drains and completion facts are recorded.
        """
        return await self._capabilities().execute_call(self, fn, name=name)

    async def run[Cfg: ConfigBase, Result](
        self, experiment: RunFunction[Cfg, Result], cfg: Cfg, *, name: str | None = None
    ) -> Completed[RunRecord[Cfg, Result]] | Failed:
        """Run a core (cfg, *, context) callable with a copied cfg, without saving.

        The host captures device settings and supplies borrowed hardware and
        operation plots. Pause retries the same captured request. Completion
        carries RunRecord and immutable attempt facts; acquire failure is Failed.
        """
        return await self._capabilities().execute_run(self, experiment, cfg, name=name)

    async def setup_device(
        self, name: str, cfg: BaseDeviceInfo
    ) -> Completed[None] | Failed:
        """Apply complete copied settings through the device owner; preserve its errors."""
        return await self._capabilities().execute_setup(self, name, cfg)

    async def wait_until(self, target: datetime) -> Completed[None] | Failed:
        """Wait for an aware absolute deadline; reject naive dates and cooperate on stop."""
        return await self._capabilities().execute_wait(self, target)

    def commit(self, record: BaseModel) -> None:
        """Snapshot one BaseModel into this scope's record.json; second commit raises.

        Publication is synchronous and bypasses Pause, including root-cancel cleanup.
        Later mutation of record does not change the committed snapshot.
        """
        _validate_record(record)
        if self._committed:
            raise ValueError("Scope is already committed")
        self._capabilities().execute_commit(self, record)
        self._committed = True

    def scope(self, label: str) -> Scope:
        """Attach an authored boundary; use with to release it after work drains."""
        return self.child(label)

    def iterations(self) -> Generator[Scope, None, None]:
        """Yield fresh iterations, releasing each at next() or generator close.

        Use a temporary generator in zip(values, env.iterations()) so break or
        an exception drops it and closes the yielded boundary. A retained
        generator must be explicitly closed by its owner.
        """
        while True:
            with self.branch("iteration") as iteration:
                yield iteration

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
            if "/" in label or label == self.path:
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
