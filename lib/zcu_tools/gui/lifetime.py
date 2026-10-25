"""Owner-thread lifetime for GUI callbacks and borrowed resources."""

from collections.abc import Callable
from inspect import Parameter, signature
from types import MethodType
from weakref import WeakMethod, ref

from PyQt6 import sip
from PyQt6.QtCore import QEvent, QObject, pyqtBoundSignal


class _EventFilter(QObject):
    def __init__(
        self,
        watched: QObject,
        callback: Callable[[QObject | None, QEvent | None], bool | None],
    ) -> None:
        super().__init__(watched)
        self._callback = callback

    def eventFilter(self, a0: QObject | None, a1: QEvent | None) -> bool:
        # Qt can call a virtual while cyclic GC has already cleared __dict__.
        # Keep that binding-specific fallback here, not in every component.
        callback = getattr(self, "_callback", None)
        return bool(callback(a0, a1)) if callback is not None else False


class LifetimeScope:
    """Retire GUI callbacks before releasing borrowed resources, on the owner thread.

    Register unsubscribers, timer.stop, event-filter removal, Qt disconnects and
    matplotlib disconnects with add_cleanup as resources are acquired. Owners
    must obtain any domain close permission before calling close, and close
    before deleting Qt children. This scope does not own a session or drain work.

    Cleanup follows WorkflowBorrower semantics: propagate the first failure,
    retain that cleanup and all earlier ones for retry, and never repeat completed
    cleanup. Even a failed close permanently invalidates callbacks. Do not delete
    children until close returns successfully. No destroyed-signal closures are
    needed; the owner explicitly closes its scope.
    """

    def __init__(self, owner: QObject | None = None) -> None:
        """Create an active scope, optionally guarding an externally owned Qt object.

        A deleted C++ owner makes callbacks no-ops, but does not replace explicit
        close before deletion. Qt parents can destroy children without closeEvent.
        """
        self._owner = ref(owner) if owner is not None else None
        self._cleanups: list[Callable[[], None]] = []
        self._closed = False

    @property
    def closed(self) -> bool:
        """Whether callbacks are inactive, even if cleanup still needs retry."""
        if self._owner is not None:
            owner = self._owner()
            if owner is None or sip.isdeleted(owner):
                return True
        return self._closed

    def callback[**P, R](self, fn: Callable[P, R]) -> Callable[P, R | None]:
        """Wrap fn with the same arguments; return None without calling it after close.

        An already-running callback is not interrupted. Callers returning from a
        reentrant owner command must check closed before further presentation.
        The wrapper does not retain the scope or a QObject bound-method receiver;
        if either is collected, delivery is a no-op, as with native Qt slots.
        """

        scope_ref = ref(self)
        if isinstance(fn, MethodType) and isinstance(fn.__self__, QObject):
            method_ref = WeakMethod(fn)

            def qt_guarded(*args: P.args, **kwargs: P.kwargs) -> R | None:
                scope = scope_ref()
                method = method_ref()
                if scope is not None and not scope.closed and method is not None:
                    return method(*args, **kwargs)
                return None

            return qt_guarded

        def guarded(*args: P.args, **kwargs: P.kwargs) -> R | None:
            scope = scope_ref()
            if scope is not None and not scope.closed:
                return fn(*args, **kwargs)
            return None

        return guarded

    def connect(self, signal: pyqtBoundSignal, fn: Callable[..., object]) -> None:
        """Connect a Python slot and register disconnection before child deletion.

        Like Qt, omit trailing signal arguments a slot does not accept. For native
        Qt methods without a Python signature, pass a Python lambda instead.
        Disconnection uses the connection handle, so an externally deleted sender
        is harmless. Queued delivery still passes through callback invalidation.
        """
        if self.closed:
            raise RuntimeError("Lifetime scope is closed")
        parameters = tuple(signature(fn).parameters.values())
        variadic = any(p.kind == Parameter.VAR_POSITIONAL for p in parameters)
        count = sum(
            p.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
            for p in parameters
        )

        callback = self.callback(fn)

        def slot(*args: object) -> object:
            return callback(*(args if variadic else args[:count]))

        connection = signal.connect(slot)

        def disconnect() -> None:
            QObject.disconnect(connection)

        self.add_cleanup(disconnect)

    def install_event_filter(
        self,
        watched: QObject,
        callback: Callable[[QObject | None, QEvent | None], bool],
    ) -> None:
        """Forward watched events to callback while active, removing the filter on close.

        The proxy returns False after retirement or during Qt cyclic destruction.
        It is parented to watched. Native parent deletion is also safe when the
        scope owner is a different object; explicit close still owns cleanup.
        """
        if self.closed:
            raise RuntimeError("Lifetime scope is closed")
        proxy = _EventFilter(watched, self.callback(callback))
        watched.installEventFilter(proxy)

        def remove() -> None:
            # The watched C++ object may have an independent, shorter lifetime.
            if not sip.isdeleted(watched) and not sip.isdeleted(proxy):
                watched.removeEventFilter(proxy)
            if not sip.isdeleted(proxy):
                proxy.deleteLater()

        self.add_cleanup(remove)

    def add_cleanup(self, cleanup: Callable[[], None]) -> None:
        """Retain cleanup in acquisition order; reject registration after close."""
        if self.closed:
            raise RuntimeError("Lifetime scope is closed")
        self._cleanups.append(cleanup)

    def close(self) -> None:
        """Release in reverse order; retain a failing cleanup for retry."""
        self._closed = True
        while self._cleanups:
            self._cleanups[-1]()
            self._cleanups.pop()
