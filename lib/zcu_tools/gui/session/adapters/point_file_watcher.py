"""Qt directory notifications drive owner-controlled, hash-based point reloads."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from qtpy.QtCore import QFileSystemWatcher, QObject, QTimer

from zcu_tools.gui.session.events import ContextSwitchedPayload

if TYPE_CHECKING:
    from zcu_tools.gui.event_bus import BaseEventBus
    from zcu_tools.gui.session.services.context import ContextService


class PointFileWatcher(QObject):
    """Watch current point directories with trailing-edge debounce.

    The context owns paths, hashes, parsing, installation and facts. The host
    supplies admit_reload, evaluated on the owner thread when work executes,
    not when a filesystem event arrives. Watch directories rather than files so
    atomic replacement and creation of the optional aliases file remain visible.
    close stops the timer, removes watches and unsubscribes the binding listener.
    """

    def __init__(
        self,
        context: ContextService,
        bus: BaseEventBus,
        *,
        admit_reload: Callable[[], bool],
    ) -> None:
        """Bind an owner-thread context/bus and execution-time admission callback.

        Requires a running Qt owner event loop for automatic reload. flush also
        supports explicit synchronous drains; no background threads are started.
        """
        super().__init__()
        self._context = context
        self._admit_reload = admit_reload
        self._closed = False
        self._watcher = QFileSystemWatcher(self)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self.flush)
        self._watcher.directoryChanged.connect(self._schedule)
        self._subscription = bus.subscribe(ContextSwitchedPayload, self._switch)
        self._bind_directories()

    def _schedule(self, _path: str) -> None:
        if not self._closed:
            self._timer.start()

    def _switch(self, _payload: ContextSwitchedPayload) -> None:
        self._timer.stop()
        self._bind_directories()

    def _bind_directories(self) -> None:
        previous = self._watcher.directories()
        if previous:
            self._watcher.removePaths(previous)
        directories = sorted(
            {str(file.path.parent) for file in self._context.get_file_revisions()}
        )
        if directories:
            self._watcher.addPaths(directories)

    def flush(self) -> None:
        """Drain debounce now, observing all selected-point files once.

        Reevaluate host admission even without a queued filesystem notification,
        allowing a host authority transition to retry pending changes. Equal
        hashes are silent. After close this is a harmless no-op.
        """
        if self._closed:
            return
        self._timer.stop()
        self._context.synchronize_files(admit_reload=self._admit_reload())

    def close(self) -> None:
        """Permanently stop notifications and release subscriptions; idempotent."""
        if self._closed:
            return
        self._closed = True
        self._timer.stop()
        self._subscription.unsubscribe()
        directories = self._watcher.directories()
        if directories:
            self._watcher.removePaths(directories)
