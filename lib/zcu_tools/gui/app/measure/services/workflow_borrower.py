"""Deterministic cleanup of pane-owned subscriptions, editors and timers."""

from collections.abc import Callable


class WorkflowBorrower:
    """Group cleanup that must finish before a pane deletes its Qt children.

    Add bound unsubscribe, timer.stop and editor.detach callbacks as acquired.
    dispose runs them in reverse order without touching session ownership.
    A failing callback stays queued for retry and the exception rejects deletion.
    First obtain successful WorkflowControlFacet.close(tab_id); a refused close
    must leave this group and its pane intact. Use only on the GUI owner thread.
    """

    def __init__(self) -> None:
        """Create an active, empty cleanup group."""
        self._cleanups: list[Callable[[], None]] = []
        self._disposed = False

    @property
    def disposed(self) -> bool:
        """Whether all callbacks completed and children can now be deleted."""
        return self._disposed

    def add_cleanup(self, cleanup: Callable[[], None]) -> None:
        """Retain a zero-argument cleanup; raise RuntimeError after disposal."""
        if self._disposed:
            raise RuntimeError("Workflow borrower is disposed")
        self._cleanups.append(cleanup)

    def dispose(self) -> None:
        """Detach once, retaining the failing and earlier callbacks for retry."""
        while self._cleanups:
            self._cleanups[-1]()
            self._cleanups.pop()
        self._disposed = True
