"""Program stop condition combined with the current acquire scope token."""

from __future__ import annotations

from zcu_tools.experiment.stop_signal import StopSignal


class AcquireCancelFlag:
    """Combine acquire-scope cancellation with a normal data stop condition.

    external is the acquire child token owned by ProgramBuilder. Create a fresh
    instance per attempt. set() ends acquisition when the authored data condition
    is satisfied; it never cancels a scope or records partial completion.
    """

    def __init__(self, external: StopSignal) -> None:
        self._external = external
        self._condition_met = False

    def is_set(self) -> bool:
        """Return whether the scope is cancelled or the data condition is met."""
        return self._external.is_set() or self._condition_met

    def set(self) -> None:
        """Mark normal data completion, leaving all scope events unchanged."""
        self._condition_met = True
