"""Fluxdep-owned interactive contexts, independent of Qt presentation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from zcu_tools.analysis.fluxdep.line_state import FluxPickAnalysis, FluxPickState
from zcu_tools.gui.app.fluxdep.event_bus import EventBus
from zcu_tools.gui.app.fluxdep.state import FluxDepState
from zcu_tools.gui.interactive.flux_pick import SharedFluxPickPlugin
from zcu_tools.gui.interactive.plugin import BackgroundSubmitter
from zcu_tools.gui.interactive.session import Session
from zcu_tools.gui.session.ports import OwnerScheduler


@dataclass(frozen=True, slots=True)
class LinePickContext:
    """Live context owned by FluxDepInteractiveOwner, not by its widgets.

    spectrum_name identifies the active spectrum whose captured arrays are used.
    plugin supplies shared device-axis commands and the accepted alignment result.
    session is the sole committed state; old contexts retain readable snapshots
    after cancellation but reject writes.
    """

    spectrum_name: str
    plugin: SharedFluxPickPlugin[FluxPickAnalysis]
    session: Session[FluxPickState]


class FluxDepInteractiveOwner:
    """Serialize one active picker and invalidate it on spectrum facts.

    Public access requires the supplied owner thread. Views may detach without
    ending a context. Switch, reload, removal and external spectrum changes close
    old input; dispose also releases bus subscriptions.
    """

    def __init__(
        self,
        state: FluxDepState,
        bus: EventBus,
        owner: OwnerScheduler,
        *,
        background: BackgroundSubmitter | None,
        publish_alignment: Callable[[str, float, float], None],
    ) -> None:
        """Capture state and runtime ports; publish through Controller's alignment.

        background must deliver callbacks on owner, or None to reject auto_align.
        publish_alignment receives name and accepted native device positions.
        """
        raise NotImplementedError

    def begin_line_pick(self, name: str) -> LinePickContext:
        """Reuse valid active context or open captured-input device-axis picking.

        InvalidInputError rejects unknown names. FailedPreconditionError rejects
        inactive names or a disposed owner. Seed inherits meaningful alignment;
        OneTone fixes magnitude_only to True. RuntimeError rejects off-owner use.
        """
        raise NotImplementedError

    def current_line_pick(self) -> LinePickContext | None:
        """Return the valid active context or None without creating a session."""
        raise NotImplementedError

    def finish_line_pick(self) -> FluxPickAnalysis:
        """Validate, close and publish alignment; return accepted numeric result.

        Missing context or invalid separation raises FailedPreconditionError;
        validation failure keeps input editable. Clear context before publication.
        Publication errors propagate and do not reopen terminal input.
        """
        raise NotImplementedError

    def cancel(self) -> None:
        """Close current input without publication; safe with no active context."""
        raise NotImplementedError

    def dispose(self) -> None:
        """Idempotently cancel and release subscriptions; reject future begin."""
        raise NotImplementedError
