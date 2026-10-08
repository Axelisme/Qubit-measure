"""Guarded synchronous native Save commands for GUI and remote adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from zcu_tools.experiment.run_saving import ArtifactRef, SaveReport
from zcu_tools.gui.session.context_types import CommandOrigin

if TYPE_CHECKING:
    from zcu_tools.gui.app.measure.state import State

    from .guard import GuardService
    from .save import SaveService


class SaveControlPort(Protocol):
    """Existing app-facing Save surface shared by GUI and remote adapters."""

    def has_tab(self, tab_id: str) -> bool: ...

    def set_comment(self, tab_id: str, comment: str) -> None: ...

    def save_data(
        self,
        tab_id: str,
        comment: str | None = None,
        *,
        origin: CommandOrigin,
        run_operation_id: int | None = None,
    ) -> SaveReport:
        """Save both data formats synchronously on the owner thread.

        tab_id identifies an idle completed active-context tab. comment=None
        retains its annotation. run_operation_id optionally binds the current
        Run source, not a Save handle. Guards/missing evidence raise
        FailedPreconditionError before mutation. Output failures are reported;
        mapping/ledger errors propagate. Return follows State publication.
        """
        ...

    def save_artifacts(
        self,
        tab_id: str,
        *,
        artifacts: tuple[ArtifactRef, ...] | None = None,
        comment: str | None = None,
        origin: CommandOrigin,
    ) -> SaveReport:
        """Synchronously save complete selected artifacts and every destination.

        None selects all declarations, including Saved. Explicit refs must be
        ordered, unique and nonempty. Invalid refs/missing evidence raise
        FailedPreconditionError before mutation; invalid pane content raises
        TypeError/ValueError. comment=None retains the annotation. Output errors
        stay in the complete report; mapping/ledger errors propagate.
        """
        ...


class SaveControlFacet:
    """Compose existing guards and synchronous native Save commands."""

    def __init__(self, *, state: State, guard: GuardService, save: SaveService) -> None:
        """Inject State, static preconditions and the owning synchronous saver."""
        self._state = state
        self._guard = guard
        self._save = save

    def has_tab(self, tab_id: str) -> bool:
        """Return whether tab_id is a current State resource."""
        return self._state.has_tab(tab_id)

    def set_comment(self, tab_id: str, comment: str) -> None:
        """Update an idle tab's annotation and mark saved Run evidence dirty.

        Busy/unavailable tabs raise FailedPreconditionError without mutation;
        unknown tabs propagate their lookup error.
        """
        self._save.require_save_available(tab_id)
        self._state.update_tab_comment(tab_id, comment)

    def save_data(
        self,
        tab_id: str,
        comment: str | None = None,
        *,
        origin: CommandOrigin,
        run_operation_id: int | None = None,
    ) -> SaveReport:
        """Execute SaveControlPort.save_data's synchronous command contract."""
        if run_operation_id is not None:
            self._state.require_run_operation(tab_id, run_operation_id)
        permit = self._guard.acquire_save_permit(tab_id)
        self._save.require_save_available(tab_id)
        return self._save.save_data(permit, comment=comment, origin=origin)

    def save_artifacts(
        self,
        tab_id: str,
        *,
        artifacts: tuple[ArtifactRef, ...] | None = None,
        comment: str | None = None,
        origin: CommandOrigin,
    ) -> SaveReport:
        """Execute SaveControlPort.save_artifacts's whole-artifact contract."""
        permit = self._guard.acquire_save_permit(tab_id)
        self._save.require_save_available(tab_id)
        return self._save.save_artifacts(
            permit, artifacts, comment=comment, origin=origin
        )
