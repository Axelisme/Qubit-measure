"""Guarded native Save commands for GUI and remote driving adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from zcu_tools.experiment.run_saving import ArtifactRef
from zcu_tools.gui.session.context_types import CommandOrigin

if TYPE_CHECKING:
    from zcu_tools.gui.app.measure.state import State

    from .guard import GuardService
    from .ports import SaveArtifactsSubmission, SaveDataSubmission
    from .save import SaveService


class SaveControlPort(Protocol):
    """App-facing native Save surface shared by GUI and remote adapters."""

    def has_tab(self, tab_id: str) -> bool: ...

    def set_comment(self, tab_id: str, comment: str) -> None: ...

    def save_data(
        self,
        tab_id: str,
        comment: str | None = None,
        *,
        origin: CommandOrigin,
        run_operation_id: int | None = None,
    ) -> SaveDataSubmission:
        """Submit both data outputs with this command's attribution.

        tab_id identifies an idle, completed, active-context tab. comment=None
        retains the annotation. run_operation_id, if supplied, must identify
        the current Run pane. Guards or absent historical evidence raise
        FailedPreconditionError before comment mutation or operation admission.
        A returned reservation is not success; terminal carries SaveReport.
        """
        ...

    def save_artifacts(
        self,
        tab_id: str,
        *,
        artifacts: tuple[ArtifactRef, ...] | None = None,
        comment: str | None = None,
        origin: CommandOrigin,
    ) -> SaveArtifactsSubmission:
        """Save whole selected artifacts, including JSON and all outputs.

        None selects every declaration, even Saved. Explicit refs are ordered,
        unique and nonempty. Invalid refs or absent historical metadata/storage
        raise FailedPreconditionError before mutations or admission. Invalid
        pane content raises TypeError/ValueError. comment=None retains the
        annotation; origin attributes this Save. Reservations are not success.
        """
        ...


class SaveControlFacet:
    """Compose static guards, dynamic availability and native Save admission."""

    def __init__(self, *, state: State, guard: GuardService, save: SaveService) -> None:
        """Inject the State owner, static preconditions and owning save service."""
        self._state = state
        self._guard = guard
        self._save = save

    def has_tab(self, tab_id: str) -> bool:
        """Return whether tab_id is a current State resource."""
        return self._state.has_tab(tab_id)

    def set_comment(self, tab_id: str, comment: str) -> None:
        """Publish an idle tab's annotation and mark saved Run evidence dirty.

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
    ) -> SaveDataSubmission:
        """Submit native data saving under SaveControlPort.save_data's contract."""
        if run_operation_id is not None:
            self._state.require_run_operation(tab_id, run_operation_id)
        permit = self._guard.acquire_save_permit(tab_id)
        self._save.require_save_available(tab_id)
        return self._save.start_save_data(permit, comment=comment, origin=origin)

    def save_artifacts(
        self,
        tab_id: str,
        *,
        artifacts: tuple[ArtifactRef, ...] | None = None,
        comment: str | None = None,
        origin: CommandOrigin,
    ) -> SaveArtifactsSubmission:
        """Submit whole artifacts under SaveControlPort.save_artifacts's contract."""
        permit = self._guard.acquire_save_permit(tab_id)
        self._save.require_save_available(tab_id)
        return self._save.start_save_artifacts(
            permit, artifacts, comment=comment, origin=origin
        )
