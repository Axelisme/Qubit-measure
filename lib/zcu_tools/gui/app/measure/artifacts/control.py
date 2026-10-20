"""Guarded synchronous native Save commands for GUI and remote adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

from zcu_tools.datafile import read_run_metadata
from zcu_tools.experiment.artifacts import ArtifactRef, SaveReport
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.session.context_types import CommandOrigin
from zcu_tools.gui.session.services.context import ContextService

if TYPE_CHECKING:
    from zcu_tools.gui.app.measure.services.guard import GuardService
    from zcu_tools.gui.app.measure.state import State

    from .save import SaveService


@dataclass(frozen=True)
class SavedRunInfo:
    """One existing native file in the opened entry, not proof it can be loaded.

    data_path is its exact path. run_id, point, experiment, started_at,
    finished_at and completion come from RunMetadata/RunSnapshot. Times are
    historical UTC ISO strings, or None when unknown. completion is historical
    progress, including unknown, not a save status. metadata_error is None on a
    successful metadata read. Otherwise it contains the decoding/I/O diagnostic,
    run_id and experiment are empty, and unavailable point/times/completion are
    None. No values are inferred from directory names, file timestamps or ledger
    events. A successful metadata read does not validate data arrays.
    """

    run_id: str
    point: str | None
    experiment: str
    started_at: str | None
    data_path: Path
    finished_at: str | None = None
    completion: Literal["complete", "partial", "stopped", "unknown"] | None = None
    metadata_error: str | None = None


class SaveControlPort(Protocol):
    """Existing app-facing Save surface shared by GUI and remote adapters."""

    def list_saved_runs(self, *, point: str | None = None) -> tuple[SavedRunInfo, ...]:
        """List opened-entry runs/*/data.h5 files in directory-name order.

        None lists all points; a point label filters by the native snapshot.
        Metadata I/O/format failures remain rows with metadata_error and a
        selectable path. Read only metadata, not arrays or ledger events. No
        opened entry raises FailedPreconditionError(no_entry). Missing runs
        directories yield no rows. This query never creates folders or writes events.
        """
        ...

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

        tab_id identifies an idle tab with a completed historical result. comment=None
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


def _saved_run_info(path: Path) -> SavedRunInfo:
    try:
        metadata = read_run_metadata(path)
    except (OSError, ValueError) as error:
        # Isolate a single unreadable file so selection still reaches Load errors.
        return SavedRunInfo("", None, "", None, path, metadata_error=str(error))
    return SavedRunInfo(
        run_id=metadata.run_id,
        point=metadata.snapshot.point,
        experiment=metadata.experiment,
        started_at=metadata.started_at,
        finished_at=metadata.finished_at,
        completion=metadata.completion,
        data_path=path,
    )


class SaveControlFacet:
    """Compose existing guards and synchronous native Save commands."""

    def __init__(
        self,
        *,
        state: State,
        guard: GuardService,
        save: SaveService,
        context: ContextService,
    ) -> None:
        """Inject State, Save owners and the current opened-entry context owner."""
        self._state = state
        self._guard = guard
        self._save = save
        self._context = context

    def list_saved_runs(self, *, point: str | None = None) -> tuple[SavedRunInfo, ...]:
        """Read native file metadata under SaveControlPort.list_saved_runs's contract."""
        entry = self._context.get_entry_info()
        if entry is None:
            raise FailedPreconditionError("No entry is opened", reason_code="no_entry")
        paths = sorted((entry.database_path / "runs").glob("*/data.h5"))
        runs = (_saved_run_info(path) for path in paths if path.is_file())
        return tuple(run for run in runs if point is None or run.point == point)

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
        self._guard.check_save(tab_id)
        self._save.require_save_available(tab_id)
        return self._save.save_data(tab_id, comment=comment, origin=origin)

    def save_artifacts(
        self,
        tab_id: str,
        *,
        artifacts: tuple[ArtifactRef, ...] | None = None,
        comment: str | None = None,
        origin: CommandOrigin,
    ) -> SaveReport:
        """Execute SaveControlPort.save_artifacts's whole-artifact contract."""
        self._guard.check_save(tab_id)
        self._save.require_save_available(tab_id)
        return self._save.save_artifacts(
            tab_id, artifacts, comment=comment, origin=origin
        )
