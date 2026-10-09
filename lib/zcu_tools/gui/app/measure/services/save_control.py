"""Guarded synchronous native Save commands for GUI and remote adapters."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from zcu_tools.experiment.run_saving import ArtifactRef, SaveReport
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.session.context_types import CommandOrigin
from zcu_tools.gui.session.services.context import ContextService
from zcu_tools.resources.entry import AcquiredPayload, SavedPayload

if TYPE_CHECKING:
    from zcu_tools.gui.app.measure.state import State
    from zcu_tools.resources.entry import RecordsLedger

    from .guard import GuardService
    from .save import SaveService


@dataclass(frozen=True)
class SavedRunInfo:
    """One opened-entry ledger run, without inspecting its data file.

    run_id is the opaque acquisition identity. point is its label or None;
    experiment is the recorded spec tag. recorded_at is the first acquired
    event time, or first saved event time for saved-only runs, never started_at.
    data_path is the latest successful data_h5 output, or None if none exists;
    it is historical evidence and need not still exist on disk.
    """

    run_id: str
    point: str | None
    experiment: str
    recorded_at: str
    data_path: Path | None


class SaveControlPort(Protocol):
    """Existing app-facing Save surface shared by GUI and remote adapters."""

    def list_saved_runs(self, *, point: str | None = None) -> tuple[SavedRunInfo, ...]:
        """Return opened-entry runs in first-event order, optionally at one point.

        None lists all points. Acquired metadata wins over saved-only metadata;
        later failed/non-native saves do not erase a successful native path.
        No opened entry raises FailedPreconditionError(no_entry). Ledger and
        saved attachment schema/I/O errors propagate, including missing records.
        Reads do not inspect HDF5, stat outputs, create folders or append events.
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


def _saved_run_info(
    ledger: RecordsLedger, event_id: str, payload: SavedPayload, recorded_at: str
) -> SavedRunInfo:
    record = ledger.get(event_id).record
    if record is None:
        raise ValueError(f"Saved event {event_id} has no record")
    point = record.get("point")
    experiment = record.get("experiment")
    if "point" not in record or (point is not None and not isinstance(point, str)):
        raise ValueError(
            f"Saved event {event_id} record point must be a string or None"
        )
    if not isinstance(experiment, str):
        raise ValueError(f"Saved event {event_id} record experiment must be a string")
    data_path = next(
        (
            Path(output.path)
            for output in reversed(payload.outputs)
            if output.format == "data_h5" and output.status == "saved"
        ),
        None,
    )
    return SavedRunInfo(payload.run_id, point, experiment, recorded_at, data_path)


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
        """Read the ledger under SaveControlPort.list_saved_runs's contract."""
        entry = self._context.get_entry_info()
        if entry is None:
            raise FailedPreconditionError("No entry is opened", reason_code="no_entry")
        ledger = self._context.entry_ledger(entry.entry_id)
        runs: dict[str, SavedRunInfo] = {}
        acquired: set[str] = set()
        for event in ledger.events():
            payload = event.payload
            if isinstance(payload, AcquiredPayload):
                if payload.run_id in acquired:
                    continue
                previous = runs.get(payload.run_id)
                runs[payload.run_id] = SavedRunInfo(
                    run_id=payload.run_id,
                    point=payload.point,
                    experiment=payload.experiment,
                    recorded_at=event.at,
                    data_path=previous.data_path if previous is not None else None,
                )
                acquired.add(payload.run_id)
            elif isinstance(payload, SavedPayload):
                saved = _saved_run_info(ledger, event.id, payload, event.at)
                previous = runs.get(payload.run_id)
                if previous is None:
                    runs[payload.run_id] = saved
                elif saved.data_path is not None:
                    runs[payload.run_id] = replace(previous, data_path=saved.data_path)
        return tuple(
            run for run in runs.values() if point is None or run.point == point
        )

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
