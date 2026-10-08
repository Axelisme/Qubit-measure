"""Native artifact saving and complete State publication on the owner thread."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from zcu_tools.datafile import RunMetadata
from zcu_tools.experiment.run_saving import (
    ArtifactRef,
    DataMember,
    FileMember,
    SaveArtifact,
    SaveReport,
    save_artifacts,
)
from zcu_tools.gui.app.measure.catalog import ExperimentAccess
from zcu_tools.gui.app.measure.events.completion import (
    SaveArtifactsFinishedPayload,
    SaveDataFinishedPayload,
)
from zcu_tools.gui.app.measure.events.tab import (
    TabInteractionChangedPayload,
    TabInteractionFact,
)
from zcu_tools.gui.app.measure.figure_export import render_save_figure_png
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.session.context_types import CommandOrigin
from zcu_tools.resources.entry import ArtifactKey, SaveLayout

from .guard import SavePermit

if TYPE_CHECKING:
    from zcu_tools.gui.app.measure.state import State
    from zcu_tools.gui.event_bus import BaseEventBus as EventBus
    from zcu_tools.resources.entry import PointStorageBinding


@dataclass(frozen=True)
class _SavePlan:
    artifacts: tuple[SaveArtifact, ...]
    layout: SaveLayout
    metadata: RunMetadata
    storage: PointStorageBinding
    origin: CommandOrigin
    comment: str


class SaveService:
    """Capture fixed pane content and synchronously save without a hardware lease.

    Public commands run on State's owner thread. Data mapping, PNG encoding,
    file/ledger I/O and report publication finish in that call. Save commands
    are non-cancellable and never retry.
    """

    def __init__(
        self,
        state: State,
        bus: EventBus,
        *,
        access: ExperimentAccess | None = None,
    ) -> None:
        """Inject the State owner, publication bus and reload/shutdown entry gate."""
        self._state = state
        self._bus = bus
        self._access = access if access is not None else ExperimentAccess()

    def require_save_available(self, tab_id: str) -> None:
        """Reject reload/shutdown or busy tabs without changing drafts or files.

        tab_id must identify an existing tab. Raise FailedPreconditionError for
        unavailable commands; unknown-tab errors propagate.
        """
        self._access.require_available()
        self._require_tab_idle(tab_id)

    def save_data(
        self, permit: SavePermit, comment: str | None = None, *, origin: CommandOrigin
    ) -> SaveReport:
        """Synchronously save native HDF5 and a fresh Labber file on the owner thread.

        permit identifies an idle guarded tab with historical storage/metadata.
        comment=None retains its annotation; origin attributes this command.
        Missing evidence, busy or an entry/point mismatch between file history
        and captured storage raises FailedPreconditionError before mapping or I/O.
        Mismatch diagnostics identify both sources; no automatic rebinding occurs.
        Output failures appear in the complete report; mapping or ledger errors
        propagate without a report. State publication completes before return.
        """
        plan = self._capture(
            permit.tab_id, (ArtifactRef("run", "measurement_data"),), comment, origin
        )
        return self._save(permit.tab_id, plan, data_only=True)

    def save_artifacts(
        self,
        permit: SavePermit,
        artifacts: tuple[ArtifactRef, ...] | None = None,
        comment: str | None = None,
        *,
        origin: CommandOrigin,
    ) -> SaveReport:
        """Synchronously save whole artifacts and publish their complete report.

        Call on State's owner thread. None selects all declarations, including
        Saved; explicit refs are ordered, unique and nonempty. Unknown refs,
        missing storage/metadata, busy or history/captured entry/point mismatch
        raise FailedPreconditionError before mapping, I/O or annotation mutation.
        Mismatch diagnostics identify both sources; no automatic rebinding occurs.
        Invalid pane content raises TypeError/ValueError.
        comment=None retains the annotation. origin attributes this command.
        All mapping, PNG encoding, writes and ledger append finish before return.
        Output failures retain diagnostics and do not stop independent outputs;
        mapping/ledger errors propagate without a report. No Save handle exists.
        """
        plan = self._capture(permit.tab_id, artifacts, comment, origin)
        return self._save(permit.tab_id, plan)

    def _require_tab_idle(self, tab_id: str) -> None:
        if self._state.is_tab_busy(tab_id):
            raise FailedPreconditionError(f"Tab {tab_id!r} is busy")

    def _capture(
        self,
        tab_id: str,
        refs: tuple[ArtifactRef, ...] | None,
        comment: str | None,
        origin: CommandOrigin,
    ) -> _SavePlan:
        self._require_tab_idle(tab_id)
        tab = self._state.get_tab(tab_id)
        metadata = tab.run.metadata
        if metadata is None or tab.save_context is None:
            raise FailedPreconditionError(
                "Run has no historical native metadata or captured storage",
                reason_code="storage_unavailable",
            )
        try:
            selected = tab.artifacts.select(refs)
        except ValueError as exc:
            raise FailedPreconditionError(str(exc)) from exc
        storage = tab.save_context.capture_storage()
        history = metadata.snapshot
        if history.entry_id != storage.entry_id or history.point != storage.point:
            raise FailedPreconditionError(
                f"Historical run entry_id={history.entry_id!r}, point={history.point!r} "
                f"does not match tab entry_id={storage.entry_id!r}, point={storage.point!r}; "
                "save from a tab captured at the matching historical entry and point."
            )
        layout = SaveLayout(
            result_path=storage.result_path,
            database_path=storage.database_path,
            run_id=metadata.run_id,
            point=storage.point,
            saved_at=datetime.now().astimezone(),
        )
        annotation = tab.save.comment if comment is None else comment
        captured: list[SaveArtifact] = []
        for artifact in selected:
            ref = artifact.ref
            if ref.section == "run":
                prepared = tab.adapter.prepare_save_data(
                    tab.run.result, comment=annotation
                )
                captured.append(
                    SaveArtifact(
                        ref,
                        (
                            DataMember(
                                ArtifactKey(ref.section, ref.name, "data"), prepared
                            ),
                        ),
                    )
                )
            else:
                if ref.section == "analysis":
                    pane = tab.analysis
                    pane_name = "analysis"
                elif ref.section == "post_analysis":
                    pane = tab.post_analysis
                    pane_name = "post_analysis"
                else:
                    raise ValueError(f"Unsupported pane section {ref.section!r}")
                members: list[FileMember] = []
                for key in artifact.members:
                    if key.member == "figure":
                        if pane.plots is None or key.member_name is None:
                            raise ValueError("Declared figure has no pane content")
                        content = pane.plots[key.member_name]
                    else:
                        content = tab.adapter.capture_analysis_document(
                            pane.result, pane.result_params, pane=pane_name
                        )
                    members.append(FileMember(key, content))
                captured.append(SaveArtifact(ref, tuple(members)))
        return _SavePlan(tuple(captured), layout, metadata, storage, origin, annotation)

    def _save(
        self, tab_id: str, plan: _SavePlan, *, data_only: bool = False
    ) -> SaveReport:
        self._state.update_tab_comment(tab_id, plan.comment)
        report = save_artifacts(
            plan.artifacts,
            layout=plan.layout,
            metadata=plan.metadata,
            ledger=plan.storage.ledger,
            origin=plan.origin.origin,
            figure_encoder=render_save_figure_png,
        )
        self._state.commit_save_report(tab_id, report)
        error = (
            "\n".join(
                f"{item.output.path}: {item.error}"
                for item in report.outputs
                if item.status == "failed"
            )
            or None
        )
        self._bus.emit(
            TabInteractionChangedPayload(
                tab_id=tab_id,
                fact=(
                    TabInteractionFact.SAVE_SUCCEEDED
                    if error is None
                    else TabInteractionFact.SAVE_FAILED
                ),
            )
        )
        if data_only:
            data_path = next(
                str(item.output.path)
                for item in report.outputs
                if item.output.format == "data_h5"
            )
            self._bus.emit(
                SaveDataFinishedPayload(
                    tab_id=tab_id, data_path=data_path, error=error, report=report
                )
            )
        else:
            self._bus.emit(
                SaveArtifactsFinishedPayload(tab_id=tab_id, error=error, report=report)
            )
        return report
