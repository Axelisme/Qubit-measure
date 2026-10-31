"""Native artifact saving and complete State publication on the owner thread."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from zcu_tools.experiment.artifacts import (
    AnalysisItem,
    ArtifactRef,
    DataItem,
    FigureItem,
    SaveArtifact,
    SaveReport,
    save_artifacts,
)
from zcu_tools.experiment.metadata import RunMetadataContent, has_native_metadata
from zcu_tools.experiment.records import RunRecord
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
from zcu_tools.resources.entry import (
    ArtifactKey,
    RunOutputBinding,
    SaveContext,
    SaveLayout,
    bind_run_outputs,
)

if TYPE_CHECKING:
    from zcu_tools.gui.app.measure.state import Session, State
    from zcu_tools.gui.event_bus import BaseEventBus as EventBus


@dataclass(frozen=True)
class OutputPlan:
    """One naming decision shared by preview and Save.

    binding fixes the recorded entry/point beneath current roots and prepares
    the ledger only when Save requests it. layout calculates artifact paths
    from that binding, the adapter's stem and the supplied save timestamp.
    Neither field implies that any output has been written.
    """

    binding: RunOutputBinding
    layout: SaveLayout


type LayoutPlanner = Callable[
    [Session[Any, Any, Any, Any], RunRecord[Any, Any], datetime], OutputPlan
]


@dataclass(frozen=True)
class _SavePlan:
    artifacts: tuple[SaveArtifact, ...]
    layout: SaveLayout
    record: RunRecord[Any, Any]
    outputs: RunOutputBinding
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
        result_root: Path,
        database_root: Path,
        access: ExperimentAccess | None = None,
    ) -> None:
        """Inject State, publication bus, current output roots and the catalog gate.

        result_root and database_root are parent roots, not entry directories.
        Save binds recorded names beneath them without consulting live points.
        """
        self._state = state
        self._bus = bus
        self._result_root = result_root
        self._database_root = database_root
        self._access = access if access is not None else ExperimentAccess()

    def plan_layout(
        self,
        tab: Session[Any, Any, Any, Any],
        record: RunRecord[Any, Any],
        saved_at: datetime,
    ) -> OutputPlan:
        """Return a binding and layout without writing files or changing status.

        tab supplies the registered adapter and its name; record supplies cfg
        and historical metadata, including entry, point, roles and run_id.
        saved_at must be timezone-aware. Use the injected current roots, never
        acquisition storage or live selection. Missing metadata, unsafe names
        or an invalid adapter stem raise ValueError. Unreadable historical
        fields raise MetadataFieldError. Read only metadata fields
        needed for naming; full Save validation remains in the Save command.
        """
        if not has_native_metadata(record.metadata):
            raise ValueError("Run has no historical native metadata")
        metadata = RunMetadataContent(record.metadata)
        history = metadata.snapshot
        stem = tab.adapter.save_stem(
            SaveContext(
                name=tab.adapter_name,
                run_id=metadata.run_id,
                point=history.point,
                roles=history.roles,
                cfg=None if record.cfg is None else record.cfg.model_dump(mode="json"),
                saved_at=saved_at,
            )
        )
        binding = bind_run_outputs(
            result_root=self._result_root,
            database_root=self._database_root,
            entry_name=history.entry_name,
            point=history.point,
        )
        return OutputPlan(binding, binding.layout(stem=stem, saved_at=saved_at))

    def require_save_available(self, tab_id: str) -> None:
        """Reject reload/shutdown or busy tabs without changing drafts or files.

        tab_id must identify an existing tab. Raise FailedPreconditionError for
        unavailable commands; unknown-tab errors propagate.
        """
        self._access.require_available()
        self._require_tab_idle(tab_id)

    def save_data(
        self, tab_id: str, comment: str | None = None, *, origin: CommandOrigin
    ) -> SaveReport:
        """Synchronously save native HDF5 and a fresh Labber file on the owner thread.

        tab_id identifies an existing idle tab with native metadata.
        comment=None retains its annotation; origin attributes this command.
        Missing evidence or busy raises FailedPreconditionError before mapping
        or I/O. Recorded names select outputs under the injected current roots.
        Retain the historical run_id and point for all
        output paths and the saved event, regardless of later global selections.
        Output failures appear in the complete report; ledger errors
        propagate without a report. State publication completes before return.
        """
        plan = self._capture(
            tab_id, (ArtifactRef("run", "measurement_data"),), comment, origin
        )
        return self._save(tab_id, plan, data_only=True)

    def save_artifacts(
        self,
        tab_id: str,
        artifacts: tuple[ArtifactRef, ...] | None = None,
        comment: str | None = None,
        *,
        origin: CommandOrigin,
    ) -> SaveReport:
        """Synchronously save whole artifacts and publish their complete report.

        tab_id identifies an existing tab. None selects all declarations, including
        Saved; explicit refs are ordered, unique and nonempty. Unknown refs,
        missing metadata or busy raise FailedPreconditionError before
        mapping, I/O or annotation mutation. No live entry, point or storage
        capability is required. All selected outputs and the saved event use the
        historical run_id and point, not the current global point.
        Invalid pane content raises TypeError/ValueError.
        comment=None retains the annotation. origin attributes this command.
        All mapping, PNG encoding, writes and ledger append finish before return.
        Output failures retain diagnostics and do not stop independent outputs;
        ledger errors propagate without a report. No Save handle exists.
        """
        plan = self._capture(tab_id, artifacts, comment, origin)
        return self._save(tab_id, plan)

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
        record = self._capture_source(tab_id)
        tab = self._state.get_tab(tab_id)
        try:
            selected = tab.artifacts.select(refs)
        except ValueError as exc:
            raise FailedPreconditionError(str(exc)) from exc
        output_plan = self.plan_layout(tab, record, datetime.now().astimezone())
        annotation = tab.save.comment if comment is None else comment
        captured: list[SaveArtifact] = []
        for artifact in selected:
            ref = artifact.ref
            if ref.section == "run":
                item = self._capture_data(
                    tab_id, ArtifactKey(ref.section, ref.name, "data"), annotation
                )
                captured.append(SaveArtifact(ref, (item,)))
                continue
            if ref.section == "analysis":
                pane = tab.analysis
                pane_name = "analysis"
            elif ref.section == "post_analysis":
                pane = tab.post_analysis
                pane_name = "post_analysis"
            else:
                raise ValueError(f"Unsupported pane section {ref.section!r}")
            members: list[FigureItem | AnalysisItem] = []
            for key in artifact.members:
                if key.member == "figure":
                    if pane.plots is None or key.member_name is None:
                        raise ValueError("Declared figure has no pane content")
                    members.append(FigureItem(key, pane.plots[key.member_name]))
                else:
                    members.append(
                        tab.adapter.analysis_item(
                            key, pane.result, pane.result_params, pane=pane_name
                        )
                    )
            captured.append(SaveArtifact(ref, tuple(members)))
        return _SavePlan(
            tuple(captured),
            output_plan.layout,
            record,
            output_plan.binding,
            origin,
            annotation,
        )

    def _capture_source(self, tab_id: str) -> RunRecord[Any, Any]:
        """Capture native save evidence after admission checks."""
        if not self._state.has_tab(tab_id):
            raise FailedPreconditionError(
                f"Unknown tab: {tab_id!r}", reason_code="unknown_tab"
            )
        tab = self._state.get_tab(tab_id)
        record = tab.run.result
        if record is None:
            raise FailedPreconditionError(
                "No run result available to save.", reason_code="no_run_result"
            )
        self._access.require_available()
        self._require_tab_idle(tab_id)
        if not has_native_metadata(record.metadata):
            raise FailedPreconditionError(
                "Run has no historical native metadata",
                reason_code="storage_unavailable",
            )
        return record

    def _capture_data(self, tab_id: str, key: ArtifactKey, comment: str) -> DataItem:
        tab = self._state.get_tab(tab_id)
        try:
            return tab.adapter.data_item(key, tab.run.result, comment=comment)
        except Exception as error:
            # Preparation belongs to this data item, not the whole Save All.
            logging.exception("Failed preparing data item %s", key)
            return DataItem.failed(key, error)

    def _save(
        self, tab_id: str, plan: _SavePlan, *, data_only: bool = False
    ) -> SaveReport:
        self._state.update_tab_comment(tab_id, plan.comment)
        report = save_artifacts(
            plan.artifacts,
            layout=plan.layout,
            record=plan.record,
            ledger=plan.outputs.prepare_ledger(),
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
