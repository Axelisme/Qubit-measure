"""Saved notebook run handles over the shared artifact persistence contract."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Generic, TypeVar
from uuid import uuid4

from matplotlib.figure import Figure
from pydantic import TypeAdapter

from zcu_tools.datafile import AnalysisDocument, RunMetadata, RunSnapshot
from zcu_tools.experiment import (
    AcquisitionEvidence,
    AxesSpec,
    ExpCfgModel,
    GroupedAxesSpec,
    RunRecord,
)
from zcu_tools.experiment.native_persistence import prepare_run_data
from zcu_tools.experiment.run_saving import (
    ArtifactRef,
    DataMember,
    FileMember,
    SaveArtifact,
    SaveReport,
    save_artifacts,
)
from zcu_tools.resources.entry import (
    AcquiredPayload,
    ArtifactKey,
    LedgerEvent,
    PointView,
    RoleView,
    SaveLayout,
    capture_point_storage,
)

CfgT = TypeVar("CfgT", bound=ExpCfgModel)
ResultT = TypeVar("ResultT")


class PartialSaveError(Exception, Generic[CfgT, ResultT]):
    """A latest-call output failure with its original SavedRun and report.

    run retains the captured context, run directory and content for an explicit
    subsequent save. report lists this call's successes/failures. There is no
    automatic retry. Ledger failures propagate separately as their original error.
    """

    def __init__(self, run: "SavedRun[CfgT, ResultT]", report: SaveReport) -> None:
        self.run = run
        self.report = report
        super().__init__(f"Some outputs failed for run {report.run_id}")


class SavedRun(Generic[CfgT, ResultT]):
    """Captured notebook content and evidence for explicitly repeated saves.

    Obtain this handle from NotebookExperiment.save_run. Its adapter verifies
    source identity before constructing it; there is no historical-source
    guessing. Saving does not recapture context, acquisition time or completion.
    """

    def __init__(
        self,
        ctx: PointView,
        *,
        source: RunRecord[CfgT, ResultT],
        spec: AxesSpec[ResultT, CfgT] | GroupedAxesSpec[ResultT, CfgT],
        acquisition: AcquisitionEvidence,
        roles: RoleView | None = None,
    ) -> None:
        """Capture adapter-validated source/spec/evidence and this point's context.

        Adapter-internal construction: source is its current successful record
        and acquisition is acquisition_for(source). spec is the core's declared
        persistence mapping. roles, if given, is resolved from ctx; None records
        no roles. Snapshot only cached context and prepare typed data. Do not
        create run folders or append events until save. Invalid source/spec
        raises TypeError/ValueError; bound storage errors propagate.
        """
        self._data = prepare_run_data(source, spec=spec, comment="")
        self._storage = capture_point_storage(ctx)
        self._metadata = RunMetadata(
            run_id=acquisition.run_id,
            experiment=spec.tag,
            started_at=acquisition.started_at.isoformat(),
            finished_at=acquisition.finished_at.isoformat(),
            completion=acquisition.completion,
            labber_path=None,
            snapshot=RunSnapshot(
                entry_id=self._storage.entry_id,
                entry_name=self._storage.entry_name,
                point=self._storage.point,
                description=self._storage.description,
                roles=dict(roles.components) if roles is not None else {},
            ),
            provenance=deepcopy(acquisition.provenance),
        )
        self._figures: dict[str, Figure] = {}
        self._analysis: tuple[AnalysisDocument, ...] = ()
        self._report = SaveReport(acquisition.run_id, ())
        self._acquired = False
        self._acquired_at = acquisition.finished_at.isoformat()

    @property
    def run_id(self) -> str:
        """Return the acquisition's immutable run identity."""
        return self._metadata.run_id

    @property
    def paths(self) -> tuple[Path, ...]:
        """Return successful destinations from the latest report only."""
        return tuple(
            item.output.path for item in self._report.outputs if item.status == "saved"
        )

    @property
    def report(self) -> SaveReport:
        """Return this handle's latest output report; no ledger receipt is stored."""
        return self._report

    def save(
        self,
        *,
        figures: Mapping[str, Figure] | None = None,
        analysis: tuple[AnalysisDocument, ...] | None = None,
        comment: str | None = None,
    ) -> SaveReport:
        """Rewrite data and retained/supplied files under the same run identity.

        Named figures become run PNGs plus point copies. Each analysis document
        becomes analysis/<pane>.json. Pass typed content to the shared saver;
        encoding failures affect that member only, not data or other files.
        None preserves retained content/comment;
        empty mapping/tuple clears that set. comment is raw user text.
        Append acquired once before the first save and saved once per call.
        PartialSaveError retains this same handle and the latest report.
        Ledger errors propagate directly; do not retry, roll back or recapture.
        """
        if comment is not None:
            comment = TypeAdapter(str).validate_python(comment, strict=True)
            self._data = replace(
                self._data,
                payload=replace(
                    self._data.payload,
                    metadata=replace(self._data.payload.metadata, comment=comment),
                ),
            )
        if figures is not None:
            self._figures = dict(figures)
        if analysis is not None:
            self._analysis = tuple(analysis)
        artifacts = self._artifacts()
        if not self._acquired:
            self._storage.ledger.append(
                LedgerEvent(
                    id=str(uuid4()),
                    kind="acquired",
                    at=self._acquired_at,
                    entry_id=str(self._storage.entry_id),
                    origin="notebook",
                    payload=AcquiredPayload(
                        run_id=self.run_id,
                        tab=self._metadata.experiment,
                        experiment=self._metadata.experiment,
                        cfg_summary=self._data.cfg.values,
                        point=self._metadata.snapshot.point,
                        roles=dict(self._metadata.snapshot.roles),
                    ),
                )
            )
            self._acquired = True
        layout = SaveLayout(
            result_path=self._storage.result_path,
            database_path=self._storage.database_path,
            run_id=self.run_id,
            point=self._storage.point,
            saved_at=datetime.now(UTC),
        )
        self._report = save_artifacts(
            artifacts,
            layout=layout,
            metadata=self._metadata,
            ledger=self._storage.ledger,
            origin="notebook",
            call_id=None,
        )
        for item in self._report.outputs:
            if item.output.format == "labber" and item.status == "saved":
                self._metadata = replace(
                    self._metadata, labber_path=str(item.output.path)
                )
        if any(item.status == "failed" for item in self._report.outputs):
            raise PartialSaveError(self, self._report)
        return self._report

    def _artifacts(self) -> tuple[SaveArtifact, ...]:
        artifacts = [
            SaveArtifact(
                ArtifactRef("run", "measurement_data"),
                (
                    DataMember(
                        ArtifactKey("run", "measurement_data", "data"), self._data
                    ),
                ),
            )
        ]
        analysis_members = [
            FileMember(ArtifactKey("analysis", "results", "figure", name), figure)
            for name, figure in self._figures.items()
        ]
        post_members: list[FileMember] = []
        for document in self._analysis:
            member = FileMember(
                ArtifactKey(document.pane, "results", "analysis", document.pane),
                document,
            )
            if document.pane == "analysis":
                analysis_members.append(member)
            else:
                post_members.append(member)
        if analysis_members:
            artifacts.append(
                SaveArtifact(
                    ArtifactRef("analysis", "results"), tuple(analysis_members)
                )
            )
        if post_members:
            artifacts.append(
                SaveArtifact(
                    ArtifactRef("post_analysis", "results"), tuple(post_members)
                )
            )
        return tuple(artifacts)
