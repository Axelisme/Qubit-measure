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

from zcu_tools.cfg_model import ModuleSpec, WaveformSpec
from zcu_tools.experiment import (
    AcquisitionEvidence,
    AxesSpec,
    ExpCfgModel,
    GroupedAxesSpec,
    RunRecord,
)
from zcu_tools.experiment.artifacts import (
    AnalysisDocument,
    AnalysisItem,
    ArtifactRef,
    DataItem,
    FigureItem,
    SaveArtifact,
    SaveReport,
    save_artifacts,
)
from zcu_tools.experiment.metadata import RunMetadata, RunSnapshot
from zcu_tools.format_version import YamlMap
from zcu_tools.resources.entry import (
    AcquiredPayload,
    ArtifactKey,
    JsonObject,
    JsonValue,
    LedgerEvent,
    ModuleWriteback,
    ParameterWriteback,
    PointView,
    PointWritebackTarget,
    RoleView,
    SampleWriteback,
    SaveLayout,
    WaveformWriteback,
    WritebackEvidence,
    WritebackItem,
    apply_writeback,
    capture_point_storage,
    resolve_path,
    snapshot,
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
        persistence mapping. roles must be resolved from the same bound entry,
        point and storage roots as ctx; another handle of that point is valid.
        None records no roles. Wrong role origins raise ValueError before any
        output or event. Snapshot only cached context and prepare typed data.
        Do not create run folders or append events until save. Invalid
        source/spec becomes failed data outputs; bound storage errors propagate.
        """
        self._data = DataItem.from_run(
            ArtifactKey("run", "measurement_data", "data"),
            source,
            spec=spec,
            comment="",
        )
        self._storage = capture_point_storage(ctx, roles=roles)
        self._point = ctx
        metadata = RunMetadata(
            run_id=acquisition.run_id,
            experiment=spec.tag,
            started_at=acquisition.started_at.isoformat(),
            finished_at=acquisition.finished_at.isoformat(),
            completion=acquisition.completion,
            snapshot=RunSnapshot(
                entry_id=self._storage.entry_id,
                point=self._storage.point,
                roles=dict(self._storage.roles),
            ),
            provenance=deepcopy(acquisition.provenance),
            device_snapshot=deepcopy(acquisition.device_snapshot),
        )
        self._record = replace(source, metadata=metadata)
        self._figures: dict[str, Figure] = {}
        self._analysis: tuple[AnalysisDocument, ...] = ()
        self._report = SaveReport(acquisition.run_id, ())
        self._acquired = False
        self._acquired_at = acquisition.finished_at.isoformat()

    @property
    def _metadata(self) -> RunMetadata:
        metadata = self._record.metadata
        if metadata is None:
            raise ValueError("SavedRun requires record metadata")
        return metadata

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
        Labber files use the host's current local date; run identity and
        acquisition timestamps stay unchanged.
        Append acquired once before the first save and saved once per call.
        PartialSaveError retains this same handle and the latest report.
        Ledger errors propagate directly; do not retry, roll back or recapture.
        """
        if comment is not None:
            comment = TypeAdapter(str).validate_python(comment, strict=True)
            self._data = self._data.with_comment(comment)
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
                        experiment=self._metadata.experiment,
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
            experiment=self._metadata.experiment,
            point=self._storage.point,
            saved_at=datetime.now(UTC).astimezone(),
        )
        self._report = save_artifacts(
            artifacts,
            layout=layout,
            record=self._record,
            ledger=self._storage.ledger,
            origin="notebook",
        )
        if any(item.status == "failed" for item in self._report.outputs):
            raise PartialSaveError(self, self._report)
        return self._report

    def accept(
        self,
        *,
        parameter: Mapping[
            str, JsonValue | complex | tuple[JsonValue | complex, YamlMap]
        ]
        | None = None,
        module: Mapping[str, ModuleSpec] | None = None,
        waveform: Mapping[str, WaveformSpec] | None = None,
        samples: Mapping[str, JsonObject] | None = None,
    ) -> str:
        """Accept results into this run's captured point, library and samples.

        parameter maps dotted point paths to working-unit JSON/complex values
        or (value, metadata) pairs. A plain value keeps the path's existing
        metadata. A pair replaces it with the caller's opaque string-keyed YAML
        mapping in the same point commit; {} clears it. The second item must be
        such a mapping; anything else, including a number or None, raises
        ValidationError before the ledger append. Keys such as stderr are not
        interpreted. module and waveform map library keys to named persistent Specs,
        not raw dictionaries. Parameter and library paths expand only their
        first segment using this run's captured roles. At least one mapping
        must be nonempty.
        samples maps concrete component names (not role aliases) to JSON field
        dictionaries. Device fields and units are optional and preserved as
        provided; the caller owns their consistency. Only run_id, point, UTC at
        and source are set automatically, overriding these keys if supplied.
        No live aliases, device state or cfg are consulted for sample values.

        Append one accepted event, then apply parameter, module, waveform and
        samples in that order, preserving each mapping's iteration order.
        Each item commits separately. An item failure raises WritebackApplyError
        with event_id/applied and the original cause; ledger append fails unchanged.
        The first failure stops later writes; prior successful items
        and the event remain. No cross-item rollback or retry is provided.
        Library Provenance kind is this run's experiment name, never
        caller-supplied. Parameter metadata receives no event, run or time.
        Return the accepted event ID after all items succeed.
        """
        if not parameter and not module and not waveform and not samples:
            raise ValueError("accept requires at least one nonempty mapping")
        document = snapshot(self._point)
        items: list[WritebackItem] = []
        for path, item in (parameter or {}).items():
            if isinstance(item, tuple):
                value, raw_metadata = item
                metadata = TypeAdapter(YamlMap).validate_python(
                    raw_metadata, strict=True
                )
            else:
                value, metadata = item, None
            actual_path = resolve_path(document, path, roles=self._storage.roles)
            items.append(ParameterWriteback(actual_path, value, metadata))
        for path, spec in (module or {}).items():
            actual_path = resolve_path(document, path, roles=self._storage.roles)
            items.append(ModuleWriteback(actual_path, spec))
        for path, spec in (waveform or {}).items():
            actual_path = resolve_path(document, path, roles=self._storage.roles)
            items.append(WaveformWriteback(actual_path, spec))
        for component, fields in (samples or {}).items():
            items.append(SampleWriteback(component, fields))
        return apply_writeback(
            tuple(items),
            target=PointWritebackTarget(self._point, self._storage),
            evidence=WritebackEvidence(
                kind=self._metadata.experiment,
                run_id=self.run_id,
                origin="notebook",
            ),
        ).event_id

    def _artifacts(self) -> tuple[SaveArtifact, ...]:
        artifacts = [
            SaveArtifact(
                ArtifactRef("run", "measurement_data"),
                (self._data,),
            )
        ]
        analysis_members: list[FigureItem | AnalysisItem] = [
            FigureItem(ArtifactKey("analysis", "results", "figure", name), figure)
            for name, figure in self._figures.items()
        ]
        post_members: list[FigureItem | AnalysisItem] = []
        for document in self._analysis:
            member = AnalysisItem(
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
