"""Typed Engine summaries for offline workflow-specific conversion tools."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path, PureWindowsPath
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

from zcu_tools.experiment.workflows import (
    EncodedRecord,
    Lifecycle,
    RunIdentity,
    RunPaths,
)
from zcu_tools.experiment.workflows.journal import ErrorInfo


class SavedManifest(BaseModel):
    """Fixed startup identity/plan/roots and known UTC lifecycle times.

    identity is the run ID/software provenance; workflow the declaration name;
    plan and initial_tunables are startup JSON captures. roots identifies the
    metadata and data directories. started_at/updated_at/ended_at are UTC times,
    with ended_at=None before termination. lifecycle is an Engine lifecycle value;
    reason is its optional terminal diagnosis.
    journal is retained only as a schema field; tools never open it. format and
    version must match the Engine document. Roots must be absolute and distinct.
    """

    model_config = ConfigDict(extra="forbid")
    format: Literal["zcu-workflow-run"]
    format_version: Literal[1]
    identity: RunIdentity
    workflow: str
    plan: JsonValue
    initial_tunables: JsonValue
    roots: RunPaths
    journal: Path
    started_at: datetime
    updated_at: datetime
    ended_at: datetime | None
    lifecycle: Lifecycle
    reason: str | None


class SavedStarted(BaseModel):
    """Invocation identity in an iteration_started envelope.

    kind is the fixed event tag; call_seq is the positive invocation number,
    commit_seq the number of prior commits, and revision the tunable capture
    revision. Counts and revision are nonnegative.
    """

    model_config = ConfigDict(extra="forbid")
    kind: Literal["iteration_started"]
    call_seq: int = Field(gt=0)
    commit_seq: int = Field(ge=0)
    revision: int = Field(ge=0)


class SavedResult(BaseModel):
    """Common result identity and data-root-relative iteration reference.

    call_seq is a positive invocation number; revision is its nonnegative
    tunable revision; iteration_dir names iter/<at least six decimal digits>.
    """

    model_config = ConfigDict(extra="forbid")
    call_seq: int = Field(gt=0)
    revision: int = Field(ge=0)
    iteration_dir: str


class SavedCommit(SavedResult):
    """Committed point: kind=step_committed, positive commit_seq, encoded_record.

    encoded_record is the Engine record encoding; readers require mode=json and
    no encoding error before validating it through the caller's point adapter.
    Identity/reference fields follow SavedResult.
    """

    kind: Literal["step_committed"]
    commit_seq: int = Field(gt=0)
    encoded_record: EncodedRecord


class SavedFinished(SavedResult):
    """Nonpoint terminal result: kind=step_finished, outcome=done or aborted.

    reason is an optional workflow diagnosis; identity fields follow SavedResult.
    """

    kind: Literal["step_finished"]
    outcome: Literal["done", "aborted"]
    reason: str | None


class SavedDiscarded(SavedResult):
    """Discarded invocation: kind=step_discarded, reason=pause or stop.

    No point is committed; identity/reference fields follow SavedResult.
    """

    kind: Literal["step_discarded"]
    reason: Literal["pause", "stop"]


class SavedFailed(SavedResult):
    """Unexpected failure: kind=step_failed and error carries ErrorInfo.

    No point is committed; identity/reference fields follow SavedResult.
    """

    kind: Literal["step_failed"]
    error: ErrorInfo


class SavedIteration(BaseModel):
    """Engine summary with format=zcu-workflow-iteration and format_version=1.

    identity/workflow/plan repeat the manifest's fixed startup values. tunables
    is this invocation's ordinary JSON capture; started identifies its revision
    and sequence; result discriminates committed/finished/discarded/failed.
    """

    model_config = ConfigDict(extra="forbid")
    format: Literal["zcu-workflow-iteration"]
    format_version: Literal[1]
    identity: RunIdentity
    workflow: str
    plan: JsonValue
    tunables: JsonValue
    started: SavedStarted
    result: Annotated[
        SavedCommit | SavedFinished | SavedDiscarded | SavedFailed,
        Field(discriminator="kind"),
    ]


@dataclass(frozen=True)
class SavedPoint[P]:
    """A caller-validated point and its exact source summary.

    point has the caller's P type; record_path is the source record.json Path.
    directory exposes its invocation root, not a discovery policy.
    """

    point: P
    record_path: Path

    @property
    def directory(self) -> Path:
        """Return this record's invocation root, never a filename-discovery root."""
        return self.record_path.parent


def read_summary(run: Path) -> EncodedRecord | None:
    """Read the author's encoded return from the exact metadata run directory.

    A missing summary.json means no persisted return and yields None. A missing
    run directory raises FileNotFoundError; a nondirectory raises
    NotADirectoryError. Malformed data raises ValueError naming summary.json.
    Other I/O errors propagate. No journal or native data files are inspected.
    """
    directory = run.resolve(strict=True)
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    source = directory / "summary.json"
    try:
        payload = source.read_bytes()
    except FileNotFoundError:
        return None
    try:
        return TypeAdapter(EncodedRecord).validate_json(payload)
    except ValueError as error:
        raise ValueError(f"{source}: {error}") from error


def read_manifest(run: Path, workflow: str) -> SavedManifest:
    """Read run/manifest.json and reject wrong workflow/roots/UTC identity.

    run is the exact metadata directory. Malformed data raises ValueError naming
    the source; I/O errors propagate. Nothing is written or read from journal.
    """
    source = run.resolve() / "manifest.json"
    try:
        manifest = SavedManifest.model_validate_json(source.read_text(encoding="utf-8"))
        if manifest.workflow != workflow:
            raise ValueError(f"Expected workflow {workflow}, got {manifest.workflow}")
        roots = manifest.roots
        if (
            not roots.metadata_root.is_absolute()
            or roots.metadata_root.resolve() != source.parent
        ):
            raise ValueError("metadata_root must identify the supplied run directory")
        if (
            not roots.data_root.is_absolute()
            or roots.data_root == roots.metadata_root
            or roots.data_root in roots.metadata_root.parents
            or roots.metadata_root in roots.data_root.parents
        ):
            raise ValueError("data_root must be absolute, distinct and non-overlapping")
        for time in (manifest.started_at, manifest.updated_at, manifest.ended_at):
            if time is not None and time.utcoffset() != timedelta(0):
                raise ValueError("Manifest times must be timezone-aware UTC")
    except ValueError as error:
        raise ValueError(f"{source}: {error}") from error
    return manifest


def _read_iteration(manifest: SavedManifest, source: Path) -> SavedIteration:
    """Read a summary and validate its manifest and invocation references.

    Validate every result classification, not just commits. I/O and validation
    errors propagate for read_points to retain the source path and cause.
    """
    saved = SavedIteration.model_validate_json(source.read_text(encoding="utf-8"))
    result = saved.result
    if (
        saved.identity != manifest.identity
        or saved.workflow != manifest.workflow
        or saved.plan != manifest.plan
    ):
        raise ValueError("Iteration identity/workflow/plan differs from manifest")
    directory_name = source.parent.name
    if (
        saved.started.call_seq != int(directory_name)
        or result.call_seq != saved.started.call_seq
        or result.revision != saved.started.revision
        or result.iteration_dir != f"iter/{directory_name}"
    ):
        raise ValueError("Iteration invocation identity/reference mismatch")
    return saved


def read_points[P](
    manifest: SavedManifest, adapter: TypeAdapter[P]
) -> tuple[tuple[SavedPoint[P], ...], tuple[Path, ...]]:
    """Read each iteration's summary; list missing record.json without recovery.

    adapter is the workflow's named point type. Only structured step_committed
    results are consumed. Validate identities, plan and invocation references
    even for discarded/failed/finished results. Source-named ValueErrors reject
    malformed/degraded input; I/O errors propagate. If summaries are missing
    and no usable committed points remain, raise ValueError listing every missing
    path, even when noncommit summaries exist. Otherwise return points and missing
    paths together. No native file scan occurs.
    """
    points: list[SavedPoint[P]] = []
    missing: list[Path] = []
    commits: set[int] = set()
    root = manifest.roots.data_root / "iter"
    for directory in sorted(root.iterdir()):
        if (
            not directory.is_dir()
            or not directory.name.isdecimal()
            or len(directory.name) < 6
        ):
            raise ValueError(f"{directory}: invalid iteration directory")
        source = directory / "record.json"
        if not source.exists():
            missing.append(source)
            continue
        try:
            saved = _read_iteration(manifest, source)
            result = saved.result
            if not isinstance(result, SavedCommit):
                continue
            if (
                result.commit_seq != saved.started.commit_seq + 1
                or result.commit_seq in commits
            ):
                raise ValueError("Invalid or duplicate commit sequence")
            commits.add(result.commit_seq)
            if (
                result.encoded_record.mode != "json"
                or result.encoded_record.error is not None
            ):
                raise ValueError("Requires structured, non-degraded point record")
            points.append(
                SavedPoint(adapter.validate_python(result.encoded_record.value), source)
            )
        except ValueError as error:
            raise ValueError(f"{source}: {error}") from error
    if missing and not points:
        raise ValueError(
            "No iteration records available; missing record.json:\n"
            + "\n".join(str(path) for path in missing)
        )
    return tuple(points), tuple(missing)


def native_path(directory: Path, filename: str) -> Path:
    """Resolve a named point reference inside its invocation, rejecting escapes.

    No candidate scanning or filename inference occurs; invalid references raise
    ValueError naming the directory and supplied filename.
    """
    path = Path(filename)
    if (
        not filename
        or path.is_absolute()
        or PureWindowsPath(filename).drive
        or "\\" in filename
        or ".." in path.parts
    ):
        raise ValueError(f"{directory}/{filename}: invalid native reference")
    resolved = (directory / path).resolve()
    if not resolved.is_relative_to(directory.resolve()):
        raise ValueError(f"{directory}/{filename}: native reference escapes iteration")
    return resolved
