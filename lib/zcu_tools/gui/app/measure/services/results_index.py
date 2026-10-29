"""Read-only project-wide result queries shared by the browser and RPC."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from zcu_tools.datafile import JsonObject, read_doc
from zcu_tools.resources.document_names import PARAMETERS_FILENAME
from zcu_tools.resources.entry import (
    AcquiredPayload,
    LedgerAcquisition,
    LedgerEventContent,
    ResultEntry,
    validate_path_segment,
)
from zcu_tools.resources.entry.aliases import AliasValue

_JSON = TypeAdapter(JsonObject)


class _SavedSnapshot(BaseModel):
    model_config = ConfigDict(strict=True)
    point: str | None = None
    roles: dict[str, AliasValue] = Field(default_factory=dict)


class _SavedMetadata(BaseModel):
    # A browsing projection, not persistence admission: missing fields and extra
    # saved facts do not require today's RunMetadata schema or experiment code.
    model_config = ConfigDict(strict=True)
    run_id: str | None = None
    experiment: str | None = None
    started_at: str | None = None
    snapshot: _SavedSnapshot = Field(default_factory=_SavedSnapshot)


@dataclass(frozen=True)
class ResultRun:
    """One saved run directory, including incomplete or unregistered outputs.

    key is an opaque project-local selector, not a caller-supplied path. entry
    names the Database entry. run_id/experiment/point are historical identities;
    None means unavailable (point also represents global runs). roles contains
    only captured bindings, each a component name or ordered name list.
    at is the acquired UTC timestamp, else metadata.started_at, else None;
    never a filesystem timestamp.
    registered means a matching acquired event exists in the same-named Result
    entry. run_path is the absolute exact run directory. data_path is an existing
    data.h5 or None. figure_paths/analysis_paths are sorted existing PNG/JSON
    members. search_text is casefolded identity, roles, cfg and analysis content,
    including tone names absent from roles. issues holds setup/ledger failures
    for this entry and per-file read failures for this run.
    """

    key: str
    entry: str
    run_id: str | None
    experiment: str | None
    point: str | None
    roles: Mapping[str, AliasValue]
    at: str | None
    registered: bool
    run_path: Path
    data_path: Path | None
    figure_paths: tuple[Path, ...]
    analysis_paths: tuple[Path, ...]
    search_text: str
    issues: tuple[str, ...]


@dataclass(frozen=True)
class ResultItemIssue:
    """An unreadable item that cannot be represented by a run row.

    entry is the Database entry name, or None for the Database root. path is the
    absolute rejected/unreadable location. reason preserves the read or path
    containment error. No contents below an escaping path have been indexed.
    """

    entry: str | None
    path: Path
    reason: str


@dataclass(frozen=True)
class ResultsIndex:
    """Detached project query snapshot; no open files, locks or live entry state.

    project_root is absolute. runs is sorted by entry then run-directory name.
    issues records entries/run directories that could not be safely enumerated;
    per-file and Result ledger failures remain in ResultRun.issues.
    Member selectors are exact indexed filenames, never arbitrary paths.
    Filesystem/schema errors propagate from document reads. Refresh by calling
    scan_results again; membership is fixed but reads see current file contents.
    """

    project_root: Path
    runs: tuple[ResultRun, ...]
    issues: tuple[ResultItemIssue, ...] = ()

    def get_run(self, key: str) -> ResultRun:
        """Return the run with opaque key; unknown keys raise KeyError."""
        for run in self.runs:
            if run.key == key:
                return run
        raise KeyError(f"unknown run key: {key!r}")

    def read_metadata(self, key: str) -> JsonObject:
        """Read the saved metadata JSON without admitting it as current RunMetadata.

        Unknown keys raise KeyError; absent data.h5 raises FileNotFoundError.
        Container/document errors and containment checks match read_cfg.
        """
        run = self.get_run(key)
        _checked_run(self.project_root, run.entry, run.run_path)
        if run.data_path is None:
            raise FileNotFoundError(run.run_path / "data.h5")
        return read_doc(_inside(run.run_path, run.data_path), "metadata")

    def read_cfg(self, key: str) -> JsonObject:
        """Read the full native cfg envelope for key, without arrays.

        Unknown keys raise KeyError; absent data.h5 raises FileNotFoundError.
        Missing docs and invalid containers retain datafile.read_doc errors.
        """
        run = self.get_run(key)
        _checked_run(self.project_root, run.entry, run.run_path)
        if run.data_path is None:
            raise FileNotFoundError(run.run_path / "data.h5")
        return read_doc(_inside(run.run_path, run.data_path), "cfg")

    def read_analysis(self, key: str, selector: str) -> JsonObject:
        """Read a finite JSON object from an indexed analysis filename.

        Unknown keys/selectors raise KeyError; invalid JSON raises ValueError.
        No producer envelope fields are removed or interpreted.
        """
        run = self.get_run(key)
        _checked_run(self.project_root, run.entry, run.run_path)
        path = _select_member(run.analysis_paths, selector, kind="analysis")
        return _read_json(_inside(run.run_path, path))

    def read_figure(self, key: str, selector: str) -> bytes:
        """Read PNG bytes for an exact indexed filename; unknowns raise KeyError.

        Paths resolving outside the indexed run root raise ValueError, including
        symlinks changed after scanning. I/O errors propagate without writes.
        """
        run = self.get_run(key)
        _checked_run(self.project_root, run.entry, run.run_path)
        path = _select_member(run.figure_paths, selector, kind="figure")
        return _inside(run.run_path, path).read_bytes()


def scan_results(
    project_root: Path, *, entries: tuple[str, ...] | None = None
) -> ResultsIndex:
    """Scan project_root/Database and same-named project_root/result ledgers.

    None selects all Database entries; entries limits to safe single-segment
    names (invalid names raise ValueError). Missing roots produce no runs.
    Inspect run directories even without native data; no ledger-only acquisitions
    are listed. Without readable native metadata, match acquired events through
    successful saved events' exact member paths, never directory-name suffixes.
    Preserve unknown identities rather than guessing experiment or time from names.
    Isolate setup, ledger and native/document failures in the affected runs'
    issues. Missing setup means no Result entry, not an error. A failed ledger
    leaves that entry's runs unregistered. Escaping members are excluded and
    diagnosed without following them. Unenumerable entries/runs produce named
    ResultsIndex.issues, not synthetic runs. Never create files or locks.
    """
    root = project_root.resolve()
    if entries is not None:
        for entry in entries:
            validate_path_segment(entry, field="entry")
    database = root / "Database"
    try:
        _inside(database, database)
        if not database.exists():
            return ResultsIndex(root, ())
        names = (
            sorted(set(entries))
            if entries is not None
            else sorted(path.name for path in database.iterdir())
        )
    except (OSError, ValueError) as error:
        return ResultsIndex(root, (), (ResultItemIssue(None, database, str(error)),))
    runs: list[ResultRun] = []
    issues: list[ResultItemIssue] = []
    for entry in names:
        runs.extend(_scan_entry(root, entry, issues))
    return ResultsIndex(root, tuple(runs), tuple(issues))


def _scan_entry(
    root: Path, entry: str, issues: list[ResultItemIssue]
) -> list[ResultRun]:
    location = root / "Database" / entry
    try:
        _inside(location, location)
        if not location.is_dir():
            return []
        location /= "runs"
        _inside(location, location)
        if not location.exists():
            return []
        paths = sorted(location.iterdir())
    except (OSError, ValueError) as error:
        issues.append(ResultItemIssue(entry, location, str(error)))
        return []
    entry_issues: list[str] = []
    events = _entry_events(root, entry, entry_issues)
    try:
        acquisitions = _acquisitions(events)
        saved_outputs = {
            Path(path): saved.run_id
            for event in events
            if (saved := event.saved_paths()) is not None
            for path in saved.paths
        }
    except ValueError as error:
        # Preserve the existing entry-level ledger failure isolation.
        entry_issues.append(str(error))
        acquisitions, saved_outputs = {}, {}
    runs: list[ResultRun] = []
    for path in paths:
        try:
            _checked_run(root, entry, path)
            if path.is_dir():
                runs.append(
                    _scan_run(entry, path, acquisitions, saved_outputs, entry_issues)
                )
        except (OSError, ValueError) as error:
            issues.append(ResultItemIssue(entry, path, str(error)))
    return runs


def _entry_events(
    root: Path, entry: str, issues: list[str]
) -> tuple[LedgerEventContent, ...]:
    result = root / "result"
    setup = result / entry / PARAMETERS_FILENAME
    try:
        _inside(result, result)
        _inside(result / entry, result / entry)
        _inside(result / entry, setup)
        if not setup.exists():
            return ()
        return ResultEntry.read_contents_snapshot(entry, result_root=result)
    except (OSError, ValueError) as error:
        issues.append(f"{setup}: {error}")
        return ()


def _inside(root: Path, path: Path) -> Path:
    # Keep root lexical: resolving both sides would trust a replaced ancestor.
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"{path}: path escapes indexed root {root}")
    return path


def _checked_run(project: Path, entry: str, run: Path) -> None:
    _inside(project, project)
    database = _inside(project / "Database", project / "Database")
    entry_root = _inside(database / entry, database / entry)
    runs = _inside(entry_root / "runs", entry_root / "runs")
    _inside(runs, run)
    _inside(run, run)


def _select_member(paths: tuple[Path, ...], selector: str, *, kind: str) -> Path:
    for path in paths:
        if path.name == selector:
            return path
    raise KeyError(f"unknown {kind} selector: {selector!r}")


def _read_json(path: Path) -> JsonObject:
    try:
        return _JSON.validate_json(path.read_bytes())
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error


def _acquisitions(
    events: tuple[LedgerEventContent, ...],
) -> Mapping[str, LedgerAcquisition]:
    return {
        acquired.payload.run_id: acquired
        for event in events
        if (acquired := event.acquisition()) is not None
    }


def _members(
    run: Path, folder: str, pattern: str, issues: list[str]
) -> tuple[Path, ...]:
    directory = run / folder
    try:
        _inside(run, directory)
        if not directory.exists():
            return ()
        paths = sorted(path for path in directory.iterdir() if path.match(pattern))
    except (OSError, ValueError) as error:
        issues.append(f"{directory}: {error}")
        return ()
    members: list[Path] = []
    for path in paths:
        try:
            _inside(run, path)
            if path.is_file():
                members.append(path)
        except (OSError, ValueError) as error:
            issues.append(f"{path}: {error}")
    return tuple(members)


def _metadata(path: Path | None, issues: list[str]) -> _SavedMetadata | None:
    if path is None:
        return None
    try:
        return _SavedMetadata.model_validate(read_doc(path, "metadata"))
    except (OSError, KeyError, ValueError) as error:
        # A broken file must remain visible and selectable in the browser.
        issues.append(f"{path}: {error}")
        return None


def _search_documents(
    data: Path | None, analysis: tuple[Path, ...], issues: list[str]
) -> list[str]:
    texts: list[str] = []
    if data is not None:
        try:
            texts.append(json.dumps(read_doc(data, "cfg"), ensure_ascii=False))
        except (OSError, KeyError, ValueError) as error:
            issues.append(f"{data}: cfg: {error}")
    for path in analysis:
        try:
            texts.append(json.dumps(_read_json(path), ensure_ascii=False))
        except (OSError, ValueError) as error:
            issues.append(f"{path}: {error}")
    return texts


def _scan_run(
    entry: str,
    path: Path,
    acquisitions: Mapping[str, LedgerAcquisition],
    saved_outputs: Mapping[Path, str],
    entry_issues: list[str],
) -> ResultRun:
    issues = list(entry_issues)
    data_path = path / "data.h5"
    try:
        _inside(path, data_path)
        data = data_path if data_path.is_file() else None
    except (OSError, ValueError) as error:
        issues.append(f"{data_path}: {error}")
        data = None
    figures = _members(path, "figures", "*.png", issues)
    analysis = _members(path, "analysis", "*.json", issues)
    metadata = _metadata(data, issues)
    event = next(
        (
            acquisitions[saved_outputs[member]]
            for member in (data_path, *figures, *analysis)
            if member in saved_outputs and saved_outputs[member] in acquisitions
        ),
        None,
    )
    if metadata is not None and metadata.run_id is not None:
        event = acquisitions.get(metadata.run_id)
    payload = event.payload if event is not None else None
    acquired = payload if isinstance(payload, AcquiredPayload) else None
    run_id = metadata.run_id if metadata is not None else None
    experiment = metadata.experiment if metadata is not None else None
    point = metadata.snapshot.point if metadata is not None else None
    roles = metadata.snapshot.roles if metadata is not None else {}
    at = metadata.started_at if metadata is not None else None
    if acquired is not None and event is not None:
        run_id, experiment, point = acquired.run_id, acquired.experiment, acquired.point
        roles, at = acquired.roles, event.at
    texts = [
        entry,
        run_id or "",
        experiment or "",
        point or "",
        json.dumps(dict(roles)),
    ]
    texts.extend(_search_documents(data, analysis, issues))
    return ResultRun(
        key=f"{entry}/{path.name}",
        entry=entry,
        run_id=run_id,
        experiment=experiment,
        point=point,
        roles=MappingProxyType(dict(roles)),
        at=at,
        registered=acquired is not None,
        run_path=path,
        data_path=data,
        figure_paths=figures,
        analysis_paths=analysis,
        search_text=" ".join(texts).casefold(),
        issues=tuple(issues),
    )
