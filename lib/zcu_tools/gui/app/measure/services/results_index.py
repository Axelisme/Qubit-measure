"""Read-only project-wide result queries shared by the browser and RPC."""

import errno
import json
import os
import stat
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from zcu_tools.datafile import JsonObject, read_doc, read_docs
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
    including tone names absent from roles. issues holds global/ledger failures
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
class ResultEntryBatch:
    """Detached rows and item issues from one completed Database entry.

    project_root is the canonical absolute scan root. entry is the scanned name.
    completed_entries counts this and earlier names;
    total_entries counts all names selected at the start, including unreadable
    or empty entries. runs is ordered by run directory name. issues records
    paths within this entry that could not be safely enumerated.
    """

    project_root: Path
    entry: str
    completed_entries: int
    total_entries: int
    runs: tuple[ResultRun, ...]
    issues: tuple[ResultItemIssue, ...]


@dataclass(frozen=True)
class ResultsIndex:
    """Detached project query snapshot; no open files, locks or live entry state.

    project_root is absolute. runs is sorted by entry then run-directory name.
    The snapshot may contain only completed entries during a background scan;
    scan_results returns a complete snapshot on success.
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


class ResultsScanCancelled(RuntimeError):
    """A cooperative stop interrupted the scan; no complete snapshot exists."""


def scan_results(
    project_root: Path,
    *,
    entries: tuple[str, ...] | None = None,
    on_entry: Callable[[ResultEntryBatch], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> ResultsIndex:
    """Scan project_root/Database and same-named project_root/result ledgers.

    None selects all Database entries; entries limits to safe single-segment
    names (invalid names raise ValueError). Missing roots produce no runs.
    Inspect run directories even without native data; no ledger-only acquisitions
    are listed. Without readable native metadata, match acquired events through
    successful saved events' exact member paths, never directory-name suffixes.
    Preserve unknown identities rather than guessing experiment or time from names.
    Isolate global, ledger and native/document failures in the affected runs'
    issues. Missing global means no Result entry, not an error. A failed ledger
    leaves that entry's runs unregistered. Escaping members are excluded and
    diagnosed without following them. Unenumerable entries/runs produce named
    ResultsIndex.issues, not synthetic runs. Never create files or locks.

    on_entry, if given, receives a detached batch on the calling thread after
    each entry and before reading the next. Empty directories and entries with
    item issues count toward progress; loose files and absent names do not.
    Database root issues appear only in the returned complete snapshot.
    Callback exceptions propagate.

    cancelled is a calling-thread predicate (for example Event.is_set). It is
    checked before scanning, between runs, and before publishing or returning.
    True raises ResultsScanCancelled instead of returning an incomplete snapshot.
    Already delivered batches remain valid; in-flight I/O is not interrupted.
    """
    root = project_root.resolve()
    if entries is not None:
        for entry in entries:
            validate_path_segment(entry, field="entry")
    _check_cancelled(cancelled)
    names, issues = _database_entries(root / "Database", entries)
    runs: list[ResultRun] = []
    for completed, entry in enumerate(names, 1):
        _check_cancelled(cancelled)
        entry_issues: list[ResultItemIssue] = []
        entry_runs = _scan_entry(root, entry, entry_issues, cancelled)
        _check_cancelled(cancelled)
        runs.extend(entry_runs)
        issues.extend(entry_issues)
        if on_entry is not None:
            on_entry(
                ResultEntryBatch(
                    root,
                    entry,
                    completed,
                    len(names),
                    tuple(entry_runs),
                    tuple(entry_issues),
                )
            )
    _check_cancelled(cancelled)
    return ResultsIndex(root, tuple(runs), tuple(issues))


def _check_cancelled(cancelled: Callable[[], bool] | None) -> None:
    if cancelled is not None and cancelled():
        raise ResultsScanCancelled("Results scan cancelled")


def _database_entries(
    database: Path, entries: tuple[str, ...] | None
) -> tuple[list[str], list[ResultItemIssue]]:
    try:
        _inside(database, database)
        if not database.exists():
            return [], []
        candidates = (
            [database / name for name in set(entries)]
            if entries is not None
            else list(database.iterdir())
        )
    except (OSError, ValueError) as error:
        return [], [ResultItemIssue(None, database, str(error))]

    names: list[str] = []
    for path in candidates:
        try:
            info = path.lstat()
        except (FileNotFoundError, NotADirectoryError):
            continue
        except OSError:
            # Let _scan_entry report access errors at the individual entry.
            names.append(path.name)
        else:
            # Do not follow redirects here. _scan_entry checks containment and
            # diagnoses unsafe entries before looking for runs.
            if stat.S_ISDIR(info.st_mode) or _is_redirect(info):
                names.append(path.name)
    return sorted(names), []


def _scan_entry(
    root: Path,
    entry: str,
    issues: list[ResultItemIssue],
    cancelled: Callable[[], bool] | None,
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
    prefixes = tuple(
        str(parent) for parent in reversed(location.parents) if parent.name
    )
    directory = _ScanDirectory(location, (*prefixes, str(location)))
    for path in paths:
        _check_cancelled(cancelled)
        try:
            info = directory.child_stat(
                path, lambda child: _checked_run(root, entry, child)
            )
            if info is not None and stat.S_ISDIR(info.st_mode):
                runs.append(
                    _scan_run(
                        entry,
                        directory.child(path),
                        acquisitions,
                        saved_outputs,
                        entry_issues,
                    )
                )
        except (OSError, ValueError) as error:
            issues.append(ResultItemIssue(entry, path, str(error)))
    return runs


def _entry_events(
    root: Path, entry: str, issues: list[str]
) -> tuple[LedgerEventContent, ...]:
    result = root / "result"
    global_path = result / entry / PARAMETERS_FILENAME
    try:
        _inside(result, result)
        _inside(result / entry, result / entry)
        _inside(result / entry, global_path)
        if not global_path.exists():
            return ()
        return ResultEntry.read_contents_snapshot(entry, result_root=result)
    except (OSError, ValueError) as error:
        issues.append(f"{global_path}: {error}")
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


def _is_redirect(info: os.stat_result) -> bool:
    # Windows directory junctions also redirect lookup, without being symlinks.
    if os.name == "nt":
        return bool(info.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)
    return stat.S_ISLNK(info.st_mode)


@dataclass(frozen=True)
class _ScanDirectory:
    """Reuse the lexical prefix chain, never its filesystem observations."""

    path: Path
    prefixes: tuple[str, ...]

    def child(self, path: Path) -> "_ScanDirectory":
        return _ScanDirectory(path, (*self.prefixes, str(path)))

    def child_stat(
        self, path: Path, check_redirect: Callable[[Path], Path | None]
    ) -> os.stat_result | None:
        # Fresh no-follow lookups detect replaced ancestors before descending.
        # Ordinary paths need no canonicalization; redirected ones use the
        # same lexical containment checks and diagnostics as standalone reads.
        try:
            if not any(_is_redirect(os.lstat(prefix)) for prefix in self.prefixes):
                info = path.lstat()
                if not _is_redirect(info):
                    return info
            check_redirect(path)
            return path.stat()
        except OSError as error:
            if error.errno in {errno.ENOENT, errno.ENOTDIR, errno.EBADF, errno.ELOOP}:
                return None
            raise


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
    run: _ScanDirectory, folder: str, pattern: str, issues: list[str]
) -> tuple[Path, ...]:
    directory = run.path / folder
    try:
        if run.child_stat(directory, lambda child: _inside(run.path, child)) is None:
            return ()
        paths = sorted(path for path in directory.iterdir() if path.match(pattern))
    except (OSError, ValueError) as error:
        issues.append(f"{directory}: {error}")
        return ()
    children = run.child(directory)
    return tuple(
        path for path in paths if _is_member_file(children, run.path, path, issues)
    )


def _is_member_file(
    directory: _ScanDirectory, run_root: Path, path: Path, issues: list[str]
) -> bool:
    try:
        info = directory.child_stat(path, lambda child: _inside(run_root, child))
        return info is not None and stat.S_ISREG(info.st_mode)
    except (OSError, ValueError) as error:
        issues.append(f"{path}: {error}")
        return False


def _native_documents(
    path: Path | None, issues: list[str]
) -> tuple[_SavedMetadata | None, str | None]:
    if path is None:
        return None, None
    try:
        docs = read_docs(path, ("metadata", "cfg"))
    except (OSError, KeyError, ValueError) as error:
        # File-wide failures affect both documents but do not remove the run.
        issues.extend((f"{path}: {error}", f"{path}: cfg: {error}"))
        return None, None
    metadata_doc = docs["metadata"]
    metadata = None
    if isinstance(metadata_doc, Exception):
        issues.append(f"{path}: {metadata_doc}")
    else:
        try:
            metadata = _SavedMetadata.model_validate(metadata_doc)
        except ValueError as error:
            issues.append(f"{path}: {error}")
    cfg = docs["cfg"]
    if isinstance(cfg, Exception):
        issues.append(f"{path}: cfg: {cfg}")
        return metadata, None
    return metadata, json.dumps(cfg, ensure_ascii=False)


def _search_documents(analysis: tuple[Path, ...], issues: list[str]) -> list[str]:
    texts: list[str] = []
    for path in analysis:
        try:
            texts.append(json.dumps(_read_json(path), ensure_ascii=False))
        except (OSError, ValueError) as error:
            issues.append(f"{path}: {error}")
    return texts


def _scan_run(
    entry: str,
    run: _ScanDirectory,
    acquisitions: Mapping[str, LedgerAcquisition],
    saved_outputs: Mapping[Path, str],
    entry_issues: list[str],
) -> ResultRun:
    issues = list(entry_issues)
    path = run.path
    data_path = path / "data.h5"
    data = data_path if _is_member_file(run, path, data_path, issues) else None
    # Read native data while its path check is fresh, before checking siblings.
    # Keep member diagnostics ahead of document diagnostics in the returned row.
    native_issues: list[str] = []
    metadata, cfg_text = _native_documents(data, native_issues)
    figures = _members(run, "figures", "*.png", issues)
    analysis = _members(run, "analysis", "*.json", issues)
    issues.extend(native_issues)
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
    if cfg_text is not None:
        texts.append(cfg_text)
    texts.extend(_search_documents(analysis, issues))
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
