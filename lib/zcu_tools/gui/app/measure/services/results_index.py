"""Read-only project-wide result queries shared by the browser and RPC."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType

from pydantic import TypeAdapter

from zcu_tools.datafile import JsonObject, read_doc
from zcu_tools.experiment.metadata import RunMetadata
from zcu_tools.experiment.native_persistence import read_run_metadata
from zcu_tools.resources.entry import (
    AcquiredPayload,
    LedgerEvent,
    RecordsLedger,
    ResultEntry,
    SaveLayout,
    validate_path_segment,
)

_JSON = TypeAdapter(JsonObject)


@dataclass(frozen=True)
class ResultRun:
    """One saved run directory, including incomplete or unregistered outputs.

    key is an opaque project-local selector, not a caller-supplied path. entry
    names the Database entry. run_id/experiment/point are historical identities;
    None means unavailable (point also represents global runs). roles contains
    only captured role-to-component bindings. at is the acquired UTC timestamp,
    else metadata.started_at, else None; never a filesystem timestamp.
    registered means a matching acquired event exists in the same-named Result
    entry. run_path is the absolute exact run directory. data_path is an existing
    data.h5 or None. figure_paths/analysis_paths are sorted existing PNG/JSON
    members. search_text is casefolded identity, roles, cfg and analysis content,
    including tone names absent from roles. issues holds per-file read failures.
    """

    key: str
    entry: str
    run_id: str | None
    experiment: str | None
    point: str | None
    roles: Mapping[str, str]
    at: str | None
    registered: bool
    run_path: Path
    data_path: Path | None
    figure_paths: tuple[Path, ...]
    analysis_paths: tuple[Path, ...]
    search_text: str
    issues: tuple[str, ...]


@dataclass(frozen=True)
class ResultsIndex:
    """Detached project query snapshot; no open files, locks or live entry state.

    project_root is absolute. runs is sorted by entry then run-directory name.
    Member selectors are exact indexed filenames, never arbitrary paths.
    Filesystem/schema errors propagate from document reads. Refresh by calling
    scan_results again; membership is fixed but reads see current file contents.
    """

    project_root: Path
    runs: tuple[ResultRun, ...]

    def get_run(self, key: str) -> ResultRun:
        """Return the run with opaque key; unknown keys raise KeyError."""
        for run in self.runs:
            if run.key == key:
                return run
        raise KeyError(f"unknown run key: {key!r}")

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
    are listed. Preserve unknown identities rather than parsing directory names.
    Isolate malformed native/doc files in ResultRun.issues. Ledger/schema errors
    propagate. Escaping symlinks raise ValueError. Never create files or locks.
    """
    root = project_root.resolve()
    if entries is not None:
        for entry in entries:
            validate_path_segment(entry, field="entry")
    database = _inside(root / "Database", root / "Database")
    if not database.exists():
        return ResultsIndex(root, ())
    names = (
        sorted(set(entries))
        if entries is not None
        else sorted(path.name for path in database.iterdir() if path.is_dir())
    )
    runs: list[ResultRun] = []
    for entry in names:
        entry_path = _inside(database / entry, database / entry)
        run_root = _inside(entry_path / "runs", entry_path / "runs")
        if not run_root.exists():
            continue
        acquisitions = _acquisitions(root, entry)
        for path in sorted(run_root.iterdir()):
            if path.is_dir():
                _checked_run(root, entry, path)
                runs.append(_scan_run(entry, path, acquisitions))
    return ResultsIndex(root, tuple(runs))


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


@dataclass(frozen=True)
class _Acquisitions:
    by_id: Mapping[str, LedgerEvent]
    by_path: Mapping[Path, LedgerEvent]


def _acquisitions(project: Path, entry: str) -> _Acquisitions:
    result_root = _inside(project / "result", project / "result")
    result = _inside(result_root / entry, result_root / entry)
    if not result.exists():
        return _Acquisitions({}, {})
    identity = ResultEntry.read_identity(entry, result_root=result_root)
    records = _inside(result, result / "records")
    if not records.exists():
        return _Acquisitions({}, {})
    _inside(records, records / "ledger.jsonl")
    events = RecordsLedger(
        records, entry_id=str(identity.entry_id)
    ).read_events_snapshot()
    by_id: dict[str, LedgerEvent] = {}
    by_path: dict[Path, LedgerEvent] = {}
    for event in events:
        if isinstance(event.payload, AcquiredPayload):
            payload = event.payload
            layout = SaveLayout(
                result_path=result,
                database_path=project / "Database" / entry,
                experiment=payload.experiment,
                run_id=payload.run_id,
                point=payload.point,
                saved_at=datetime.fromisoformat(event.at),
            )
            by_id[payload.run_id] = event
            by_path[layout.run_path] = event
    return _Acquisitions(by_id, by_path)


def _members(run: Path, folder: str, pattern: str) -> tuple[Path, ...]:
    directory = _inside(run, run / folder)
    return tuple(
        _inside(run, path) for path in sorted(directory.glob(pattern)) if path.is_file()
    )


def _metadata(path: Path | None, issues: list[str]) -> RunMetadata | None:
    if path is None:
        return None
    try:
        return read_run_metadata(path)
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


def _scan_run(entry: str, path: Path, acquisitions: _Acquisitions) -> ResultRun:
    data_path = _inside(path, path / "data.h5")
    data = data_path if data_path.is_file() else None
    figures = _members(path, "figures", "*.png")
    analysis = _members(path, "analysis", "*.json")
    issues: list[str] = []
    metadata = _metadata(data, issues)
    event = acquisitions.by_path.get(path)
    if metadata is not None:
        event = acquisitions.by_id.get(metadata.run_id)
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
