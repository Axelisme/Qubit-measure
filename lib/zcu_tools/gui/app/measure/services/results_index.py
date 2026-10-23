"""Read-only project-wide result queries shared by the browser and RPC."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from zcu_tools.datafile import JsonObject


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
        raise KeyError(key)

    def read_cfg(self, key: str) -> JsonObject:
        """Read the full native cfg envelope for key, without arrays.

        Unknown keys raise KeyError; absent data.h5 raises FileNotFoundError.
        Missing docs and invalid containers retain datafile.read_doc errors.
        """
        self.get_run(key)
        raise FileNotFoundError(key)

    def read_analysis(self, key: str, selector: str) -> JsonObject:
        """Read a finite JSON object from an indexed analysis filename.

        Unknown keys/selectors raise KeyError; invalid JSON raises ValueError.
        No producer envelope fields are removed or interpreted.
        """
        self.get_run(key)
        raise KeyError(selector)

    def read_figure(self, key: str, selector: str) -> bytes:
        """Read PNG bytes for an exact indexed filename; unknowns raise KeyError.

        Paths resolving outside the indexed run root raise ValueError, including
        symlinks changed after scanning. I/O errors propagate without writes.
        """
        self.get_run(key)
        raise KeyError(selector)


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
    return ResultsIndex(project_root.resolve(), ())
