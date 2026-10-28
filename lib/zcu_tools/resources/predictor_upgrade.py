"""Explicit predictor v1-to-v2 file upgrades; runtime readers never invoke this."""

import json
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from filelock import FileLock, Timeout

from zcu_tools.resources.file_lock import lock_path_for
from zcu_tools.resources.predictor_file import (
    PredictorFile,
    save_predictor_file,
    upgrade_predictor_document,
)
from zcu_tools.resources.upgrade_backup import upgrade_backup_path


@dataclass(frozen=True)
class PredictorUpgradeResult:
    """One independent predictor upgrade outcome.

    path is the requested JSON file. status is dry-run, applied, skipped (valid
    v2) or rejected. candidate holds detached v2 inputs on success. backup_path
    names the original-byte backup when created, even if replacement failed.
    error describes rejection only. No status implies a multi-file transaction.
    """

    path: Path
    status: Literal["dry-run", "applied", "skipped", "rejected"]
    candidate: PredictorFile | None = None
    backup_path: Path | None = None
    error: str | None = None


def upgrade_predictor_files(
    paths: Sequence[Path], *, apply: bool = False, backup_root: Path | None = None
) -> tuple[PredictorUpgradeResult, ...]:
    """Preview or explicitly upgrade exact predictor JSON paths independently.

    Default dry-run validates/converts without creating files or locks. Apply
    re-reads under a per-user runtime lock. It backs up original bytes exclusively,
    then writes atomically through the v2 owner. Current v2 files validate then
    skip unchanged. backup_root overrides the external project sibling root;
    see upgrade_backup_path for placement and pre-write rejection rules.
    Malformed/unsupported/incomplete v1 input, I/O and lock failures become
    rejected results without stopping siblings. Unexpected failures propagate.
    """
    return tuple(
        _upgrade_file(path, apply=apply, backup_root=backup_root) for path in paths
    )


def _upgrade_file(
    path: Path, *, apply: bool, backup_root: Path | None
) -> PredictorUpgradeResult:
    backup: Path | None = None
    try:
        backup_candidate = (
            upgrade_backup_path(path, backup_root=backup_root) if apply else None
        )
        path.parent.stat()
        lock = FileLock(str(lock_path_for(path)), timeout=5) if apply else nullcontext()
        with lock:
            original = path.read_bytes()
            document: object = json.loads(original.decode("utf-8"))
            candidate = upgrade_predictor_document(document)
            if isinstance(document, dict) and document["version"] == 2:
                return PredictorUpgradeResult(path, "skipped", candidate)
            if not apply:
                return PredictorUpgradeResult(path, "dry-run", candidate)
            if backup_candidate is None:
                raise RuntimeError("apply requires a backup destination")
            backup_candidate.parent.mkdir(parents=True, exist_ok=True)
            with backup_candidate.open("xb") as stream:
                stream.write(original)
            backup = backup_candidate
            save_predictor_file(path, candidate)
            return PredictorUpgradeResult(path, "applied", candidate, backup)
    except (OSError, ValueError, Timeout) as error:
        return PredictorUpgradeResult(
            path, "rejected", backup_path=backup, error=str(error)
        )
