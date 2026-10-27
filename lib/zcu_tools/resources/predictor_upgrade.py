"""Explicit predictor v1-to-v2 file upgrades; runtime readers never invoke this."""

import json
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import uuid4

from filelock import FileLock, Timeout

from zcu_tools.resources.predictor_file import (
    PredictorFile,
    save_predictor_file,
    upgrade_predictor_document,
)


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
    paths: Sequence[Path], *, apply: bool = False
) -> tuple[PredictorUpgradeResult, ...]:
    """Preview or explicitly upgrade exact predictor JSON paths independently.

    Default dry-run validates/converts without creating files or locks. Apply
    re-reads under a sidecar lock, backs up original bytes exclusively, and uses
    the v2 owner's atomic writer. Current v2 files validate then skip unchanged.
    Malformed/unsupported/incomplete v1 input, I/O and lock failures become
    rejected results without stopping siblings. Unexpected failures propagate.
    """
    return tuple(_upgrade_file(path, apply=apply) for path in paths)


def _upgrade_file(path: Path, *, apply: bool) -> PredictorUpgradeResult:
    backup: Path | None = None
    try:
        path.parent.stat()
        lock = FileLock(f"{path}.lock", timeout=5) if apply else nullcontext()
        with lock:
            original = path.read_bytes()
            document: object = json.loads(original.decode("utf-8"))
            candidate = upgrade_predictor_document(document)
            if isinstance(document, dict) and document["version"] == 2:
                return PredictorUpgradeResult(path, "skipped", candidate)
            if not apply:
                return PredictorUpgradeResult(path, "dry-run", candidate)
            backup_candidate = path.with_name(f"{path.name}.{uuid4().hex}.bak")
            with backup_candidate.open("xb") as stream:
                stream.write(original)
            backup = backup_candidate
            save_predictor_file(path, candidate)
            return PredictorUpgradeResult(path, "applied", candidate, backup)
    except (OSError, ValueError, Timeout) as error:
        return PredictorUpgradeResult(
            path, "rejected", backup_path=backup, error=str(error)
        )
