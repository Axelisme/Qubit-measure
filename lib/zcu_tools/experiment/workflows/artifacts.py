"""Engine-owned run artifacts and no-clobber single-experiment publication."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath, PureWindowsPath
from re import fullmatch
from threading import RLock
from typing import Literal
from uuid import uuid4

from pydantic import JsonValue, TypeAdapter

from .journal import (
    IterationResult,
    IterationStarted,
    JournalPayload,
    RunMetadata,
    RunStarted,
)
from .models import (
    Capability,
    CommittedRecord,
    Completed,
    Lifecycle,
    RunIdentity,
    RunPaths,
)
from .ports import Clock


@dataclass(frozen=True)
class RunManifest:
    """Run entry document, not a checkpoint or point-record store.

    identity/workflow/plan/initial_tunables/requires/roots retain startup
    provenance. roots and journal are absolute paths; journal locates JSONL.
    started_at is the original UTC start;
    updated_at is the last lifecycle change, ended_at is None until terminal.
    lifecycle uses the Engine lifecycle enum; reason is diagnostic or None.
    format and format_version identify this document's schema.
    """

    identity: RunIdentity
    workflow: str
    plan: JsonValue
    initial_tunables: JsonValue
    requires: tuple[Capability, ...]
    roots: RunPaths
    journal: Path
    started_at: datetime
    updated_at: datetime
    ended_at: datetime | None
    lifecycle: Lifecycle
    reason: str | None
    format: Literal["zcu-workflow-run"] = field(default="zcu-workflow-run", init=False)
    format_version: Literal[1] = field(default=1, init=False)


_payload_adapter = TypeAdapter(JournalPayload)
_manifest_adapter = TypeAdapter(RunManifest)
_json_adapter = TypeAdapter(JsonValue)


def _json_object[T](value: T, adapter: TypeAdapter[T]) -> dict[str, JsonValue]:
    projected = _json_adapter.validate_python(
        adapter.dump_python(value, mode="json", warnings="error")
    )
    if not isinstance(projected, dict):
        raise TypeError("Artifact payload must encode as a JSON object")
    return projected


def _json_line(value: dict[str, JsonValue]) -> str:
    # Necessary provenance must fail, not inherit the record encoder's fallback.
    return json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n"


def _check_utc(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError("Artifact timestamps must be timezone-aware UTC")


def _relative_path(value: str) -> None:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or PureWindowsPath(value).drive
        or ".." in path.parts
        or "\\" in value
    ):
        raise ValueError("Artifact references must be relative paths without '..'")


def _check_event_paths(event: JournalPayload) -> None:
    if isinstance(event, (IterationStarted, IterationResult, CommittedRecord)):
        _relative_path(event.iteration_dir)
    if isinstance(event, (IterationResult, CommittedRecord)):
        for path in event.run_files:
            _relative_path(path)


class RunArtifacts:
    """Create and maintain one run's metadata and iteration directories.

    metadata contains detached, validated startup provenance. clock supplies
    timezone-aware UTC. control_lock must be the Engine's shared RLock, also
    used for revision capture and control publication. All appends use it.

    Construction resolves roots to absolute paths, rejects existing or
    overlapping roots, creates both roots, writes a running manifest and a
    run_started line. No workflow init or hardware executes here. Failures
    propagate and retain any directories/files already created.

    This package-internal I/O seam is not exported by the workflow barrel.
    The Engine owns lifecycle transitions and state publication after I/O.
    """

    def __init__(
        self, metadata: RunMetadata, clock: Clock, control_lock: RLock
    ) -> None:
        roots = RunPaths(
            metadata.roots.metadata_root.resolve(), metadata.roots.data_root.resolve()
        )
        _check_utc(metadata.started_at)
        self._check_roots(roots)
        metadata = deepcopy(replace(metadata, roots=roots))
        self._clock = clock
        self._lock = control_lock
        self._paths = roots
        self._run_id = metadata.identity.run_id
        self._seq = 0
        self._journal_broken = False
        self._manifest = RunManifest(
            identity=metadata.identity,
            workflow=metadata.workflow,
            plan=metadata.plan,
            initial_tunables=metadata.tunables,
            requires=metadata.requires,
            roots=roots,
            journal=self.journal_path,
            started_at=metadata.started_at,
            updated_at=metadata.started_at,
            ended_at=None,
            lifecycle="running",
            reason=None,
        )
        # Encode mandatory fields before creating either run directory.
        _json_line(_json_object(self._manifest, _manifest_adapter))
        roots.metadata_root.mkdir(parents=True, exist_ok=False)
        roots.data_root.mkdir(parents=True, exist_ok=False)
        self._write_manifest(self._manifest)
        self.journal_path.touch(exist_ok=False)
        self.append(
            RunStarted(
                identity=metadata.identity,
                workflow=metadata.workflow,
                plan=metadata.plan,
                tunables=metadata.tunables,
                requires=metadata.requires,
                roots=roots,
                started_at=metadata.started_at,
            )
        )

    @staticmethod
    def _check_roots(roots: RunPaths) -> None:
        first, second = roots.metadata_root, roots.data_root
        if first == second or first in second.parents or second in first.parents:
            raise ValueError("Run roots must be distinct and non-overlapping")
        for root in (first, second):
            if root.exists():
                raise FileExistsError(f"Run root already exists: {root}")

    @property
    def paths(self) -> RunPaths:
        """Return the normalized absolute run roots (an immutable path pair)."""
        return self._paths

    @property
    def journal_path(self) -> Path:
        """Return the absolute UTF-8 journal.jsonl path in metadata_root."""
        return self._paths.metadata_root / "journal.jsonl"

    def append(self, event: JournalPayload) -> None:
        """Append and flush one typed event with the next contiguous sequence.

        The common envelope adds format_version=1, seq, run_id, and UTC time.
        event path references must be relative, without '..'. Serialization
        errors propagate without writing or consuming a sequence. Write/flush
        failures propagate and poison further appends, since a partial line may
        exist. No fsync or crash-recovery guarantee is provided.
        """
        with self._lock:
            if self._journal_broken:
                raise RuntimeError("Journal is unusable after an I/O failure")
            _check_event_paths(event)
            payload = _json_object(event, _payload_adapter)
            now = self._clock.now()
            _check_utc(now)
            payload.update(
                format_version=1,
                seq=self._seq + 1,
                run_id=self._run_id,
                time=now.isoformat(),
            )
            line = _json_line(payload)
            self._append_line(line)
            self._seq += 1

    def _append_line(self, line: str) -> None:
        try:
            with self.journal_path.open("a", encoding="utf-8", newline="\n") as writer:
                if writer.write(line) != len(line):
                    raise OSError("Journal write was incomplete")
                writer.flush()
        except OSError:
            # A short/failed write cannot safely be followed by a failure event.
            self._journal_broken = True
            raise

    def set_lifecycle(self, lifecycle: Lifecycle, reason: str | None = None) -> None:
        """Replace the manifest for a lifecycle change, preserving startup data.

        lifecycle is the Engine-selected new state; transition policy belongs
        to Engine. The same state raises ValueError. Terminal states set
        ended_at to clock.now(); other states use None. Only successful replace
        updates this store's manifest. Errors retain temporary files and
        propagate; the Engine must report failure and append it if possible.
        """
        with self._lock:
            if lifecycle == self._manifest.lifecycle:
                raise ValueError("Manifest updates require a lifecycle change")
            now = self._clock.now()
            _check_utc(now)
            terminal = lifecycle in ("done", "aborted", "stopped", "failed")
            manifest = replace(
                self._manifest,
                updated_at=now,
                ended_at=now if terminal else None,
                lifecycle=lifecycle,
                reason=reason,
            )
            self._write_manifest(manifest)
            self._manifest = manifest

    def _write_manifest(self, manifest: RunManifest) -> None:
        line = _json_line(_json_object(manifest, _manifest_adapter))
        temporary = self._paths.metadata_root / f".manifest-{uuid4().hex}.tmp.json"
        with temporary.open("x", encoding="utf-8", newline="\n") as writer:
            if writer.write(line) != len(line):
                raise OSError("Manifest write was incomplete")
            writer.flush()
        temporary.replace(self._paths.metadata_root / "manifest.json")

    def new_iteration(self, call_seq: int) -> IterationArtifacts:
        """Create iter/<call_seq>/runs and files and return its publication seam.

        call_seq is a positive invocation number, not a commit or flux index.
        It is padded to at least six digits and never wraps. Existing invocation
        directories are rejected; partially created directories remain on error.
        This operation does not append iteration_started or commit anything.
        """
        if type(call_seq) is not int or call_seq <= 0:
            raise ValueError("call_seq must be a positive integer")
        relative = f"iter/{call_seq:06d}"
        root = self._paths.data_root / relative
        root.mkdir(parents=True, exist_ok=False)
        (root / "runs").mkdir()
        (root / "files").mkdir()
        return IterationArtifacts(root, relative)


class IterationArtifacts:
    """Publish Completed results in one newly allocated invocation directory.

    root is an absolute directory with existing runs/ and files/ children.
    iteration_dir is its data-root-relative journal reference. RunArtifacts
    constructs this object; the Engine is its sole writer. It neither inspects
    nor scans workflow-owned files/ or the saver-owned file format.
    """

    def __init__(self, root: Path, iteration_dir: str) -> None:
        _relative_path(iteration_dir)
        if not root.is_absolute():
            raise ValueError("Iteration root must be absolute")
        self._root = root
        self._iteration_dir = iteration_dir
        self._run_files: dict[int, str] = {}

    @property
    def iteration_dir(self) -> str:
        """Return the data-root-relative invocation reference, iter/<call_seq>."""
        return self._iteration_dir

    @property
    def files_dir(self) -> Path:
        """Return the absolute workflow-owned files/ directory exposed in env."""
        return self._root / "files"

    @property
    def run_files(self) -> tuple[str, ...]:
        """Return published finals relative to this invocation, in run_no order.

        Failed experiments leave gaps. Temporary files are never listed; finals
        published before a later cleanup error are retained and listed.
        """
        return tuple(self._run_files[number] for number in sorted(self._run_files))

    def save[Cfg, Result](
        self,
        run_no: int,
        experiment: str,
        completed: Completed[Cfg, Result],
        saver: Callable[[Completed[Cfg, Result], Path], None],
    ) -> None:
        """Save to a unique temporary .h5 and publish without overwriting finals.

        run_no counts experiment effects from 1, including recoverable failures.
        experiment is the validated ASCII function-name stem. completed is the
        paired cfg/Result; saver writes/closes the exact supplied temporary path.
        Only successful publication is added to run_files. Saver, link, and
        unlink errors propagate; files are retained, no retry occurs. After link
        succeeds, an unlink error retains and lists the final and fails the run.
        The Engine must call this before handing Completed back to workflow.
        """
        if type(run_no) is not int or run_no <= 0:
            raise ValueError("run_no must be a positive integer")
        if fullmatch(r"[A-Za-z0-9_]+", experiment) is None:
            raise ValueError("Experiment stem must contain only ASCII letters/digits/_")
        stem = f"{run_no:02d}-{experiment}"
        final = self._root / "runs" / f"{stem}.h5"
        if run_no in self._run_files or final.exists():
            raise FileExistsError(f"Run file already exists: {final}")
        temporary = final.with_name(f".{stem}-{uuid4().hex}.tmp.h5")
        saver(completed, temporary)
        # Same-directory hard link is atomic and refuses an existing final.
        os.link(temporary, final)
        self._run_files[run_no] = f"runs/{final.name}"
        temporary.unlink()
