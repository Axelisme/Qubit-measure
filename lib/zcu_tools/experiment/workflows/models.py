"""Typed workflow results, control observations, and provenance."""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import JsonValue

from .ports import DeviceSnapshot

type Capability = Literal["soc", "devices", "context"]
type Lifecycle = Literal[
    "running", "pausing", "paused", "stopping", "done", "aborted", "stopped", "failed"
]


class Effect:
    """Type marker for core-owned suspension requests, not an extension point.

    Workflow code receives requests through ``yield from env.<effect>(...)``.
    Direct construction is rejected; the engine also rejects non-core requests.
    """

    def __init__(self) -> None:
        raise TypeError("Use yield from env.<effect>(...), not Effect()")


@dataclass(frozen=True)
class Next[R, S]:
    """Commit one step: ``record`` is its summary, ``state`` is the next input.

    Both values must support deepcopy. Curve data belongs in saved run files,
    not in record. The engine owns the commit and copies both values.
    """

    record: R
    state: S


@dataclass(frozen=True)
class Done:
    """Finish successfully without a new point record or state commit."""


@dataclass(frozen=True)
class Aborted:
    """Finish deliberately; ``reason`` is a nonempty human-readable explanation."""

    reason: str

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError("Aborted.reason must not be empty")


type Step[S, R] = Generator[Effect, None, Next[R, S] | Done | Aborted]


@dataclass(frozen=True)
class Completed[Cfg, Result]:
    """Saved successful experiment output.

    ``cfg`` is a deepcopy of run.cfg at experiment return, never the last
    Schedule's local cfg. ``result`` is the experiment's returned data.
    The engine invokes the saver before returning this value to the workflow.
    """

    cfg: Cfg
    result: Result


@dataclass(frozen=True)
class Failed:
    """Recoverable Schedule failure; ``reason`` describes its original cause.

    No partial Result or run file accompanies this value. The journal retains
    the original exception diagnostics; workflow code decides how to continue.
    """

    reason: str


@dataclass(frozen=True)
class Actor:
    """Identity of a tunables editor.

    ``kind`` is user or agent. ``name`` is the nonempty operator/agent name.
    """

    kind: Literal["user", "agent"]
    name: str

    def __post_init__(self) -> None:
        if self.kind not in ("user", "agent") or not self.name:
            raise ValueError("Actor requires kind user/agent and a nonempty name")


@dataclass(frozen=True)
class TunableChange:
    """One leaf replacement in a tunables batch.

    ``path`` is a model-field dot path, never an index or expression. ``value``
    is a JSON leaf or a complete list/tuple replacement in JSON form.
    Batch overlap, schema, and revision checks belong to Engine.update_tunables.
    """

    path: str
    value: JsonValue


@dataclass(frozen=True)
class TunablesSnapshot:
    """Detached control view: ``revision`` is its version, ``values`` is JSON.

    Values contain the complete tunables model, not only the most recent patch.
    """

    revision: int
    values: JsonValue


@dataclass(frozen=True)
class ProgressSnapshot:
    """Display-only progress, not proof of a committed step.

    ``name`` identifies the bar. ``total`` is its backend's target, or None.
    ``completed`` is the absolute displayed progress, never an increment.
    """

    name: str
    total: float | None
    completed: float


@dataclass(frozen=True)
class RunStatus:
    """Detached state of the engine's current or last run.

    ``run_id`` is the host's run identity; ``workflow`` is the declared name.
    ``lifecycle`` is running, pausing, paused, stopping, done, aborted, stopped,
    or failed. ``call_seq`` is the latest invocation number, or None before any
    invocation. ``committed_seq`` starts at zero and counts successful Next
    commits. ``revision`` is the latest tunables version. ``reason`` describes
    an abort/failure/stop when available. ``progress`` contains display snapshots.
    """

    run_id: str
    workflow: str
    lifecycle: Lifecycle
    call_seq: int | None
    committed_seq: int
    revision: int
    reason: str | None
    progress: tuple[ProgressSnapshot, ...]


@dataclass(frozen=True)
class RunPaths:
    """Two host-allocated, not-yet-existing run directories.

    ``metadata_root`` holds manifest/journal. ``data_root`` holds iterations.
    Engine.start validates distinct roots and refuses existing directories.
    """

    metadata_root: Path
    data_root: Path


@dataclass(frozen=True)
class RunIdentity:
    """Host-supplied identity and provenance, not inferred from directory names.

    ``run_id`` is a nonempty stable run ID. ``devices`` is the detached start
    snapshot. ``hostname`` identifies the host. ``git_commit``/``git_dirty``
    describe workflow source, with None for unknown commit. ``entry_id`` and
    ``entry_label`` identify the workpoint, or None. ``soc_fingerprint`` identifies
    the SoC, or None when absent. The host retains hardware ownership.
    """

    run_id: str
    devices: DeviceSnapshot
    hostname: str
    git_commit: str | None = None
    git_dirty: bool = False
    entry_id: str | None = None
    entry_label: str | None = None
    soc_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("RunIdentity.run_id must not be empty")


@dataclass(frozen=True)
class EncodedRecord:
    """Record projection persisted in the journal.

    ``mode`` is json for structured encoding or repr for degradation. ``value``
    is JSON in json mode and a diagnostic string in repr mode. ``error`` is None
    on success, otherwise the serialization/repr failure description.
    """

    mode: Literal["json", "repr"]
    value: JsonValue
    error: str | None


@dataclass(frozen=True)
class CommittedRecord:
    """One journal-authoritative point returned by Engine.records.

    ``commit_seq`` counts Next commits; ``call_seq`` counts step invocations.
    ``revision`` is the tunables version captured for this invocation.
    ``encoded_record`` contains the point summary. ``iteration_dir`` is relative
    to data_root. ``run_files`` are final paths relative to that iteration, in
    effect order. Temporary files are not listed.
    """

    commit_seq: int
    call_seq: int
    revision: int
    encoded_record: EncodedRecord
    iteration_dir: str
    run_files: tuple[str, ...]


class MissingCapability(RuntimeError):
    """A required host capability is absent; ``capability`` names that seam."""

    def __init__(self, capability: str) -> None:
        self.capability = capability
        super().__init__(f"Missing workflow capability: {capability}")


class RunMismatch(ValueError):
    """Wrong run identity: ``expected`` is requested, ``actual`` is attached."""

    def __init__(self, expected: str, actual: str | None) -> None:
        self.expected = expected
        self.actual = actual
        super().__init__(f"Requested run {expected!r}, attached run is {actual!r}")


class InvalidRunState(RuntimeError):
    """Illegal lifecycle operation; ``action`` and ``lifecycle`` explain rejection."""

    def __init__(self, action: str, lifecycle: Lifecycle) -> None:
        self.action = action
        self.lifecycle = lifecycle
        super().__init__(f"Cannot {action} while workflow is {lifecycle}")


class RevisionConflict(ValueError):
    """Stale edit: ``expected`` is the caller revision, ``actual`` is current."""

    def __init__(self, expected: int, actual: int) -> None:
        self.expected = expected
        self.actual = actual
        super().__init__(f"Expected tunables revision {expected}, actual is {actual}")
