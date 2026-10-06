"""Typed inputs and durable state for explicit, offline storage migration."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal
from uuid import UUID

from zcu_tools.datafile import (
    CfgSnapshot,
    JsonObject,
    ParameterSnapshot,
    SoftwareProvenance,
    VariableSchema,
)

type MigrationPart = Literal["parameters", "data", "all"]
type MappingAction = Literal["value", "stderr", "module", "remove", "pending"]


@dataclass(frozen=True, kw_only=True)
class KeyRule:
    """One explicit legacy key rule, without unit conversion or evaluation.

    old_key is the literal legacy key. target_path is its dotted component,
    provenance or module destination, required for value/stderr/module actions.
    action selects value, stderr, deferred module conversion, removal or pending.
    reason explains the decision for the human-facing report. wrap_key, when
    non-None, wraps a value in one named field for a complete container write;
    only value actions support it. requires_keys lists explicit, non-expression
    keys required in the same source JSON; absent evidence leaves this pending.
    """

    old_key: str
    target_path: str | None
    action: MappingAction
    reason: str
    wrap_key: str | None = None
    requires_keys: tuple[str, ...] = ()


@dataclass(frozen=True, kw_only=True)
class ModuleRule:
    """One deferred flat-library name with an explicit destination and reference.

    old_name is the literal source library key. target_path is the future library
    key (component.partition.name), not a published cfg. reference_path is a
    component module-slot path or None. Rule order chooses the first present
    candidate per slot; other candidates retain their own destinations. reason
    explains this declaration without validating or converting module cfg.
    """

    old_name: str
    target_path: str
    reference_path: str | None
    reason: str


@dataclass(frozen=True, kw_only=True)
class MigrationMapping:
    """Caller-owned declarations, not a runtime plugin registry.

    mapping_version is a major.minor revision, matched exactly on resume.
    components maps component names to complete seeds in working units.
    rules is an ordered tuple of explicit legacy key decisions.
    roles maps role names to component names.
    data_schemas maps (experiment tag, cfg_type) pairs to disk-unit schemas;
    both strings are exact historical identities, with no tag-only fallback.
    module_rules declares deferred flat module names in reference priority order.
    The composition root registers the seeds' kinds with entry before use.
    """

    mapping_version: str
    components: Mapping[str, JsonObject]
    rules: tuple[KeyRule, ...]
    roles: Mapping[str, str]
    data_schemas: Mapping[tuple[str, str], tuple[VariableSchema, ...]]
    module_rules: tuple[ModuleRule, ...] = ()


@dataclass(frozen=True, kw_only=True)
class LegacySnapshotEvidence:
    """Historical entry state with no invented legacy entry UUID.

    entry_name is the name at acquisition. point is the historical context label
    or explicit None; description is historical text or None. roles maps role
    names to component names. params maps dotted paths to historical values and
    self-contained audit sources, in working units. The converter assigns the
    new entry's UUID only when constructing a native RunSnapshot.
    """

    entry_name: str
    point: str | None
    description: str | None
    roles: Mapping[str, str]
    params: Mapping[str, ParameterSnapshot]


@dataclass(frozen=True, kw_only=True)
class LegacyRunEvidence:
    """Explicit supplemental acquisition evidence for one legacy Labber file.

    source is relative to Database/source_chip/source_qubit, never absolute or
    containing '..'. source_hash is its lowercase SHA256. experiment is a
    declared tag, not a filename inference. cfg is the complete historical cfg,
    or None when cfg/cfg_type evidence is missing; the converter keeps it pending
    and the document retains the complete unresolved JSON in raw.
    started_at is a UTC ISO acquisition time; finished_at is UTC ISO or None.
    completion is the historical complete/partial/stopped state.
    snapshot holds historical entry/point/parameter evidence. provenance holds
    historical software evidence; its None values mean missing evidence.
    Source containment, hash and existing metadata agreement are checked by the
    converter against the actual source tree, not guessed by the JSON loader.
    """

    source: Path
    source_hash: str
    experiment: str
    cfg: CfgSnapshot | None
    started_at: str
    finished_at: str | None
    completion: Literal["complete", "partial", "stopped"]
    snapshot: LegacySnapshotEvidence
    provenance: SoftwareProvenance


@dataclass(frozen=True, kw_only=True)
class MigrationRunEvidenceDocument:
    """Validated JSON evidence plus its lossless original JSON tree.

    format is zcu.migration-run-evidence. format_version is supported major 1
    with any nonnegative minor. entries is the known typed evidence projection.
    raw is the complete detached document, including unknown nested fields;
    manifest persistence must retain raw rather than reconstructing it from
    entries. No file metadata, host state or current parameters are captured.
    """

    format: str
    format_version: str
    entries: tuple[LegacyRunEvidence, ...]
    raw: JsonObject


@dataclass(frozen=True, kw_only=True)
class MigrationRequest:
    """Explicit roots, source identity and selected offline migration steps.

    result_root and database_root contain the old source_chip/source_qubit tree.
    results_root contains the new name entry; database_root also holds its heavy
    data entry. source_chip, source_qubit and name are safe single path segments,
    not physical identifiers to parse. part selects parameters, data or both.
    dry_run reads only and publishes nothing. resume requires this tool's prior
    manifest, never an arbitrary existing entry. report_path is an optional
    independent report file; None selects records/migration-report.json.
    run_evidence is the complete caller-loaded document or None. The converter
    resolves paths and rejects source/destination overlaps and identity changes.
    """

    result_root: Path
    database_root: Path
    results_root: Path
    source_chip: str
    source_qubit: str
    name: str
    part: MigrationPart
    dry_run: bool = False
    resume: bool = False
    report_path: Path | None = None
    run_evidence: MigrationRunEvidenceDocument | None = None


@dataclass(frozen=True, kw_only=True)
class MigrationSource:
    """Resolved absolute old result and Database entry paths, respectively."""

    result_path: Path
    database_path: Path


@dataclass(frozen=True, kw_only=True)
class MigrationDestination:
    """Resolved absolute new results and Database entry paths, respectively."""

    result_path: Path
    database_path: Path


@dataclass(frozen=True, kw_only=True)
class KeyMappingItem:
    """Report location of one key decision.

    old_file is an absolute source file; old_key is its literal legacy key.
    new_file and new_path are the absolute target file and dotted target path,
    or None when removed/pending. action selects the explicit rule operation;
    reason explains conversion, deferral or removal without evaluating values.
    """

    old_file: Path
    old_key: str
    new_file: Path | None
    new_path: str | None
    action: MappingAction
    reason: str


@dataclass(frozen=True, kw_only=True)
class FileMigrationItem:
    """Completed file action with absolute source/destination and SHA256 hashes.

    status is converted for native, moved only after Labber source removal, or
    preserved for a byte-identical copy. Pending and failed actions are separate.
    """

    source: Path
    destination: Path
    source_hash: str
    destination_hash: str
    status: Literal["converted", "moved", "preserved"]


@dataclass(frozen=True, kw_only=True)
class PendingItem:
    """Unresolved source (absolute Path), field location, reason and next action."""

    source: Path
    location: str
    reason: str
    suggested_action: str


@dataclass(frozen=True, kw_only=True)
class FailureItem:
    """Failed absolute source, operation name and diagnostic, without traceback."""

    source: Path
    operation: str
    error: str


@dataclass(frozen=True, kw_only=True)
class MigrationReport:
    """Cumulative report of attempted parts, not a claim that pending is empty.

    source/destination identify both resolved roots. entry_id is the new UUID.
    part is the union of attempted parts. key_mappings holds explicit decisions.
    converted_files, moved_labber_files and preserved_files contain completed
    actions of the corresponding status. pending/failures hold unresolved items.
    format/format_version identify zcu.migration-report 1.x. raw retains a loaded
    document's unknown fields; an empty raw denotes a newly constructed report.
    """

    source: MigrationSource
    destination: MigrationDestination
    entry_id: UUID
    part: MigrationPart
    key_mappings: tuple[KeyMappingItem, ...] = ()
    converted_files: tuple[FileMigrationItem, ...] = ()
    moved_labber_files: tuple[FileMigrationItem, ...] = ()
    preserved_files: tuple[FileMigrationItem, ...] = ()
    pending: tuple[PendingItem, ...] = ()
    failures: tuple[FailureItem, ...] = ()
    format: str = "zcu.migration-report"
    format_version: str = "1.0"
    raw: JsonObject = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class MigrationFileState:
    """Manifest-owned file publication state; paths are absolute and resolved.

    operation is copy, move_labber or native. source_hash is baseline SHA256.
    destination_hash is None while planned, then SHA256 of prepared temp bytes.
    temp_path names only this action's owned same-directory temporary file.
    phase is planned/prepared/published, or source_removed for move_labber only.
    native_validated is False/True for native, None for all other operations.
    A published native is not removable-source evidence until validation passes.
    """

    operation: Literal["copy", "move_labber", "native"]
    source: Path
    destination: Path
    temp_path: Path
    source_hash: str
    destination_hash: str | None
    phase: Literal["planned", "prepared", "published", "source_removed"]
    native_validated: bool | None


@dataclass(frozen=True, kw_only=True)
class RunAssignment:
    """Persisted new run identity, assigned before any native publication.

    source is absolute; source_hash is baseline SHA256; run_id is opaque.
    assigned_at is timezone-aware assignment time, never acquisition evidence.
    """

    source: Path
    source_hash: str
    run_id: str
    assigned_at: datetime


@dataclass(frozen=True, kw_only=True)
class MigrationIdentity:
    """Immutable resolved roots, safe source segments, new name and entry UUID.

    part and evidence are deliberately absent: they may extend a pending run.
    """

    result_root: Path
    database_root: Path
    results_root: Path
    source_chip: str
    source_qubit: str
    name: str
    entry_id: UUID


@dataclass(frozen=True, kw_only=True)
class MigrationManifest:
    """Durable state for this tool's entry, not a multi-file transaction.

    identity is immutable; mapping_version must match exactly on resume.
    report_path is the immutable resolved report location. source_hashes maps
    absolute source path strings to baseline SHA256. evidence stores the full
    original JSON document or None, including unknown fields. files records
    owned publications; runs records fixed run assignments. parameters_complete
    and data_complete record finished steps (pending may remain). report is the
    cumulative report to rebuild after interrupted report publication.
    report_published records ownership of the first report publication; False
    means an existing path must match that exact initial report before recovery.
    format/format_version identify zcu.storage-migration 1.x. raw retains loaded
    unknown fields; an empty raw denotes a new manifest.
    """

    identity: MigrationIdentity
    mapping_version: str
    report_path: Path
    source_hashes: Mapping[str, str]
    evidence: JsonObject | None
    files: tuple[MigrationFileState, ...]
    runs: tuple[RunAssignment, ...]
    parameters_complete: bool
    data_complete: bool
    report: MigrationReport
    report_published: bool = False
    format: str = "zcu.storage-migration"
    format_version: str = "1.0"
    raw: JsonObject = field(default_factory=dict)
