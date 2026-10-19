"""Typed inputs and durable state for explicit, offline storage migration."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Literal
from uuid import UUID

from pydantic import JsonValue
from zcu_tools.datafile import (
    CfgSnapshot,
    JsonObject,
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
    """One flat-library key with an explicit family, destination and slot reference.

    old_name is the literal source library key. target_path is the proposed concrete
    key (component.partition.name). Publication requires successful Spec conversion.
    reference_path is a component module-slot path or None. Rule order chooses the
    first present candidate per slot; other candidates retain their own destinations. reason
    explains this declaration. kind selects the source modules or waveforms
    partition; waveform rules must not supply a module-slot reference_path.
    """

    old_name: str
    target_path: str
    reference_path: str | None
    reason: str
    kind: Literal["module", "waveform"] = "module"


@dataclass(frozen=True, kw_only=True)
class MigrationMapping:
    """Caller-owned declarations, not a runtime plugin registry.

    mapping_version is a major.minor revision, matched exactly on resume.
    components maps component names to complete seeds in working units.
    rules is an ordered tuple of explicit legacy key decisions.
    roles maps role names to component names.
    data_schemas maps (experiment tag, cfg_type) pairs to disk-unit schemas;
    both strings are exact historical identities, with no tag-only fallback.
    native_tags maps declared historical pairs to canonical native experiment tags;
    absent pairs keep their historical tag. It never changes source evidence,
    Labber tags or callback lookup identities, and is not a runtime alias.
    module_rules declares deferred flat module names in reference priority order.
    The composition root registers the seeds' kinds with entry before use.
    """

    mapping_version: str
    components: Mapping[str, JsonObject]
    rules: tuple[KeyRule, ...]
    roles: Mapping[str, str]
    data_schemas: Mapping[tuple[str, str], tuple[VariableSchema, ...]]
    module_rules: tuple[ModuleRule, ...] = ()
    native_tags: Mapping[tuple[str, str], str] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class LegacyCloneOrigin:
    """Historical source entry UUID and optional work-point label."""

    entry_id: UUID
    point: str | None


@dataclass(frozen=True, kw_only=True)
class LegacyParameterSource:
    """Audit origin of a parameter value in working units.

    source is manual or a local ledger event ID. kind and run_id are optional
    historical identifiers. at is a UTC ISO timestamp. stderr is an optional
    uncertainty in the parameter's working units. cloned_from identifies the
    original entry/work point for a clone, or None for a non-cloned value.
    Persistence does not resolve or invent ledger evidence.
    """

    source: str
    kind: str | None
    run_id: str | None
    at: str
    stderr: float | None
    cloned_from: LegacyCloneOrigin | None


@dataclass(frozen=True, kw_only=True)
class LegacyParameterSnapshot:
    """JSON value, optional working-unit label and its audit source."""

    value: JsonValue
    unit: str | None
    source: LegacyParameterSource


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
    params: Mapping[str, LegacyParameterSnapshot]


@dataclass(frozen=True, kw_only=True)
class LegacyRunEvidence:
    """Explicit supplemental acquisition evidence for one legacy Labber file.

    source is relative to Database/source_chip/source_qubit, never absolute or
    containing '..'. source_hash is its lowercase SHA256. experiment is a
    declared tag, not a filename inference. cfg is the complete historical cfg,
    or None when cfg/cfg_type evidence is missing; the converter keeps it pending
    and the document retains the complete unresolved JSON in raw.
    started_at is a UTC ISO acquisition time, or None when unproven. Missing
    time fields default to None; completion defaults to unknown. With no start,
    native metadata uses None/unknown/None, retaining all original evidence here.
    Otherwise completion preserves complete/partial/stopped/unknown evidence.
    run_id is an explicit trusted historical native-format identity, or None when
    missing. Only missing IDs receive a new migration-assigned identity.
    snapshot holds historical entry/point/parameter evidence. provenance holds
    historical software evidence; its None values mean missing evidence.
    Source containment, hash and existing metadata agreement are checked by the
    converter against the actual source tree, not guessed by the JSON loader.
    """

    source: Path
    source_hash: str
    experiment: str
    cfg: CfgSnapshot | None
    started_at: str | None = None
    finished_at: str | None = None
    completion: Literal["complete", "partial", "stopped", "unknown"] = "unknown"
    snapshot: LegacySnapshotEvidence
    provenance: SoftwareProvenance
    run_id: str | None = None


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
    independent report file; None selects the shared backup's dated report.
    Completion also publishes a dated report in the entry records directory.
    run_evidence is the complete caller-loaded document or None. The converter
    resolves paths and rejects source/destination overlaps and identity changes.
    repo_root is the checkout to exclude from backup, or None for this tool's
    checkout. backup_parent is an external directory, or None for repo_root.parent;
    the tool accumulates chip/qubit mirrors in its <repo-name>-backup child.
    Either existing source mirror is refused before migration writes.
    assume_flux_unit supplies A/mA/V/mV only when flux unit keys are absent;
    explicit invalid/conflicting evidence stays pending. assume_jpa_flux_unit
    independently supplies A/V only for absent JPA unit evidence; it never uses
    assume_flux_unit. sample_value_column and
    sample_source_unit declare the legacy CSV coordinate column and A/mA/V/mV
    unit independently of flux assumptions. All four are fixed on resume.
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
    repo_root: Path | None = None
    backup_parent: Path | None = None
    assume_flux_unit: Literal["A", "mA", "V", "mV"] | None = None
    assume_jpa_flux_unit: Literal["A", "V"] | None = None
    sample_value_column: str | None = None
    sample_source_unit: Literal["A", "mA", "V", "mV"] | None = None


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
    new_file is the absolute target file, or None when no destination handle is
    acquired (including library dry-run rows) or the key is removed/pending.
    new_path is the proposed dotted target path, or None without a declared target.
    A library dry-run may have new_path with new_file=None; this alone is not pending.
    action selects the explicit rule operation. reason explains conversion, deferral
    or removal without evaluating values.
    library_kind is module/waveform for a library entry, or None for meta keys.
    Same-name library keys in different source partitions remain distinct.
    """

    old_file: Path
    old_key: str
    new_file: Path | None
    new_path: str | None
    action: MappingAction
    reason: str
    library_kind: Literal["module", "waveform"] | None = None


@dataclass(frozen=True, kw_only=True)
class FileMigrationItem:
    """Completed file action with absolute source/destination and SHA256 hashes.

    status is converted for native or typed module cfg, moved after Labber source removal, or
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
class MigrationDeclarations:
    """Operator declarations, immutable across parts/resume of one migration.

    assume_flux_unit is A/mA/V/mV or None; it fills absent flux unit evidence,
    never explicit null/invalid/conflicting values. assume_jpa_flux_unit is A/V
    or None and fills only absent JPA evidence, independently of global flux.
    sample_value_column is the
    literal legacy CSV coordinate header or None. sample_source_unit is its
    explicit A/mA/V/mV unit or None, never inherited from assume_flux_unit.
    """

    assume_flux_unit: Literal["A", "mA", "V", "mV"] | None = None
    assume_jpa_flux_unit: Literal["A", "V"] | None = None
    sample_value_column: str | None = None
    sample_source_unit: Literal["A", "mA", "V", "mV"] | None = None


@dataclass(frozen=True, kw_only=True)
class MigrationReport:
    """Cumulative report of attempted parts, not a claim that pending is empty.

    source/destination identify both resolved roots. entry_id is the new UUID.
    part is the union of attempted parts. key_mappings holds explicit decisions.
    converted_files, moved_labber_files and preserved_files contain completed
    actions of the corresponding status. pending/failures hold unresolved items.
    format/format_version identify zcu.migration-report 1.x. raw retains a loaded
    document's unknown fields; an empty raw denotes a newly constructed report.
    backup_path is the external verified backup, or its planned path in dry_run;
    None is reserved for reports loaded without the backup extension.
    declarations records the operator inputs shared with the typed manifest.
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
    backup_path: Path | None = None
    declarations: MigrationDeclarations = field(default_factory=MigrationDeclarations)
    format: str = "zcu.migration-report"
    format_version: str = "1.0"
    raw: JsonObject = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class MigrationFileState:
    """Manifest-owned file publication state; paths are absolute and resolved.

    operation is copy, move_labber, native or samples (CSV to JSONL).
    source_hash is baseline SHA256; samples/native output need not equal it.
    destination_hash is None while planned, then SHA256 of prepared temp bytes.
    temp_path names only this action's owned same-directory temporary file.
    phase is planned/prepared/published, or source_removed for move_labber only.
    native_validated is False/True for native, None for all other operations.
    A published native is not removable-source evidence until validation passes.
    """

    operation: Literal["copy", "move_labber", "native", "samples"]
    source: Path
    destination: Path
    temp_path: Path
    source_hash: str
    destination_hash: str | None
    phase: Literal["planned", "prepared", "published", "source_removed"]
    native_validated: bool | None


@dataclass(frozen=True, kw_only=True)
class RunAssignment:
    """Persisted run identity, fixed before any native publication.

    source is absolute; source_hash is baseline SHA256; run_id is the trusted
    historical ID or a migration-assigned opaque ID. assigned_at is the
    timezone-aware migration assignment time, never acquisition evidence.
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
class ModuleFilePublication:
    """A completed single-file library commit, not a publication phase.

    source and destination are absolute legacy/canonical module_cfg paths.
    source_hash and destination_hash are SHA256 hex digests after successful
    library edit. Resume verifies them without rewriting the document.
    This record supplies no recovery for a write not yet recorded in manifest.
    """

    source: Path
    destination: Path
    source_hash: str
    destination_hash: str


@dataclass(frozen=True, kw_only=True)
class MigrationBackup:
    """Verified external backup identity fixed before any migration writes.

    path is the resolved shared <repo-name>-backup directory. inventory_hashes
    maps result and Database to SHA256 of their respective records/*.sha256 lists,
    binding complete relative file hashes and directory lists for this chip/qubit.
    source_hashes maps absolute original files to their backed-up SHA256, so
    publications reject changed/new sources before any individual move deletes.
    """

    path: Path
    inventory_hashes: Mapping[str, str]
    source_hashes: Mapping[str, str]


@dataclass(frozen=True, kw_only=True)
class MigrationManifest:
    """Durable state for this tool's entry, not a multi-file transaction.

    identity is immutable; mapping_version must match exactly on resume.
    report_path is the immutable resolved report location. report_date is the
    first execution's UTC ISO date, also naming the completion report in entry
    records. completion_report_published records ownership of that entry copy.
    source_hashes maps
    absolute source path strings to baseline SHA256. evidence stores the full
    original JSON document or None, including unknown fields. files records
    owned publications; runs records fixed run assignments. parameters_complete
    and data_complete record finished steps (pending may remain). report is the
    cumulative report to rebuild after interrupted report publication.
    report_published records ownership of the first report publication; False
    means an existing path must match that exact initial report before recovery.
    format/format_version identify zcu.storage-migration 1.x. raw retains loaded
    unknown fields; an empty raw denotes a new manifest. module_files records
    completed setup/point library commits; pending entries may remain in report.
    Module files and manifest are not a cross-file atomic transaction.
    backup binds the verified external inventory; None is invalid for execution.
    cleanup_phase is pending before full completion, cleaning after verified
    deletion authorization, or complete after both old entry trees are removed.
    A cleaning checkpoint permits resuming deletion of missing backed-up files.
    """

    identity: MigrationIdentity
    mapping_version: str
    report_path: Path
    report_date: date
    source_hashes: Mapping[str, str]
    evidence: JsonObject | None
    files: tuple[MigrationFileState, ...]
    runs: tuple[RunAssignment, ...]
    parameters_complete: bool
    data_complete: bool
    report: MigrationReport
    report_published: bool = False
    completion_report_published: bool = False
    format: str = "zcu.storage-migration"
    format_version: str = "1.0"
    raw: JsonObject = field(default_factory=dict)
    module_files: tuple[ModuleFilePublication, ...] = ()
    backup: MigrationBackup | None = None
    cleanup_phase: Literal["pending", "cleaning", "complete"] = "pending"
