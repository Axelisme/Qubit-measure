"""Explicit whole-document upgrades using caller-owned pure schema steps."""

import argparse
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal
from uuid import uuid4

from filelock import FileLock, Timeout
from pydantic import TypeAdapter
from ruamel.yaml import YAML, YAMLError

from zcu_tools.format_version import YamlMap
from zcu_tools.resources.document_store import DocumentValue, FieldPath
from zcu_tools.resources.parameter_schema import ParameterSchemaPolicy
from zcu_tools.resources.record_table.upgrade import upgrade_record_files
from zcu_tools.utils import atomic_replace


@dataclass(frozen=True)
class ParameterDocument:
    """Detached complete YAML mapping passed between pure upgrade callbacks.

    data includes container headers, schema_version and every authored section.
    Callbacks must preserve unrelated values and must not perform I/O. A runner
    detaches input/output so callbacks cannot mutate the caller's document.
    source_path is optional diagnostic/container-kind context, never permission
    for callbacks to perform I/O. The file runner supplies an absolute path and
    preserves it even when a step returns a fresh record without this field.
    """

    data: YamlMap
    source_path: Path | None = None


@dataclass(frozen=True)
class UpgradeStep:
    """One pure source_version -> source_version+1 structural conversion.

    source_version is a positive integer. transform receives a validated source
    document and must return a complete document with the next schema_version.
    ValueError describes unsupported or incomplete input; no files are touched.
    """

    source_version: int
    transform: Callable[[ParameterDocument], ParameterDocument]


@dataclass(frozen=True)
class UpgradePlan:
    """Caller declarations for a linear chain; no lab version defaults exist.

    policy supplies the current component version and remediation command.
    steps lists retained consecutive UpgradeSteps in ascending source order,
    ending at current_version-1; an empty tuple supports current documents only.
    validate(document, version) must check the complete container and component
    shape for that version, raising ValueError with failed checks. It runs before
    and after each conversion, and before a current document can be skipped.
    classify_legacy(document) returns True only after full unversioned-v1 shape
    classification; False or ValueError rejects corrupt/partial input. Runtime
    readers never call it. Callbacks are pure and receive detached data.
    """

    policy: ParameterSchemaPolicy
    steps: tuple[UpgradeStep, ...]
    validate: Callable[[ParameterDocument, int], None]
    classify_legacy: Callable[[ParameterDocument], bool]

    def __post_init__(self) -> None:
        if not self.steps:
            return
        versions = tuple(step.source_version for step in self.steps)
        first = versions[0]
        if any(
            type(version) is not int or version < 1 for version in versions
        ) or versions != tuple(range(first, self.policy.current_version)):
            raise ValueError(
                "upgrade steps must be consecutive and end at current_version"
            )


def upgrade_document(
    document: ParameterDocument, *, plan: UpgradePlan
) -> ParameterDocument:
    """Return a detached current document, or raise ValueError without I/O.

    Validate every version with caller checks, classify unversioned input only
    here, and apply consecutive pure steps. Reject future/malformed versions,
    missing steps and steps returning anything but the next version. The input
    and plan are never changed, including when a callback fails.
    """
    candidate = deepcopy(document)
    if "schema_version" not in candidate.data:
        if not plan.classify_legacy(deepcopy(candidate)):
            raise ValueError(
                "unversioned document failed complete legacy-v1 classification"
            )
        candidate.data["schema_version"] = 1
    version = candidate.data.get("schema_version")
    if type(version) is not int or not 1 <= version <= plan.policy.current_version:
        raise ValueError(f"unsupported schema_version {version!r}")
    plan.validate(deepcopy(candidate), version)
    while version < plan.policy.current_version:
        step = next(
            (step for step in plan.steps if step.source_version == version), None
        )
        if step is None:
            raise ValueError(f"missing upgrade step v{version} -> v{version + 1}")
        transformed = step.transform(deepcopy(candidate))
        candidate = ParameterDocument(deepcopy(transformed.data), document.source_path)
        next_version = candidate.data.get("schema_version")
        if type(next_version) is not int or next_version != version + 1:
            raise ValueError(
                f"step v{version} must produce schema_version {version + 1}"
            )
        version += 1
        plan.validate(deepcopy(candidate), version)
    return candidate


@dataclass(frozen=True)
class UpgradeSource:
    """One exact YAML path and its optional caller-supplied point label.

    path identifies a setup, point, template or another component document.
    point is display-only context; None denotes no point label. No discovery or
    lab shape is inferred from either field. Relative paths use the current cwd.
    """

    path: Path
    point: str | None = None


@dataclass(frozen=True)
class UpgradeChange:
    """A field difference in decoded YAML, with literal mapping-key segments.

    path identifies a leaf; sequences are replaced as one field. before and
    after distinguish missing fields from explicit YAML null. Values are
    detached working-unit values; no scientific interpretation occurs.
    """

    path: FieldPath
    before: DocumentValue
    after: DocumentValue


@dataclass(frozen=True)
class UpgradeResult:
    """Outcome for one file; failures never prevent processing its siblings.

    source identifies the requested path/point. status is dry-run (changes only),
    applied (replaced after backup), skipped (validated current, unchanged), or
    rejected (no replacement). changes holds field differences on success.
    backup_path is the exclusive byte-for-byte backup, if one was completed,
    including when subsequent replacement failed. error is a failed check or
    I/O diagnostic only for rejected documents; otherwise None.
    """

    source: UpgradeSource
    status: Literal["dry-run", "applied", "skipped", "rejected"]
    changes: tuple[UpgradeChange, ...] = ()
    backup_path: Path | None = None
    error: str | None = None


def upgrade_files(
    sources: Sequence[UpgradeSource], *, plan: UpgradePlan, apply: bool = False
) -> tuple[UpgradeResult, ...]:
    """Validate/upgrade exact files independently, defaulting to read-only dry-run.

    sources is an ordered collection of paths with optional point labels. plan
    owns all component classification/validation. apply takes each file's same
    sidecar lock as DocumentStore, re-reads under lock, creates an exclusive
    sibling backup before atomic replacement, and returns its path. Dry-run
    creates no files or locks. Current documents are fully validated then skipped.
    YAML, ValueError, lock timeout and I/O failures become rejected results;
    unexpected callback errors propagate. No multi-file or power-loss guarantee.
    """
    return tuple(_upgrade_file(source, plan=plan, apply=apply) for source in sources)


def _field_changes(
    before: YamlMap, after: YamlMap, path: FieldPath = ()
) -> tuple[UpgradeChange, ...]:
    changes: list[UpgradeChange] = []
    for key in sorted(before.keys() | after.keys()):
        left, right = before.get(key), after.get(key)
        location = (*path, key)
        if (
            key in before
            and key in after
            and isinstance(left, dict)
            and isinstance(right, dict)
        ):
            changes.extend(_field_changes(left, right, location))
        elif (
            key not in before
            or key not in after
            or type(left) is not type(right)
            or left != right
        ):
            changes.append(
                UpgradeChange(
                    location,
                    DocumentValue(key in before, deepcopy(left)),
                    DocumentValue(key in after, deepcopy(right)),
                )
            )
    return tuple(changes)


def _replace_yaml(path: Path, document: YamlMap) -> None:
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            YAML(typ="rt").dump(document, stream)
        atomic_replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _upgrade_file(
    source: UpgradeSource, *, plan: UpgradePlan, apply: bool
) -> UpgradeResult:
    path = source.path.resolve()
    backup: Path | None = None
    try:
        # Check before FileLock: missing inputs must not create parent directories.
        path.parent.stat()
        lock = FileLock(f"{path}.lock", timeout=5) if apply else nullcontext()
        with lock:
            original = path.read_bytes()
            decoded = YAML(typ="safe").load(original.decode("utf-8"))
            document = ParameterDocument(
                TypeAdapter(YamlMap).validate_python(decoded, strict=True), path
            )
            candidate = upgrade_document(document, plan=plan)
            changes = _field_changes(document.data, candidate.data)
            if not changes:
                return UpgradeResult(source, "skipped")
            if not apply:
                return UpgradeResult(source, "dry-run", changes)
            backup_candidate = path.with_name(f"{path.name}.{uuid4().hex}.bak")
            with backup_candidate.open("xb") as stream:
                stream.write(original)
            backup = backup_candidate
            _replace_yaml(path, candidate.data)
            return UpgradeResult(source, "applied", changes, backup)
    except (OSError, ValueError, YAMLError, Timeout) as error:
        return UpgradeResult(source, "rejected", backup_path=backup, error=str(error))


def main(argv: Sequence[str] | None = None, *, plan: UpgradePlan | None = None) -> int:
    """Run explicit component or record upgrades in process; return 0 or 1.

    argv selects exact paths, --scope components (default) or records, and optional
    --apply; None reads process arguments. Components require a caller-owned plan.
    Records require --device-name, an operator declaration, and do not use plan.
    Default dry-run prints file/field or line classifications without writes.
    Apply reports original-bytes backups. Rejected files do not stop siblings;
    any rejection returns 1. Invalid CLI inputs raise argparse SystemExit(2).
    No lab modules are imported, no versions or device names are inferred.
    """
    parser = argparse.ArgumentParser(
        description="Upgrade component YAML or record JSONL (default: dry-run)"
    )
    parser.add_argument(
        "paths", nargs="+", type=Path, help="exact source paths; shell globs accepted"
    )
    parser.add_argument(
        "--apply", action="store_true", help="back up then replace valid documents"
    )
    parser.add_argument(
        "--scope", choices=("components", "records"), default="components"
    )
    parser.add_argument(
        "--device-name",
        help="operator-declared device for old record values (not inferred)",
    )
    options = parser.parse_args(argv)
    paths = TypeAdapter(list[Path]).validate_python(options.paths)
    apply = TypeAdapter(bool).validate_python(options.apply)
    device_name = TypeAdapter(str | None).validate_python(options.device_name)
    if options.scope == "records":
        if device_name is None:
            parser.error("--scope records requires --device-name")
        try:
            return _record_cli(paths, device_name=device_name, apply=apply)
        except ValueError as error:
            parser.error(str(error))
    if device_name is not None:
        parser.error("--device-name is only valid with --scope records")
    if plan is None:
        parser.error("component scope requires a caller-configured upgrade plan")
    results = upgrade_files(
        tuple(UpgradeSource(path) for path in paths), plan=plan, apply=apply
    )
    for result in results:
        if result.backup_path is not None:
            print(f"{result.source.path}: backup: {result.backup_path}")
        print(f"{result.source.path}: {result.status}")
        for change in result.changes:
            before = repr(change.before.value) if change.before.present else "<missing>"
            after = repr(change.after.value) if change.after.present else "<missing>"
            print(f"  {change.path!r}: {before} -> {after}")
        if result.error is not None:
            print(f"  failed checks: {result.error}")
    return int(any(result.status == "rejected" for result in results))


def _record_cli(paths: Sequence[Path], *, device_name: str, apply: bool) -> int:
    results = upgrade_record_files(paths, device_name=device_name, apply=apply)
    print(f"operator-declared device: {device_name!r} (not inferred)")
    for result in results:
        if result.backup_path is not None:
            print(f"{result.path}: backup: {result.backup_path}")
        print(f"{result.path}: {result.status}")
        for row in result.rows:
            print(f"  line {row.line_number}: {row.classification}")
            if row.error is not None:
                print(f"    failed checks: {row.error}")
        if result.error is not None:
            print(f"  failed checks: {result.error}")
    return int(any(result.status == "rejected" for result in results))
