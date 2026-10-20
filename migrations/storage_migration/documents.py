"""Owner-typed document baselines and prepared single-file publications."""

import hashlib
import json
import os
import re
import shutil
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from pydantic import TypeAdapter
from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError
from zcu_tools.datafile import JsonObject, json_values_equal
from zcu_tools.resources.context.library import (
    MODULE_LIBRARY_FORMAT,
    MODULE_LIBRARY_VERSION,
    ModuleLibraryDocument,
)
from zcu_tools.resources.dispersive_file import load_dispersive_file
from zcu_tools.resources.document_store import DocumentStore
from zcu_tools.resources.entry import EditView, PointView, SetupView, source_path
from zcu_tools.resources.entry.schema import (
    PARAMETER_FORMAT,
    PARAMETER_VERSION,
    PointDocument,
    SetupDocument,
)
from zcu_tools.resources.predictor_file import PredictorFile, load_predictor_file
from zcu_tools.resources.t1_curve_file import load_t1_curve_file

from .errors import MigrationInputError
from .models import DocumentBaseline, DocumentKind, ExistingItem, MigrationDifference
from .paths import contained_path, source_identity
from .state import MigrationSession, overlay_known_fields

_JSON = TypeAdapter(JsonObject)
_PREDICTOR = TypeAdapter(PredictorFile)


def document_value(path: Path, kind: DocumentKind) -> JsonObject:
    """Read without mutation through the public owner, retaining unknown raw data.

    kind selects setup/point/library YAML or predictor/dispersive/t1_curve JSON.
    YAML default normalization is provided by DocumentStore, without library
    cleanup. JSON owners define null/default and active-channel semantics.
    Invalid/unreadable documents propagate their schema or I/O errors.
    """
    if kind == "setup":
        known = (
            DocumentStore(
                path,
                SetupDocument,
                format=PARAMETER_FORMAT,
                supported_version=PARAMETER_VERSION,
            )
            .snapshot()
            .model_dump(mode="json")
        )
    elif kind == "point":
        known = (
            DocumentStore(
                path,
                PointDocument,
                format=PARAMETER_FORMAT,
                supported_version=PARAMETER_VERSION,
            )
            .snapshot()
            .model_dump(mode="json")
        )
    elif kind == "library":
        known = (
            DocumentStore(
                path,
                ModuleLibraryDocument,
                format=MODULE_LIBRARY_FORMAT,
                supported_version=MODULE_LIBRARY_VERSION,
            )
            .snapshot()
            .model_dump(mode="json")
        )
    elif kind == "predictor":
        known = _PREDICTOR.dump_python(load_predictor_file(path), mode="json")
    elif kind == "dispersive":
        known = load_dispersive_file(path).model_dump(mode="json", exclude_unset=True)
    else:
        known = load_t1_curve_file(path).model_dump(mode="json", exclude_unset=True)
    raw = (
        _JSON.validate_python(
            YAML(typ="safe").load(path.read_text(encoding="utf-8")), strict=True
        )
        if kind in ("setup", "point", "library")
        else _JSON.validate_json(path.read_bytes(), strict=True)
    )
    return overlay_known_fields(raw, _JSON.validate_python(known, strict=True))


def summary(value: JsonObject | None) -> str:
    """Return a deterministic readable digest of normalized content, or absent."""
    if value is None:
        return "absent"
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return "sha256:" + hashlib.sha256(encoded.encode()).hexdigest()


def difference(
    source: Path, destination: Path, comparison: str, expected: str, actual: str
) -> MigrationInputError:
    """Build one existing typed input conflict; this does not write diagnostics."""
    return MigrationInputError(
        "migration comparison differs",
        differences=(
            MigrationDifference(
                source=source,
                destination=destination,
                comparison=comparison,
                expected=expected,
                actual=actual,
            ),
        ),
    )


def _actual(baseline: DocumentBaseline) -> JsonObject | None:
    path = baseline.destination
    for ancestor in (path, *path.parents):
        if ancestor.is_symlink():
            raise difference(
                baseline.source, path, baseline.kind, summary(baseline.value), "symlink"
            )
    if not path.exists():
        return None
    try:
        return document_value(path, baseline.kind)
    except (ValueError, OSError, YAMLError) as error:
        raise difference(
            baseline.source, path, baseline.kind, summary(baseline.value), str(error)
        ) from error


def validate_document(baseline: DocumentBaseline) -> bool:
    """Require the fixed value or an explicitly prepared previous value.

    Return True only for the expected published value. Missing new prepared
    destinations return False. A third value/schema/path conflict raises typed
    MigrationInputError. No files change.
    """
    actual = _actual(baseline)
    if actual is not None and json_values_equal(actual, baseline.value):
        return True
    if baseline.phase == "prepared" and (
        (actual is None and baseline.previous is None)
        or (
            actual is not None
            and baseline.previous is not None
            and json_values_equal(actual, baseline.previous)
        )
    ):
        if not baseline.temp_path.is_file() or baseline.temp_path.is_symlink():
            raise difference(
                baseline.source,
                baseline.destination,
                baseline.kind,
                "prepared owner artifact",
                "missing staging artifact",
            )
        if not json_values_equal(
            document_value(baseline.temp_path, baseline.kind), baseline.value
        ):
            raise difference(
                baseline.source,
                baseline.temp_path,
                baseline.kind,
                summary(baseline.value),
                "staging artifact changed",
            )
        return False
    raise difference(
        baseline.source,
        baseline.destination,
        baseline.kind,
        summary(baseline.value),
        summary(actual),
    )


def capture_document(
    session: MigrationSession, source: Path, destination: Path, kind: DocumentKind
) -> None:
    """Record trusted initial owner output before any conversion or formal rename.

    Only bootstrap/point staging calls this for newly created owner artifacts.
    It must never be used to adopt existing completed destinations on rerun.
    """
    documents = dict(session.manifest.documents or {})
    documents[str(destination)] = DocumentBaseline(
        source=source,
        destination=destination,
        temp_path=destination.with_name(
            f".{destination.name}.migration-{uuid4().hex}.tmp"
        ),
        kind=kind,
        value=document_value(destination, kind),
        previous=None,
        phase="published",
    )
    session.manifest = replace(session.manifest, documents=documents)


def recover_documents(session: MigrationSession) -> None:
    """Finish verified prepared owner artifacts after all rerun comparisons pass.

    No writes in dry-run. Do not replace equal destinations. A prepared artifact
    only replaces its recorded previous value, never an arbitrary third value.
    """
    for baseline in tuple((session.manifest.documents or {}).values()):
        equal = validate_document(baseline)
        if session.dry_run or baseline.phase == "published":
            continue
        if not equal:
            os.replace(baseline.temp_path, baseline.destination)
        else:
            baseline.temp_path.unlink(missing_ok=True)
        documents = dict(session.manifest.documents or {})
        documents[str(baseline.destination)] = replace(
            baseline, phase="published", previous=None
        )
        session.manifest = replace(session.manifest, documents=documents)
        session.checkpoint()


def publish_document(
    session: MigrationSession,
    source: Path,
    destination: Path,
    kind: DocumentKind,
    build: Callable[[Path], object],
) -> None:
    """Build with an owner in a sibling temp, fix typed intent, then publish once.

    build receives an absent same-directory temp path and must write a complete
    owner artifact. Only a recorded previous value (or absent new destination) is
    replaced. Equal data is not rewritten. Prepared state survives interruptions.
    Callers must have validated all owned state before invoking this operation.
    """
    if session.dry_run:
        return
    old = (session.manifest.documents or {}).get(str(destination))
    if old is not None:
        validate_document(old)
    elif destination.exists() or destination.is_symlink():
        raise difference(
            source,
            destination,
            kind,
            "absent owned destination",
            "unowned existing document",
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.migration-{uuid4().hex}.tmp"
    )
    prepared = False
    try:
        build(temporary)
        value = document_value(temporary, kind)
        previous = document_value(destination, kind) if destination.is_file() else None
        baseline = DocumentBaseline(
            source=source,
            destination=destination,
            temp_path=temporary,
            kind=kind,
            value=value,
            previous=previous,
            phase="prepared",
        )
        documents = dict(session.manifest.documents or {})
        documents[str(destination)] = baseline
        report = session.manifest.report
        skipped = tuple(
            item for item in report.skipped if item.destination != destination
        )
        if previous is not None and json_values_equal(previous, value):
            skipped = (*skipped, existing_document(baseline))
        session.manifest = replace(
            session.manifest,
            documents=documents,
            report=replace(report, skipped=skipped),
        )
        prepared = True
        session.checkpoint()
    except Exception:
        if not prepared:
            temporary.unlink(missing_ok=True)
        raise
    recover_documents(session)


@contextmanager
def edit_parameter(
    session: MigrationSession, source: Path, view: SetupView | PointView
) -> Generator[EditView]:
    """Stage one parameter edit through DocumentStore/EditView before publication.

    view is the currently bound destination; source locates the legacy input.
    Owner serializers/defaults/unknown raw are applied to the staged copy first.
    Refresh the original view only after verified publication. Failures before
    preparation discard staging; prepared failures retain the owner artifact.
    """
    path = source_path(view)
    kind: DocumentKind = "setup" if isinstance(view, SetupView) else "point"
    temporary = path.with_name(f".{path.name}.migration-edit-{uuid4().hex}.tmp")
    shutil.copyfile(path, temporary)
    try:
        if kind == "setup":
            with DocumentStore(
                temporary,
                SetupDocument,
                format=PARAMETER_FORMAT,
                supported_version=PARAMETER_VERSION,
            ).edit() as draft:
                yield EditView(draft)
        else:
            with DocumentStore(
                temporary,
                PointDocument,
                format=PARAMETER_FORMAT,
                supported_version=PARAMETER_VERSION,
            ).edit() as point_draft:
                yield EditView(point_draft)
        publish_document(
            session,
            source,
            path,
            kind,
            lambda target: shutil.copyfile(temporary, target),
        )
        view.refresh()
    finally:
        temporary.unlink(missing_ok=True)


def validate_document_baselines(
    session: MigrationSession,
    *,
    labels: Sequence[str] = (),
    report_existing: bool = True,
) -> None:
    """Validate complete fixed document evidence before opening entry handles.

    labels optionally includes currently visible point labels. Pending owned
    stages/final renames are validated by the point publication owner instead.
    Missing baselines are typed conflicts, never populated from current outputs.
    Successful comparisons update only the in-memory skipped report.
    """
    manifest = session.manifest
    documents = manifest.documents
    if documents is None:
        raise difference(
            manifest.report.source.result_path,
            session.path,
            "document-baseline",
            "complete document baseline",
            "缺比較依據",
        )
    root = manifest.report.destination.result_path
    required = {root / "setup.yaml", root / "module_cfg.yaml"}
    required.update(item.destination for item in manifest.module_files)
    required.update(
        row.new_file
        for row in manifest.report.key_mappings
        if row.new_file is not None
        and row.action != "pending"
        and row.new_file.parent
        in (root / "predictors", root / "dispersive", root / "t1_curve")
    )
    pending_labels = {
        label
        for point in manifest.points
        if not point.published
        for label in (point.label, point.temporary_label)
    }
    published = {point.label for point in manifest.points if point.published}
    for label in set(labels) | published:
        if label not in pending_labels:
            required.update(
                root / "points" / label / name
                for name in ("point.yaml", "module_cfg.yaml")
            )
    missing = required - {item.destination for item in documents.values()}
    if missing:
        raise difference(
            manifest.report.source.result_path,
            session.path,
            "document-baseline",
            "complete document baseline",
            "缺比較依據: " + str(sorted(missing)),
        )
    skipped = list(manifest.report.skipped)
    for key, baseline in documents.items():
        _validate_document_paths(session, key, baseline)
        if validate_document(baseline) and report_existing:
            item = existing_document(baseline)
            skipped = [old for old in skipped if old.destination != item.destination]
            skipped.append(item)
    session.manifest = replace(
        manifest, report=replace(manifest.report, skipped=tuple(skipped))
    )


def _validate_document_paths(
    session: MigrationSession, key: str, baseline: DocumentBaseline
) -> None:
    root = session.manifest.report.destination.result_path
    if (
        key != str(baseline.destination)
        or contained_path(baseline.destination, root) != baseline.destination
    ):
        raise MigrationInputError("manifest: invalid document destination")
    if (
        baseline.temp_path.parent != baseline.destination.parent
        or re.fullmatch(
            re.escape(f".{baseline.destination.name}.migration-")
            + r"[0-9a-f]{32}\.tmp",
            baseline.temp_path.name,
        )
        is None
    ):
        raise MigrationInputError("manifest: invalid document staging path")
    relative = baseline.destination.relative_to(root)
    parts = relative.parts
    if baseline.kind == "setup":
        valid = parts == ("setup.yaml",)
    elif baseline.kind == "point":
        valid = len(parts) == 3 and parts[0] == "points" and parts[2] == "point.yaml"
    elif baseline.kind == "library":
        valid = parts == ("module_cfg.yaml",) or (
            len(parts) == 3 and parts[0] == "points" and parts[2] == "module_cfg.yaml"
        )
    else:
        directory = "predictors" if baseline.kind == "predictor" else baseline.kind
        valid = len(parts) == 2 and parts[0] == directory and parts[1].endswith(".json")
    if not valid:
        raise MigrationInputError("manifest: document kind/path disagrees")
    source_identity(baseline.source, session.manifest)


def existing_document(baseline: DocumentBaseline) -> ExistingItem:
    """Describe a verified fixed document for the public skipped report."""
    return ExistingItem(
        source=baseline.source,
        destination=baseline.destination,
        comparison=baseline.kind,
        summary=summary(baseline.value),
    )
