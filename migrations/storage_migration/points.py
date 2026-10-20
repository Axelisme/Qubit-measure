"""Manifest-owned staging for complete points with stable creation times."""

import re
import shutil
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from zcu_tools.resources.entry import PointView, ResultEntry

from .backup import inventory_paths
from .documents import difference, document_value, validate_document
from .models import DocumentBaseline, MigrationManifest, PointPublication
from .paths import contained_path, validate_segment
from .state import MigrationSession


def _paths(manifest: MigrationManifest, item: PointPublication) -> tuple[Path, Path]:
    validate_segment(item.label, "point label")
    if re.fullmatch(r"\.migration-point-[0-9a-f]{32}", item.temporary_label) is None:
        raise difference(
            manifest.report.source.result_path,
            manifest.report.destination.result_path,
            "point-ownership",
            "tool staging label",
            item.temporary_label,
        )
    validate_segment(item.temporary_label, "temporary point label")
    root = manifest.report.destination.result_path / "points"
    staged, final = root / item.temporary_label, root / item.label
    for path in (staged, final):
        if path.is_symlink() or contained_path(path, root) != path:
            raise difference(
                manifest.report.source.result_path,
                path,
                "point-path",
                "contained directory",
                "symlink or escape",
            )
    return staged, final


def validate_points(manifest: MigrationManifest) -> None:
    """Read-only validation of pending point ownership and all prepared contents.

    Reject collisions, extra files/directories, symlinks and changed typed/raw
    content. An unprepared stage belongs only to the exact manifest label and may
    be discarded later; its regular-file inventory is still checked for safety.
    Published points are checked by the main document/publication validator.
    """
    labels: set[str] = set()
    temporary: set[str] = set()
    for item in manifest.points:
        staged, final = _paths(manifest, item)
        if (
            item.label in labels
            or item.temporary_label in temporary
            or item.label == item.temporary_label
        ):
            raise difference(
                manifest.report.source.result_path,
                final,
                "point-ownership",
                "unique labels",
                "duplicate",
            )
        labels.add(item.label)
        temporary.add(item.temporary_label)
        if item.published:
            if staged.exists():
                raise difference(
                    manifest.report.source.result_path,
                    staged,
                    "point-ownership",
                    "absent published stage",
                    "reappeared",
                )
            continue
        if item.documents is None:
            if final.exists():
                raise difference(
                    manifest.report.source.result_path,
                    final,
                    "point-ownership",
                    "absent unprepared destination",
                    "existing",
                )
            if staged.exists():
                inventory_paths({"point": staged})
            continue
        _validate_prepared_point(manifest, item, staged, final)


def _validate_prepared_point(
    manifest: MigrationManifest, item: PointPublication, staged: Path, final: Path
) -> None:
    if item.documents is None or set(item.documents) != {
        "point.yaml",
        "module_cfg.yaml",
    }:
        raise difference(
            manifest.report.source.result_path,
            final,
            "point-baseline",
            "complete point documents",
            "缺比較依據",
        )
    for name, baseline in item.documents.items():
        if (
            baseline.destination != final / name
            or baseline.kind != ("point" if name == "point.yaml" else "library")
            or baseline.phase != "published"
        ):
            raise difference(
                manifest.report.source.result_path,
                final,
                "point-baseline",
                "fixed formal document path/kind",
                "invalid baseline",
            )
    if staged.exists() and final.exists():
        raise difference(
            manifest.report.source.result_path,
            final,
            "point-ownership",
            "one prepared directory",
            "both stage and final exist",
        )
    root = staged if staged.exists() else final
    current = inventory_paths({"point": root})
    if (
        set(current.hashes) != set(item.hashes)
        or current.directories != item.directories
    ):
        raise difference(
            manifest.report.source.result_path,
            root,
            "point-inventory",
            str(sorted(item.hashes)),
            current.summary,
        )
    for key, expected in item.hashes.items():
        relative = key.removeprefix("point/")
        document = item.documents.get(relative)
        if document is not None:
            validate_document(replace(document, destination=root / relative))
        elif current.hashes[key] != expected:
            raise difference(
                manifest.report.source.result_path,
                root / relative,
                "sha256",
                expected,
                current.hashes[key],
            )


def recover_points(session: MigrationSession) -> None:
    """Finish verified prepared renames and install fixed document baselines.

    Call only after all session comparisons pass. Dry-run writes nothing. A
    renamed point is reopened by the caller, never returned with staging binding.
    """
    validate_points(session.manifest)
    if session.dry_run:
        return
    for item in tuple(session.manifest.points):
        if item.published or item.documents is None:
            continue
        staged, final = _paths(session.manifest, item)
        if staged.exists():
            staged.rename(final)
        documents = dict(session.manifest.documents or {})
        for baseline in item.documents.values():
            documents[str(baseline.destination)] = baseline
        session.manifest = replace(
            session.manifest,
            documents=documents,
            points=tuple(
                replace(old, published=True) if old.label == item.label else old
                for old in session.manifest.points
            ),
        )
        session.checkpoint()


def create_point(
    session: MigrationSession, entry: ResultEntry, label: str, source: Path
) -> PointView:
    """Create/recover an owned point via new_point, prepared baseline, then rename.

    Fix the staging label in the manifest before owner creation. Only its exact
    unprepared staging directory may be removed. Prepared contents/time never
    regenerate. Arbitrary formal points or stages are not adopted. Reopen through
    use_point(label) after publication so all subsequent edits bind the final path.
    """
    item = next((old for old in session.manifest.points if old.label == label), None)
    if item is None:
        item = PointPublication(
            label=label, temporary_label=f".migration-point-{uuid4().hex}"
        )
        staged, final = _paths(session.manifest, item)
        if staged.exists() or final.exists():
            raise difference(
                source,
                final,
                "point-ownership",
                "absent new point",
                "unowned existing directory",
            )
        session.manifest = replace(
            session.manifest, points=(*session.manifest.points, item)
        )
        session.checkpoint()
    validate_points(session.manifest)
    if not item.published and item.documents is None:
        staged, final = _paths(session.manifest, item)
        if staged.exists():
            shutil.rmtree(staged)
        entry.new_point(item.temporary_label)
        documents: dict[str, DocumentBaseline] = {}
        for name, kind in (("point.yaml", "point"), ("module_cfg.yaml", "library")):
            path = final / name
            documents[name] = DocumentBaseline(
                source=source,
                destination=path,
                temp_path=path.with_name(f".{name}.migration-{uuid4().hex}.tmp"),
                kind="point" if kind == "point" else "library",
                value=document_value(
                    staged / name, "point" if kind == "point" else "library"
                ),
                previous=None,
                phase="published",
            )
        inventory = inventory_paths({"point": staged})
        prepared = replace(
            item,
            documents=documents,
            hashes=inventory.hashes,
            directories=inventory.directories,
        )
        session.manifest = replace(
            session.manifest,
            points=tuple(
                prepared if old.label == label else old
                for old in session.manifest.points
            ),
        )
        session.checkpoint()
    recover_points(session)
    return entry.use_point(label)
