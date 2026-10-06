"""Offline entry conversion through explicit mappings and owned recovery."""

from collections.abc import Callable
from pathlib import Path

from zcu_tools.datafile import CfgSnapshot

from .bootstrap import prepare_session, publish_report, validate_mapping
from .evidence import select_run_evidence
from .models import (
    MigrationMapping,
    MigrationReport,
    MigrationRequest,
)
from .parameters import convert_parameters
from .preservation import report_unrecognized_files
from .runs import convert_data


def migrate_storage(
    request: MigrationRequest,
    *,
    mapping: MigrationMapping,
    validate_cfg: Callable[[str, CfgSnapshot], None],
    validate_native: Callable[[Path, str, str], None],
) -> MigrationReport:
    """Convert an explicit legacy entry without modifying its old result tree.

    request names roots, safe source segments, destination, selected parts and
    optional acquisition evidence. mapping supplies complete registered seeds,
    explicit key rules, disk schemas and native-tag renames; register kinds in entry's shared registry
    before calling. Values/stderr keep working units; expressions are not evaluated.
    module_cfg conversion remains pending for the cfg owner, not generic YAML.
    validate_cfg receives the known historical source tag and CfgSnapshot.
    It must check cfg_type, schema_version and values with the concrete cfg owner,
    return None without mutation, and raise ValueError for unconvertible cfg.
    Such cfg stays located pending before run assignment/native planning; other
    failures propagate. Parameters-only skips it; dry_run may invoke this pure check.
    validate_native receives the published native path, historical source tag and cfg_type;
    it must load against its declared experiment spec without modifying the file.
    It runs synchronously before the corresponding Labber source can be deleted.

    First execution refuses either existing destination and independent report.
    resume only accepts this tool's matching manifest, immutable entry identity,
    exact mapping revision and owned file hashes. It reuses run assignments and
    completes planned/prepared/published moves without deleting unverified sources.
    Completed setup/point documents remain editable and are validated, not restored.
    dry_run reads/validates sources and returns a report without any writes, moves
    or native validation callback. Missing cfg identity/acquisition evidence or an undeclared
    (tag, cfg_type) schema is pending; never fall back to a tag-only declaration.

    Return the cumulative report of attempted parts. Normal runs persist manifest
    checkpoints and the fixed report location. Input/manifest/hash conflicts raise
    MigrationInputError, destination collisions raise FileExistsError, schema and
    I/O failures propagate. Publication and callback failures retain retry state.
    No hardware, concurrent-writer or power-loss transaction guarantee is provided.
    """
    validate_mapping(mapping)
    session, entry = prepare_session(request, mapping)
    evidence = select_run_evidence(request, session)
    try:
        if request.part in ("parameters", "all"):
            convert_parameters(session, entry, mapping)
        if request.part in ("data", "all"):
            convert_data(
                session, entry, mapping, evidence, validate_cfg, validate_native
            )
        report_unrecognized_files(session)
        report = session.manifest.report
        if not request.dry_run:
            publish_report(session)
        return report
    except Exception:
        if not request.dry_run:
            publish_report(session)
        raise
