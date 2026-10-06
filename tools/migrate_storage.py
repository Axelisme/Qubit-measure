"""Permanent offline CLI; no legacy runtime owners or hardware control."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from zcu_tools.datafile import CfgSnapshot
from zcu_tools.resources.entry import component_registry
from zcu_tools.resources.storage_migration import (
    MigrationInputError,
    MigrationRequest,
    load_run_evidence,
    migrate_storage,
    report_json,
)

from zcu_lab.components import register_all
from zcu_lab.migration_experiments import MIGRATION_EXPERIMENTS
from zcu_lab.storage_migration import build_mapping

# Fixed declarations, keyed by both proven identities. This is not registration
# or discovery; data schemas and readers come from the same declaration objects.
_DECLARATIONS = {
    (item.source_tag, item.cfg_type): item for item in MIGRATION_EXPERIMENTS
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--database-root", required=True, type=Path)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--source-chip", required=True)
    parser.add_argument("--source-qubit", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--part", required=True, choices=("parameters", "data", "all"))
    parser.add_argument(
        "--qubit-kind", required=True, choices=("qubit/fluxonium", "qubit/transmon")
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--run-evidence", type=Path)
    return parser


def _validate_cfg(tag: str, snapshot: CfgSnapshot) -> None:
    declaration = _DECLARATIONS[tag, snapshot.cfg_type]
    declaration.validate_cfg(snapshot)


def _validate_native(path: Path, tag: str, cfg_type: str) -> None:
    try:
        declaration = _DECLARATIONS[tag, cfg_type]
    except KeyError as exc:
        raise MigrationInputError(
            f"Undeclared native identity {(tag, cfg_type)!r}"
        ) from exc
    declaration.validate_native(path)


def main(argv: Sequence[str] | None = None) -> int:
    """Migrate caller-selected offline roots with explicit kind and evidence.

    argv is a CLI argument sequence, or None for sys.argv. Print the cumulative
    JSON report to stdout, including pending items and forward-minor raw fields.
    Register this lab's entry definitions once in the fresh CLI process. Dry-run
    creates no files and never invokes native validation or source removal.
    Return 0 after completed definite work (pending may remain), 2 for argparse,
    input/manifest/destination conflicts, or 1 for execution/schema/I/O failures.
    Write located errors to stderr; never substitute an empty success report.
    """
    args = _parser().parse_args(argv)
    try:
        register_all(component_registry)
        # argparse owns choice validation; the branch preserves the literal type
        # for the concrete profile rather than asserting an untyped Namespace.
        kind = (
            "qubit/fluxonium"
            if args.qubit_kind == "qubit/fluxonium"
            else "qubit/transmon"
        )
        mapping = build_mapping(
            qubit_kind=kind,
            data_schemas={
                identity: item.schemas for identity, item in _DECLARATIONS.items()
            },
            native_tags={
                identity: item.native_tag
                for identity, item in _DECLARATIONS.items()
                if item.source_tag != item.native_tag
            },
        )
        request = MigrationRequest(
            result_root=args.result_root,
            database_root=args.database_root,
            results_root=args.results_root,
            source_chip=args.source_chip,
            source_qubit=args.source_qubit,
            name=args.name,
            part=args.part,
            dry_run=args.dry_run,
            resume=args.resume,
            report_path=args.report,
            run_evidence=load_run_evidence(args.run_evidence)
            if args.run_evidence is not None
            else None,
        )
        report = migrate_storage(
            request,
            mapping=mapping,
            validate_cfg=_validate_cfg,
            validate_native=_validate_native,
        )
        print(
            json.dumps(
                report_json(report), ensure_ascii=False, indent=2, allow_nan=False
            )
        )
        return 0
    except (MigrationInputError, FileExistsError) as exc:
        print(f"migration conflict: {exc}", file=sys.stderr)
        return 2
    except Exception:
        # This standalone process is the execution isolation/exit-code boundary.
        logging.exception("migration failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
