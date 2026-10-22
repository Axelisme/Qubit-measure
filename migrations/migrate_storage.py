"""Permanent offline CLI; no legacy runtime owners or hardware control."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from zcu_tools.experiment.metadata import CfgSnapshot
from zcu_tools.resources.entry import ComponentRegistry, component_registry

from migrations.experiment_declaration import MigrationExperiment
from migrations.storage_mapping import build_mapping
from migrations.storage_migration import (
    MigrationInputError,
    MigrationRequest,
    load_run_evidence,
    migrate_storage,
    report_json,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--database-root", required=True, type=Path)
    parser.add_argument(
        "--entry-result-root",
        required=True,
        type=Path,
        help="Root for the new result entry; --result-root selects legacy sources",
    )
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
    parser.add_argument(
        "--repo-root",
        type=Path,
        help="Checkout to exclude from backup (default: tool checkout)",
    )
    parser.add_argument(
        "--backup-parent",
        type=Path,
        help="External parent for <repo-name>-backup (default: checkout parent)",
    )
    parser.add_argument("--assume-flux-unit", choices=("A", "mA", "V", "mV"))
    parser.add_argument("--assume-jpa-flux-unit", choices=("A", "V"))
    parser.add_argument("--sample-value-column")
    parser.add_argument("--sample-source-unit", choices=("A", "mA", "V", "mV"))
    parser.add_argument("--run-evidence", type=Path)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    register_components: Callable[[ComponentRegistry], None] | None = None,
    experiments: Sequence[MigrationExperiment] | None = None,
) -> int:
    """Migrate caller-selected offline roots with explicit kind and evidence.

    argv is a CLI argument sequence, or None for sys.argv. Print the cumulative
    JSON report to stdout, including pending items and forward-minor raw fields.
    register_components is called once per invocation with component_registry;
    None lazily selects this lab's existing register_all. The caller owns registry
    bootstrap and teardown, including duplicate-registration errors. experiments
    supplies fixed schema/validation declarations; None lazily selects all shipped
    declarations. Explicit empty declarations are valid for parameter-only/raw
    inputs, not a fallback for undeclared runs. No global table is changed.
    Declaration validators preserve snapshot/raw input. Default CLI composition
    registers once in its fresh process. Dry-run
    creates no files and never invokes native validation or source removal.
    Return 0 after completed definite work (pending may remain), 2 for
    input/manifest/destination conflicts, or 1 for execution/schema/I/O failures.
    argparse raises SystemExit(2) for invalid CLI arguments.
    Write located errors to stderr; never substitute an empty success report.
    """
    args = _parser().parse_args(argv)
    try:
        if register_components is None:
            from zcu_lab.components import register_all

            register_components = register_all
        if experiments is None:
            from migrations.migration_experiments import MIGRATION_EXPERIMENTS

            experiments = MIGRATION_EXPERIMENTS
        register_components(component_registry)
        declarations = {(item.source_tag, item.cfg_type): item for item in experiments}

        def validate_cfg(tag: str, snapshot: CfgSnapshot) -> None:
            try:
                declaration = declarations[tag, snapshot.cfg_type]
            except KeyError as error:
                raise MigrationInputError(
                    f"Undeclared cfg identity {(tag, snapshot.cfg_type)!r}"
                ) from error
            declaration.validate_cfg(snapshot)

        def validate_native(path: Path, tag: str, cfg_type: str) -> None:
            try:
                declaration = declarations[tag, cfg_type]
            except KeyError as error:
                raise MigrationInputError(
                    f"Undeclared native identity {(tag, cfg_type)!r}"
                ) from error
            declaration.validate_native(path)

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
                identity: item.schemas for identity, item in declarations.items()
            },
            native_tags={
                identity: item.native_tag
                for identity, item in declarations.items()
                if item.source_tag != item.native_tag
            },
        )
        request = MigrationRequest(
            result_root=args.result_root,
            database_root=args.database_root,
            results_root=args.entry_result_root,
            source_chip=args.source_chip,
            source_qubit=args.source_qubit,
            name=args.name,
            part=args.part,
            dry_run=args.dry_run,
            resume=args.resume,
            report_path=args.report,
            repo_root=args.repo_root,
            backup_parent=args.backup_parent,
            assume_flux_unit=args.assume_flux_unit,
            assume_jpa_flux_unit=args.assume_jpa_flux_unit,
            sample_value_column=args.sample_value_column,
            sample_source_unit=args.sample_source_unit,
            run_evidence=load_run_evidence(args.run_evidence)
            if args.run_evidence is not None
            else None,
        )
        report = migrate_storage(
            request,
            mapping=mapping,
            validate_cfg=validate_cfg,
            validate_native=validate_native,
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
