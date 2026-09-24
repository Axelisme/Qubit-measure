"""Export an autofluxdep run artifact to a notebook-style SampleTable CSV.

Usage:

    .venv/bin/python script/export_autofluxdep_sample_table.py \
        results/autofluxdep_runs/20260706-120000_flux-sweep-abcd1234

The input may be either an ``autofluxdep_runs/<run_slug>`` directory or its
``manifest.json`` file. Passing the paired heavy data root under
``Database/.../autofluxdep_runs/<run_slug>`` is also supported when its metadata
manifest can be found nearby. The artifact must carry an authoritative A/V
flux unit snapshotted at run creation (`manifest.json` `workflow.flux.unit`);
artifacts without one (bare sweeps, missing/unsupported devices, legacy
empty-unit artifacts) fail before any file is written. Existing output CSV
files are appended by default, and missing sample fields are left blank.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from zcu_tools.gui.app.autofluxdep.services.sample_table_export import (
    export_sample_table_from_artifact,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export autofluxdep run data to SampleTable CSV format.",
    )
    parser.add_argument(
        "run",
        help=(
            "autofluxdep metadata run directory, manifest.json path, or paired "
            "Database data run directory"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        help=(
            "output CSV path; defaults to <data_root>/exports/sample/samples.csv "
            "and appends to any existing file"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace the output CSV instead of appending rows",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="suppress the success message",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = export_sample_table_from_artifact(
        args.run,
        filepath=args.output,
        append=not args.overwrite,
    )
    if not args.quiet:
        print(f"Exported {result.row_count} sample row(s) to {result.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
