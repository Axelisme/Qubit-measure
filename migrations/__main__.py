"""Thin command-line entry for explicit legacy sample CSV conversion."""

import argparse
from pathlib import Path
from typing import Literal

from pydantic import TypeAdapter
from zcu_tools.resources.entry import ResultEntry

from .sample_csv import MigrationReport, convert_sample_csv


def main() -> int:
    """Parse migration arguments, append rows and print a typed JSON report.

    Existing result/database entries are required. Pending rows produce exit 1;
    a complete conversion produces exit 0. Configuration and I/O failures
    propagate; this CLI does not create entries or modify source CSVs.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--database-root", required=True, type=Path)
    parser.add_argument("--entry", required=True)
    parser.add_argument("--component", required=True)
    parser.add_argument("--dev-value-column")
    parser.add_argument("--source-unit", choices=("A", "mA", "V", "mV"))
    parser.add_argument("--point")
    args = parser.parse_args()
    unit = TypeAdapter(Literal["A", "mA", "V", "mV", None]).validate_python(
        args.source_unit
    )
    entry = ResultEntry.open(
        args.entry, result_root=args.result_root, database_root=args.database_root
    )
    report = convert_sample_csv(
        args.sources,
        entry=entry,
        component=args.component,
        dev_value_column=args.dev_value_column,
        source_unit=unit,
        point=args.point,
    )
    print(TypeAdapter(MigrationReport).dump_json(report, indent=2).decode("utf-8"))
    return 1 if report.pending_rows else 0


if __name__ == "__main__":
    raise SystemExit(main())
