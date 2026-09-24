"""Explicit manual migration shell for legacy SampleTable CSVs to the v2 contract.

Usage::

    .venv/bin/python script/migrate_sample_table_v2.py SOURCE DESTINATION \\
        --dev-value-column "calibrated mA" \\
        --dev-value-unit A \\
        [--flux-column Flux] \\
        [--flux-int VALUE --flux-period VALUE --frame-unit A]

Dry-run by default: only an explicit ``--write`` creates the destination, and a
write is refused when the destination already exists (no-clobber) or equals the
source. One scalar ``--dev-value-unit`` applies to the whole source column; the
tool never infers a per-row unit. Source CSV files are never modified; any
failure preserves the source and leaves no partial destination.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from zcu_tools.meta_tool import (
    DEV_UNIT_COLUMN,
    DEV_VALUE_COLUMN,
    FLUX_COLUMN,
    FLUX_INT_COLUMN,
    FLUX_PERIOD_COLUMN,
    LegacySampleFluxFrame,
    migrate_sample_table_v2,
)
from zcu_tools.meta_tool.sample_schema import _is_non_real_coordinate

LEGACY_UNITS = ("A", "mA", "V", "mV")
# Per-row source-value problems are reported by the dry-run (convertible counts);
# all structural problems fail in both modes.
_PER_ROW_VALUE_REASONS = frozenset(
    {"non_numeric_coordinate", "null_required_value", "non_finite_value"}
)


def _conversion_description(unit: str) -> str:
    base = "A" if unit in ("A", "mA") else "V"
    factor = "×1" if unit in ("A", "V") else "÷1000"
    return f"{unit} → {base} ({factor})"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Migrate a legacy user SampleTable CSV to the flat SampleTable v2 "
            "coordinate contract. Dry-run by default; --write creates a distinct "
            "no-clobber destination."
        ),
    )
    parser.add_argument("source", help="legacy CSV to migrate (never modified)")
    parser.add_argument("destination", help="v2 output CSV path")
    parser.add_argument(
        "--dev-value-column",
        required=True,
        help="source column holding device values in the declared unit",
    )
    parser.add_argument(
        "--dev-value-unit",
        required=True,
        choices=LEGACY_UNITS,
        help="single caller-declared unit of the whole source column",
    )
    parser.add_argument(
        "--flux-column",
        help="trusted source flux column to map to the v2 'flux' column",
    )
    parser.add_argument(
        "--flux-int",
        type=float,
        help="legacy frame flux=0 anchor in --frame-unit (with --flux-period)",
    )
    parser.add_argument(
        "--flux-period",
        type=float,
        help="legacy frame one-Phi0 device-value span in --frame-unit",
    )
    parser.add_argument(
        "--frame-unit",
        choices=LEGACY_UNITS,
        help="legacy unit of --flux-int / --flux-period values",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="create the destination (temporary file + atomic no-clobber)",
    )
    return parser


def _require_frame_group(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    parts = [args.flux_int, args.flux_period, args.frame_unit]
    if any(part is not None for part in parts) and not all(
        part is not None for part in parts
    ):
        parser.error(
            "--flux-int, --flux-period and --frame-unit must be provided together"
        )


def _coordinate_plan_columns(
    args: argparse.Namespace, flux_frame: LegacySampleFluxFrame | None
) -> list[str]:
    columns = [DEV_VALUE_COLUMN, DEV_UNIT_COLUMN]
    if args.flux_column is not None:
        columns.insert(0, FLUX_COLUMN)
    if flux_frame is not None:
        columns.extend([FLUX_INT_COLUMN, FLUX_PERIOD_COLUMN])
    return columns


def _convertible_row_count(samples: pd.DataFrame, column: str) -> int:
    if column not in samples.columns:
        return 0
    safe_values = [
        np.nan if _is_non_real_coordinate(value) else value
        for value in samples[column].array
    ]
    values = pd.to_numeric(pd.Series(safe_values), errors="coerce")
    return int(np.isfinite(values.to_numpy(dtype=np.float64, na_value=np.nan)).sum())


def _write_csv_atomic(destination: Path, migrated: pd.DataFrame) -> None:
    """Publish a complete sibling temp file atomically without clobbering."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
        )
        os.close(fd)
        temp_path = Path(temp_name)
        migrated.to_csv(temp_path, index=False)
        # The sibling temp file is on the destination filesystem. A hard link
        # publishes its complete contents atomically and fails with EEXIST if a
        # destination appears after the caller's early no-clobber check.
        os.link(temp_path, destination)
        temp_path.unlink()
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _require_frame_group(parser, args)
    source = Path(args.source)
    destination = Path(args.destination)
    if source.resolve() == destination.resolve():
        print(
            f"error: destination must differ from source: {destination}",
            file=sys.stderr,
        )
        return 1
    if not source.is_file():
        print(f"error: source CSV not found: {source}", file=sys.stderr)
        return 1
    try:
        samples = pd.read_csv(source)
    except pd.errors.EmptyDataError:
        samples = pd.DataFrame()
    flux_frame = None
    if args.flux_int is not None:
        assert args.flux_period is not None and args.frame_unit is not None
        try:
            flux_frame = LegacySampleFluxFrame(
                dev_unit=args.frame_unit,
                flux_int=args.flux_int,
                flux_period=args.flux_period,
            )
        except ValueError as exc:
            print(f"error: migration failed: {exc}", file=sys.stderr)
            return 1
    migration_error: ValueError | None = None
    migrated: pd.DataFrame | None = None
    try:
        migrated = migrate_sample_table_v2(
            samples,
            dev_value_column=args.dev_value_column,
            dev_value_unit=args.dev_value_unit,
            flux_column=args.flux_column,
            flux_frame=flux_frame,
        )
    except ValueError as exc:
        migration_error = exc
    if migration_error is not None:
        reason = getattr(migration_error, "reason", None)
        if reason not in _PER_ROW_VALUE_REASONS or args.write:
            print(f"error: migration failed: {migration_error}", file=sys.stderr)
            return 1
    total = len(samples)
    convertible = _convertible_row_count(samples, args.dev_value_column)
    print("SampleTable v2 migration plan")
    print(f"  source:       {source}")
    print(f"  destination:  {destination}")
    print(f"  source column: {args.dev_value_column!r}")
    print(f"  declared unit: {_conversion_description(args.dev_value_unit)}")
    print(f"  rows:         total={total}, convertible={convertible}")
    print("  v2 coordinate: " + ", ".join(_coordinate_plan_columns(args, flux_frame)))
    if args.flux_column is None:
        print("  flux:         not mapped (--flux-column not given)")
    else:
        print(f"  flux:         mapped from {args.flux_column!r}")
    if flux_frame is None:
        print(
            "  frame:        not provided "
            "(--flux-int/--flux-period/--frame-unit not given)"
        )
    else:
        print(
            f"  frame:        flux_int={flux_frame.flux_int} "
            f"flux_period={flux_frame.flux_period} unit={flux_frame.dev_unit}"
        )
    if migration_error is not None:
        print(
            f"  warning:      write would fail: {migration_error}",
            file=sys.stderr,
        )
    elif convertible < total:
        print(
            f"  warning:      {total - convertible} row(s) have no finite "
            "numeric source value and would fail v2 validation",
            file=sys.stderr,
        )
    if not args.write:
        if destination.exists():
            print(
                "  warning:      destination exists; --write would be refused",
                file=sys.stderr,
            )
        print(
            "dry-run: no files written; re-run with --write to create the destination."
        )
        return 0
    if destination.exists():
        print(
            f"error: destination already exists (no-clobber): {destination}",
            file=sys.stderr,
        )
        return 1
    assert migrated is not None
    try:
        _write_csv_atomic(destination, migrated)
    except FileExistsError:
        print(
            f"error: destination already exists (no-clobber): {destination}",
            file=sys.stderr,
        )
        return 1
    except OSError as exc:
        print(f"error: failed to write destination: {exc}", file=sys.stderr)
        return 1
    print(f"Migrated {len(migrated)} row(s) to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
