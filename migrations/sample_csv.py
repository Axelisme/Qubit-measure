"""Explicit offline CSV conversion into entry-bound JSONL sample history."""

import csv
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pydantic import JsonValue
from zcu_tools.resources.entry import ResultEntry
from zcu_tools.resources.sample_table import (
    SAMPLE_COORDINATE_COLUMNS,
    DeviceValueUnit,
    LegacyDeviceValueUnit,
    migrate_sample_table_v2,
    validate_sample_table_v2,
)

# Evidence: notebook_md/single_qubit.md's sample_table.add_sample producer.
# Units are already microseconds / MHz; only names change here.
_MEASUREMENT_NAMES = {
    "Freq (MHz)": "freq_mhz",
    "T1 (us)": "t1_us",
    "T1err (us)": "t1_err_us",
    "T2r (us)": "t2r_us",
    "T2r err (us)": "t2r_err_us",
    "T2e (us)": "t2e_us",
    "T2e err (us)": "t2e_err_us",
    "Tcomment": "tcomment",
    "date": "date",
}
_TEXT_COLUMNS = frozenset({"Tcomment", "date"})


@dataclass(frozen=True)
class PendingRow:
    """One data row omitted because its conversion needs operator judgment.

    source_path is the supplied CSV path. row_number is the 1-based data row,
    excluding the header (not a physical line number for multiline CSV cells).
    reason names the ambiguous fields or invalid values; no row is written.
    """

    source_path: Path
    row_number: int
    reason: str


@dataclass(frozen=True)
class MigrationReport:
    """Outcome of a completed offline conversion.

    converted_rows counts appended rows, including repeated synthetic keys.
    pending_rows lists omitted data rows in input order, with their reasons.
    output_path is the entry-bound samples/<component>.jsonl destination; it
    need not exist when nothing was converted. Source files are never changed.
    """

    converted_rows: int
    pending_rows: tuple[PendingRow, ...]
    output_path: Path


@dataclass(frozen=True)
class _ConvertedRow:
    dev_value: float
    dev_unit: DeviceValueUnit
    fields: dict[str, JsonValue]


def _synthetic_key(path: Path, number: int) -> str:
    return f"legacy-{path.stem}-{number}"


def _measurement_fields(row: dict[str, str | None]) -> dict[str, JsonValue]:
    fields: dict[str, JsonValue] = {}
    for column, value in row.items():
        if column not in _MEASUREMENT_NAMES:
            continue
        target = _MEASUREMENT_NAMES[column]
        if value is None or value == "":
            fields[target] = None
        elif column in _TEXT_COLUMNS:
            fields[target] = value
        else:
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"{column}: measurement must be finite")
            fields[target] = number
    return fields


def _convert_row(
    row: dict[str, str | None],
    *,
    dev_value_column: str | None,
    source_unit: LegacyDeviceValueUnit | None,
) -> _ConvertedRow:
    frame = pd.DataFrame([row]).replace("", None)
    coordinates: set[str]
    if "dev_value" in row or "dev_unit" in row:
        coordinates = set(SAMPLE_COORDINATE_COLUMNS)
        validate_sample_table_v2(frame)
    else:
        if dev_value_column is None or source_unit is None:
            raise ValueError(
                "legacy CSV requires explicit dev_value_column and source_unit"
            )
        coordinates = {dev_value_column}
        # The schema owner alone defines scaling and coordinate validation.
        frame = migrate_sample_table_v2(
            frame,
            dev_value_column=dev_value_column,
            dev_value_unit=source_unit,
        )
    unknown = sorted(set(row) - coordinates - _MEASUREMENT_NAMES.keys())
    if unknown:
        raise ValueError(f"unmapped measurement column(s): {unknown}")
    fields = _measurement_fields(row)
    for column in SAMPLE_COORDINATE_COLUMNS:
        if column in ("dev_value", "dev_unit") or column not in frame.columns:
            continue
        value = frame[column].iloc[0]
        fields[column] = None if pd.isna(value) else float(value)
    unit = frame["dev_unit"].iloc[0]
    if unit not in ("A", "V"):
        raise ValueError(f"dev_unit: unsupported migrated unit {unit!r}")
    return _ConvertedRow(float(frame["dev_value"].iloc[0]), unit, fields)


def convert_sample_csv(
    sources: Sequence[str | Path],
    *,
    entry: ResultEntry,
    component: str,
    dev_value_column: str | None = None,
    source_unit: LegacyDeviceValueUnit | None = None,
    point: str | None = None,
) -> MigrationReport:
    """Append synthetic samples from one or more operator-selected CSV files.

    entry is an already-open ResultEntry; component selects its sample table.
    v2 CSVs provide dev_value/dev_unit directly. For legacy CSVs, explicitly
    declare dev_value_column and source_unit (A/mA/V/mV); neither is inferred
    from a header. Unmapped legacy coordinate columns require operator review.
    point is copied into every appended row, or left null when not supplied.

    Each row gets legacy-<Path.stem>-<1-based data row>, excluding the header,
    and source manual. Known measurements are renamed to snake_case with unit
    suffixes. Unknown measurements, absent legacy declarations and invalid
    data become PendingRow entries and are not written. Blank measurements
    become null. Coordinate validation/scaling belongs to the sample owner.

    Empty inputs, unsupported declared units, duplicate stems, unsafe component
    names and a source aliasing the destination raise ValueError before writes.
    CSV headers must be nonempty and unique; malformed CSV/I/O errors propagate.
    Successful rows append in input order. I/O failures may leave earlier rows
    appended; no batch transaction, retry, or concurrent-writer guarantee.
    Repeating a conversion appends history with the same synthetic keys.
    """
    paths = [Path(source) for source in sources]
    if not paths:
        raise ValueError("at least one source CSV is required")
    if source_unit is not None and source_unit not in ("A", "mA", "V", "mV"):
        raise ValueError(f"unsupported source_unit {source_unit!r}")
    stems: set[str] = set()
    for path in paths:
        if path.stem in stems:
            raise ValueError(f"duplicate input stem {path.stem!r}")
        stems.add(path.stem)
    table = entry.samples(component)
    output = entry.result_path / "samples" / f"{component}.jsonl"
    for path in paths:
        if output.exists() and path.exists() and path.samefile(output):
            raise ValueError(f"{path}: source aliases sample output {output}")
    converted = 0
    pending: list[PendingRow] = []
    for path in paths:
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream, strict=True)
            header = next(reader, [])
            if (
                not header
                or any(not name for name in header)
                or len(set(header)) != len(header)
            ):
                raise ValueError(f"{path}: CSV headers must be nonempty and unique")
            for number, values in enumerate(reader, start=1):
                if len(values) != len(header):
                    pending.append(
                        PendingRow(path, number, "CSV row does not match header width")
                    )
                    continue
                row: dict[str, str | None] = dict(zip(header, values, strict=True))
                try:
                    sample = _convert_row(
                        row,
                        dev_value_column=dev_value_column,
                        source_unit=source_unit,
                    )
                except ValueError as exc:
                    # This boundary isolates invalid source rows for operator review.
                    pending.append(PendingRow(path, number, str(exc)))
                    continue
                table.append(
                    run_id=_synthetic_key(path, number),
                    point=point,
                    source="manual",
                    at=None,
                    dev_value=sample.dev_value,
                    dev_unit=sample.dev_unit,
                    **sample.fields,
                )
                converted += 1
    return MigrationReport(converted, tuple(pending), output)
