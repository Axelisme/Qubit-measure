"""Explicit offline CSV conversion into entry-bound JSONL sample history."""

import csv
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pydantic import JsonValue
from zcu_tools.datafile import JsonObject, json_values_equal
from zcu_tools.resources.entry import ResultEntry
from zcu_tools.resources.record_table import (
    RECORD_COORDINATE_COLUMNS,
    LegacyDeviceValueUnit,
    migrate_record_table_v2,
    validate_record_table_v2,
)

from migrations.comparison import ExistingItem, MigrationDifference, MigrationInputError

# Evidence: notebook_md/single_qubit.md's record_table.add_sample producer.
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
_RESERVED_FIELDS = frozenset({"run_id", "point", "source", "at", "legacy_fields"})


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
class PreservedCsvField:
    """One unsupported CSV field retained verbatim, not semantically converted.

    source_path is the supplied CSV path. row_number is the 1-based data row,
    excluding the header, not a physical line number. column and value are the
    original header and cell strings, including whitespace and empty strings.
    run_id is the destination's synthetic legacy-<stem>-<row> key, not historical
    evidence. destination_keys is a sequence of literal JSON object keys:
    (column,) for a root extra or ("legacy_fields", column) for a collision with
    producer metadata, a canonical measurement name or legacy_fields itself.
    Dots in a key are literal, never path separators.
    """

    source_path: Path
    row_number: int
    column: str
    value: str
    run_id: str
    destination_keys: tuple[str, ...]


@dataclass(frozen=True)
class MigrationReport:
    """Outcome of a completed offline conversion.

    converted_rows counts newly appended rows; identical synthetic keys skip.
    pending_rows lists omitted data rows in input order, with their reasons.
    output_path is the entry-bound records/components/<component>.jsonl destination; it
    need not exist when nothing was converted. preserved_fields locates raw
    unsupported fields in appended rows, distinct from converted measurements.
    skipped lists verified identical synthetic keys without appending history.
    Source files are never changed.
    """

    converted_rows: int
    pending_rows: tuple[PendingRow, ...]
    output_path: Path
    preserved_fields: tuple[PreservedCsvField, ...] = ()
    skipped: tuple[ExistingItem, ...] = ()


@dataclass(frozen=True)
class _ConvertedRow:
    """Owner-validated coordinates and renamed measurements ready for append."""

    dev_value: float
    dev_unit: str
    fields: dict[str, JsonValue]
    preserved_fields: tuple[PreservedCsvField, ...]


def _synthetic_key(path: Path, number: int) -> str:
    return f"legacy-{path.stem}-{number}"


def _measurement_fields(row: dict[str, str]) -> dict[str, JsonValue]:
    fields: dict[str, JsonValue] = {}
    for column, value in row.items():
        if column not in _MEASUREMENT_NAMES:
            continue
        target = _MEASUREMENT_NAMES[column]
        if value == "":
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
    row: dict[str, str],
    *,
    source_path: Path,
    row_number: int,
    dev_value_column: str | None,
    source_unit: LegacyDeviceValueUnit | None,
) -> _ConvertedRow:
    frame = pd.DataFrame([row]).replace("", None)
    coordinates: set[str]
    if "dev_value" in row or "dev_unit" in row:
        coordinates = set(RECORD_COORDINATE_COLUMNS)
        validate_record_table_v2(frame)
    else:
        if dev_value_column is None or source_unit is None:
            raise ValueError(
                "legacy CSV requires explicit dev_value_column and source_unit"
            )
        coordinates = {*RECORD_COORDINATE_COLUMNS, dev_value_column}
        # The schema owner alone defines scaling and coordinate validation.
        frame = migrate_record_table_v2(
            frame,
            dev_value_column=dev_value_column,
            dev_value_unit=source_unit,
        )
    fields = _measurement_fields(row)
    preserved: list[PreservedCsvField] = []
    legacy_fields: dict[str, JsonValue] = {}
    for column, raw in row.items():
        if column in coordinates or column in _MEASUREMENT_NAMES:
            continue
        if column in _RESERVED_FIELDS or column in _MEASUREMENT_NAMES.values():
            legacy_fields[column] = raw
            destination_keys = ("legacy_fields", column)
        else:
            fields[column] = raw
            destination_keys = (column,)
        preserved.append(
            PreservedCsvField(
                source_path,
                row_number,
                column,
                raw,
                _synthetic_key(source_path, row_number),
                destination_keys,
            )
        )
    if legacy_fields:
        fields["legacy_fields"] = legacy_fields
    for column in RECORD_COORDINATE_COLUMNS:
        if column in ("dev_value", "dev_unit") or column not in frame.columns:
            continue
        value = frame[column].iloc[0]
        fields[column] = None if pd.isna(value) else float(value)
    return _ConvertedRow(
        float(frame["dev_value"].iloc[0]),
        str(frame["dev_unit"].iloc[0]),
        fields,
        tuple(preserved),
    )


@dataclass(frozen=True)
class SampleCsvRow:
    """One validated synthetic sample ready for JSONL append.

    run_id is legacy-<source stem>-<1-based data row>, not a historical run ID.
    dev_value is finite and expressed in dev_unit (A/V/native). fields holds
    renamed measurements with original working units, optional coordinates and
    unsupported original string fields. Colliding extras live in legacy_fields.
    """

    run_id: str
    dev_value: float
    dev_unit: str
    fields: dict[str, JsonValue]


@dataclass(frozen=True)
class SampleCsvPlan:
    """Read-only conversion outcome shared by append and storage publication.

    rows contains accepted rows in source order. pending contains every omitted
    row with source and 1-based data row location. preserved_fields locates
    unsupported fields retained verbatim in accepted rows, in source column
    order. It does not claim semantic conversion. Planning writes nothing.
    """

    rows: tuple[SampleCsvRow, ...]
    pending: tuple[PendingRow, ...]
    preserved_fields: tuple[PreservedCsvField, ...] = ()


def plan_sample_csv(
    sources: Sequence[str | Path],
    *,
    dev_value_column: str | None = None,
    source_unit: LegacyDeviceValueUnit | None = None,
) -> SampleCsvPlan:
    """Read CSVs and validate conversions without acquiring a destination.

    v2 CSV declares dev_value/dev_unit (A/V/native). Legacy input requires the
    literal coordinate header and A/mA/V/mV source_unit; no unit inference.
    Unknown columns retain raw strings, including blanks, at the literal
    destination_keys reported by preserved_fields. Metadata/canonical name
    collisions go under legacy_fields without overwriting converted fields.
    Missing declarations, invalid coordinates/measurements and width mismatches
    omit that row and produce located pending. Empty sources,
    duplicate stems, invalid unit or malformed headers raise ValueError before
    writing anything. CSV syntax and I/O errors propagate. Uses the same
    sample-table validation and renaming contract as convert_sample_csv.
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
    rows: list[SampleCsvRow] = []
    pending: list[PendingRow] = []
    preserved: list[PreservedCsvField] = []
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
                row = dict(zip(header, values, strict=True))
                try:
                    sample = _convert_row(
                        row,
                        source_path=path,
                        row_number=number,
                        dev_value_column=dev_value_column,
                        source_unit=source_unit,
                    )
                except ValueError as exc:
                    pending.append(PendingRow(path, number, str(exc)))
                    continue
                preserved.extend(sample.preserved_fields)
                rows.append(
                    SampleCsvRow(
                        run_id=_synthetic_key(path, number),
                        dev_value=sample.dev_value,
                        dev_unit=sample.dev_unit,
                        fields=sample.fields,
                    )
                )
    return SampleCsvPlan(tuple(rows), tuple(pending), tuple(preserved))


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
    v2 CSVs provide dev_value/dev_unit directly (A/V/native), without scaling.
    For legacy CSVs, explicitly declare dev_value_column and source_unit
    (A/mA/V/mV); neither is inferred from a header. Unmapped legacy coordinate columns require operator review.
    point is copied into every appended row, or left null when not supplied.

    Each row gets legacy-<Path.stem>-<1-based data row>, excluding the header,
    and source manual. Known measurements are renamed to snake_case with unit
    suffixes. Unknown fields retain original names and string values, including
    blanks, without type inference. Collisions with metadata/canonical names
    and legacy_fields itself go in the legacy_fields mapping. preserved_fields
    reports source cells and literal destination keys, not semantic conversion.
    Absent legacy declarations and invalid data become PendingRow entries and
    are not written. Blank known measurements become null. Coordinate
    validation/scaling belongs to the sample owner.

    Empty inputs, unsupported declared units, duplicate stems, unsafe component
    names and a source aliasing the destination raise ValueError before writes.
    CSV headers must be nonempty and unique; malformed CSV/I/O errors propagate.
    Successful rows append in input order. I/O failures may leave earlier rows
    appended; no batch transaction, retry, or concurrent-writer guarantee.
    Repeating equal synthetic keys skips without appending. Different content
    for a present key raises MigrationInputError with typed differences before
    any row is appended. The existing generated at is retained for comparison;
    this is not a general history merge or a change to table owner semantics.
    """
    paths = [Path(source) for source in sources]
    table = entry.records(component)
    output = table.path
    for path in paths:
        if output.exists() and path.exists() and path.samefile(output):
            raise ValueError(f"{path}: source aliases sample output {output}")
    plan = plan_sample_csv(
        paths, dev_value_column=dev_value_column, source_unit=source_unit
    )
    skipped: list[ExistingItem] = []
    differences: list[MigrationDifference] = []
    accepted: list[SampleCsvRow] = []
    for sample in plan.rows:
        existing = table.latest(sample.run_id)
        if existing is None:
            accepted.append(sample)
            continue
        source = next(
            path
            for path in paths
            if sample.run_id.rsplit("-", 1)[0] == f"legacy-{path.stem}"
        )
        expected: JsonObject = {
            "run_id": sample.run_id,
            "point": point,
            "source": "manual",
            "at": existing["at"],
            "dev_value": sample.dev_value,
            "dev_unit": sample.dev_unit,
            **sample.fields,
        }
        if json_values_equal(existing, expected):
            skipped.append(
                ExistingItem(
                    source=source,
                    destination=output,
                    comparison="sample-row",
                    summary=sample.run_id,
                )
            )
        else:
            differences.append(
                MigrationDifference(
                    source=source,
                    destination=output,
                    comparison="sample-row",
                    expected=json.dumps(expected, sort_keys=True),
                    actual=json.dumps(existing, sort_keys=True),
                )
            )
    if differences:
        raise MigrationInputError(
            "synthetic sample differs", differences=tuple(differences)
        )
    for sample in accepted:
        table.append(
            run_id=sample.run_id,
            point=point,
            source="manual",
            at=None,
            dev_value=sample.dev_value,
            dev_unit=sample.dev_unit,
            **sample.fields,
        )
    return MigrationReport(
        len(accepted), plan.pending, output, plan.preserved_fields, tuple(skipped)
    )
