"""Lab-owned record rows shared by live workflows and offline re-export."""

from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter
from zcu_tools.datafile import JsonObject, load
from zcu_tools.experiment.native_persistence import metadata_from_container
from zcu_tools.experiment.workflows.saved_run import native_path
from zcu_tools.resources.entry.storage import bind_record_table
from zcu_tools.resources.record_table import JsonlRecordTable


class SampleField(BaseModel):
    """One finite JSON field; owner identity fields cannot be replaced."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", strict=True, allow_inf_nan=False
    )
    name: str = Field(min_length=1)
    value: JsonValue

    def model_post_init(self, _context: object) -> None:
        if self.name in {"run_id", "point", "at", "source"}:
            raise ValueError("Sample field cannot replace owner metadata")


@dataclass(frozen=True)
class SampleRow:
    """Workflow run/index key with unique, validated producer-defined fields."""

    key: str
    fields: tuple[SampleField, ...]

    def __post_init__(self) -> None:
        if not self.key or len({f.name for f in self.fields}) != len(self.fields):
            raise ValueError("Sample key must be nonempty and field names unique")


class SampleExportError(RuntimeError):
    """Append failure with this call's written prefix and original OSError."""

    def __init__(self, written_count: int, cause: OSError) -> None:
        self.written_count = written_count
        self.cause = cause
        super().__init__(f"Sample export failed after {written_count} rows: {cause}")


@dataclass(frozen=True)
class SampleExportReport:
    """Append receipt: workflow identity, this-call row count and owner path."""

    run_id: str
    written_count: int
    path: Path


def append_records(
    run_id: str,
    *,
    rows: tuple[SampleRow, ...],
    table: JsonlRecordTable,
    point: str | None,
) -> SampleExportReport:
    """Append prepared rows; retries use the owner's latest-wins keys.

    No rollback occurs. OSError becomes SampleExportError with the successful
    prefix count. point is the historical work-point label, not a current focus.
    """
    if not run_id:
        raise ValueError("Sample export requires a run_id")
    written = 0
    for row in rows:
        try:
            table.append(
                run_id=row.key,
                point=point,
                source=f"workflow:{run_id}",
                at=None,
                **{field.name: field.value for field in row.fields},
            )
        except OSError as cause:
            raise SampleExportError(written, cause) from cause
        written += 1
    return SampleExportReport(run_id, written, table.path)


def append_sample(
    result: Path, component: str, run_id: str, point: str | None, row: SampleRow
) -> SampleExportReport:
    """Append one prepared row to the exact entry Result/component record table."""
    return append_records(
        run_id, rows=(row,), table=bind_record_table(result, component), point=point
    )


def sample_row(key: str, fields: JsonObject) -> SampleRow:
    """Validate a complete finite JSON payload before any append."""
    return SampleRow(
        key, tuple(SampleField(name=k, value=v) for k, v in fields.items())
    )


def native_devices(directory: Path, files: tuple[str, ...]) -> JsonValue:
    """Copy the last saved acquisition's complete device snapshot, or null.

    files follow scientific acquisition order, including rejected/partial traces.
    Explicit references are checked through the native owner, never inferred by
    scanning. Missing/malformed native files raise; no live capture substitutes.
    """
    devices: JsonValue = None
    for file in files:
        metadata = metadata_from_container(load(native_path(directory, file)))
        devices = TypeAdapter(JsonValue).validate_python(metadata.device_snapshot)
    return devices


def lifetime_fields(
    status: str,
    reason: str | None,
    us: float | None,
    err_us: float | None,
    models: JsonValue,
    *,
    detune_mhz: float | None = None,
) -> JsonObject:
    """Project accepted microsecond scalars and retain all model diagnostics.

    Nonaccepted primary scalars are null. Unattempted members have an explicit
    skipped reason. Model records retain their own recorded/discarded/failed status.
    """
    return {
        "status": status,
        "reason": reason or ("not attempted" if status == "skipped" else None),
        "us": us if status == "accepted" else None,
        "err_us": err_us if status == "accepted" else None,
        "detune_mhz": detune_mhz if status == "accepted" else None,
        "models": models,
    }
