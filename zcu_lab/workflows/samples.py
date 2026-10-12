"""Coherence accepted scalars and actual last-Completed device coordinates."""

from dataclasses import dataclass
from math import isfinite
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, JsonValue
from zcu_tools.datafile import load_run_data
from zcu_tools.resources.sample_table import JsonlSampleTable

from zcu_lab.v2.twotone.freq.core import FreqExp
from zcu_lab.v2.twotone.rabi.len_rabi.core import LenRabiExp
from zcu_lab.v2.twotone.ro_optimize.freq_gain.core import FreqGainExp

from .coherence import CoherencePlan, CoherencePoint
from .export import load_lifetime_trace, load_trace
from .export_inputs import SavedPoint, native_path


class SampleField(BaseModel):
    """Producer-defined finite JSON field, never owner metadata."""

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
    """One sample: workflow run/index key and uniquely named validated fields."""

    key: str
    fields: tuple[SampleField, ...]

    def __post_init__(self) -> None:
        if not self.key or len({f.name for f in self.fields}) != len(self.fields):
            raise ValueError("Sample key must be nonempty and field names unique")


class SampleExportError(RuntimeError):
    """I/O failure retains written_count prefix and original cause/__cause__."""

    def __init__(self, written_count: int, cause: OSError) -> None:
        self.written_count = written_count
        self.cause = cause
        super().__init__(f"Sample export failed after {written_count} rows: {cause}")


@dataclass(frozen=True)
class SampleExportReport:
    """Successful append receipt: workflow identity, this-call count and owner path."""

    run_id: str
    written_count: int
    path: Path


def append_samples(
    run_id: str,
    *,
    rows: tuple[SampleRow, ...],
    table: JsonlSampleTable,
    point: str | None,
) -> SampleExportReport:
    """Append an entirely prepared batch, retaining prefix/count/cause on OSError.

    table is the explicitly bound result/component owner. point is the historical
    plan point. Retry appends every row; owner reads latest complete row per key.
    All rows must already be validated. No ledger/accept event or rollback occurs.
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


def _positive(value: float | None, name: str) -> float:
    if value is None or not isfinite(value) or value <= 0:
        raise ValueError(f"Accepted {name} must be finite and positive")
    return value


def _last_completed(saved: SavedPoint[CoherencePoint]) -> str:
    point = saved.point
    references: list[str] = []
    # Validate every explicit reference, not just the snapshot chosen for the row.
    for file in point.frequency.files:
        load_trace(saved.directory, file, FreqExp().persistence_spec())
        references.append(file)
    if point.rabi.file is not None:
        load_trace(saved.directory, point.rabi.file, LenRabiExp().persistence_spec())
        references.append(point.rabi.file)
    if point.readout.file is not None:
        load_trace(
            saved.directory, point.readout.file, FreqGainExp().persistence_spec()
        )
        references.append(point.readout.file)
    for item in point.measurements:
        if item.file is not None:
            load_lifetime_trace(saved.directory, item.file, item.kind)
            references.append(item.file)
    if not references:
        raise ValueError(
            f"{saved.record_path}: Accepted sample requires a Completed acquisition"
        )
    return references[-1]


def _coordinate(
    saved: SavedPoint[CoherencePoint], device: str
) -> tuple[SampleField, SampleField]:
    path = native_path(saved.directory, _last_completed(saved))
    snapshot = load_run_data(path).metadata.device_snapshot
    if snapshot is None or device not in snapshot:
        raise ValueError(f"{path}: Native snapshot lacks flux device {device}")
    value, unit = snapshot[device].get("value"), snapshot[device].get("value_unit")
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
    ):
        raise ValueError(f"{path}: Device value must be finite real")
    if not isinstance(unit, str) or not unit:
        raise ValueError(f"{path}: Device value_unit must be nonempty")
    return SampleField(name="dev_value", value=value), SampleField(
        name="dev_unit", value=unit
    )


def prepare_samples(
    run_id: str, plan: CoherencePlan, points: tuple[SavedPoint[CoherencePoint], ...]
) -> tuple[SampleRow, ...]:
    """Prepare at most one row per point: accepted frequency OR any lifetime.

    Only accepted values appear in MHz/us fields (t1, t2r, t2e). Calibration
    success is not a row gate. Coordinates preserve last Completed native device
    value/unit in frequency→Rabi→readout→T1→Ramsey→Echo order, including rejected
    fits. No plan/live/alias substitution or fitting occurs. Validate the whole
    batch before returning; malformed scalars/references/snapshots raise with
    their record/native source and nothing is appended.
    """
    rows: list[SampleRow] = []
    for saved in points:
        point = saved.point
        fields: list[SampleField] = []
        try:
            if point.frequency.status == "accepted":
                fields.append(
                    SampleField(
                        name="freq_mhz",
                        value=_positive(point.frequency.frequency_mhz, "frequency"),
                    )
                )
            for item in point.measurements:
                if item.status != "accepted":
                    continue
                quantity = {"t1": "t1", "t2ramsey": "t2r", "t2echo": "t2e"}[item.kind]
                value = _positive(item.lifetime_us, quantity)
                if (
                    item.error_us is None
                    or not isfinite(item.error_us)
                    or item.error_us < 0
                ):
                    raise ValueError(
                        f"Accepted {quantity} error must be finite/nonnegative"
                    )
                fields.extend(
                    (
                        SampleField(name=quantity + "_us", value=value),
                        SampleField(name=quantity + "_err_us", value=item.error_us),
                    )
                )
            if fields:
                fields.extend(_coordinate(saved, plan.flux_dev))
                rows.append(SampleRow(f"{run_id}/{point.index}", tuple(fields)))
        except ValueError as error:
            raise ValueError(f"{saved.record_path}: {error}") from error
    return tuple(rows)
