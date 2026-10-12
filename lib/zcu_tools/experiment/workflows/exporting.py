"""Qt-free inputs and receipts for explicit terminal fluxdep exports."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from zcu_tools.datafile import CfgSnapshot, ExperimentPayload, RunSnapshot

from .models import CommittedRecord, RunIdentity, RunPaths


@dataclass(frozen=True)
class SampleRun[P: BaseModel]:
    """Fixed run artifacts supplied to a pure science samples producer.

    run_id is the accepted workflow identity, not an acquisition ID or token.
    plan is its detached native Start plan. data_root is the exact native
    acquisition directory. records is the committed Engine.records tuple in
    effect order; uncommitted files and current drafts are not evidence.
    Empty run_id raises ValueError. This envelope performs no I/O or admission.
    """

    run_id: str
    plan: P
    data_root: Path
    records: tuple[CommittedRecord, ...]

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("Sample run_id must not be empty")


class SampleField(BaseModel):
    """One producer-defined named JSON measurement, validated before append.

    name is a nonempty science field, never owner metadata run_id/point/at/source.
    value is strict finite JSON, including nested values or null when science
    explicitly permits it. Invalid names/values raise ValueError. The framework
    neither catalogs quantity names nor derives units or physical coordinates.
    """

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
    """One fully prepared row for the host's append-only sample owner.

    key is the nonempty producer identity (workflow run ID plus point index).
    fields holds unique named science values, already validated by SampleField.
    Duplicate names or empty key raise ValueError. The host only converts this
    tuple to append keyword fields; it does not repair or interpret measurements.
    """

    key: str
    fields: tuple[SampleField, ...]

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("Sample key must not be empty")
        names = [field.name for field in self.fields]
        if len(names) != len(set(names)):
            raise ValueError("Sample field names must be unique")


@dataclass(frozen=True)
class PreparedFluxdepExport:
    """Science-owned data shared by the native and independent Labber writers.

    payload holds ordered SI/discrete variables with inner-first axes. Signal
    and actual delay Time are flux-by-index matrices; the fitted lifetime is a
    Flux-only vector, not a broadcast matrix. cfg contains the complete fixed
    plan, including its declaration identity/version. Arrays are detached from
    acquisitions but not copied by this envelope. Preparation does no writes.
    """

    payload: ExperimentPayload
    cfg: CfgSnapshot


@dataclass(frozen=True)
class FluxdepExportSnapshot:
    """Host capture for a derived workflow artifact, never live GUI input.

    identity supplies the workflow run ID and historical software provenance.
    paths contains exact workflow roots; native output is data_root/merged.h5.
    snapshot preserves the Start entry UUID/name, point, description and roles.
    started_at is the host's Start time, not an acquisition timestamp.
    finished_at is host terminal delivery time, or None if unknown. Both known
    times must be timezone-aware UTC; invalid times raise ValueError. Export
    never reads devices or fabricates an acquisition/device snapshot.
    """

    identity: RunIdentity
    paths: RunPaths
    snapshot: RunSnapshot
    started_at: datetime
    finished_at: datetime | None

    def __post_init__(self) -> None:
        for name, value in (
            ("started_at", self.started_at),
            ("finished_at", self.finished_at),
        ):
            if value is not None and value.utcoffset() != timedelta(0):
                raise ValueError(f"{name} must be timezone-aware UTC")


@dataclass(frozen=True)
class FluxdepExportOutput:
    """One destination attempt from this call, not a measurement outcome.

    format is data_h5 or labber. path is the exact allocated local destination,
    retained even if its writer failed. Only failed Labber allocation may have
    path=None. status is saved or failed; saved requires a known Path and no
    error. failed requires a nonempty diagnostic. Invalid combinations raise
    ValueError. Success does not guarantee that the file still exists later.
    """

    format: Literal["data_h5", "labber"]
    path: Path | None
    status: Literal["saved", "failed"]
    error: str | None

    def __post_init__(self) -> None:
        if self.format not in {"data_h5", "labber"}:
            raise ValueError("unknown export format")
        if self.path is None and (self.format != "labber" or self.status != "failed"):
            raise ValueError("only failed Labber allocation may have path=None")
        if self.status == "saved":
            if self.error is not None:
                raise ValueError("saved output requires error=None")
        elif self.status == "failed":
            if not isinstance(self.error, str) or not self.error.strip():
                raise ValueError("failed output requires a nonempty error")
        else:
            raise ValueError("status must be saved or failed")


@dataclass(frozen=True)
class FluxdepExportReport:
    """Latest explicit export receipt; no previous successes are merged in.

    run_id identifies the workflow. outputs contains native then Labber attempts,
    including allocation failures. ledger_error is None after saved-event append,
    otherwise a nonempty diagnostic; ledger failure does not change file results.
    The caller can show successful paths even when another output or ledger fails.
    """

    run_id: str
    outputs: tuple[FluxdepExportOutput, ...]
    ledger_error: str | None
