"""Append-only, entry-bound sample rows; CSV is only an export format."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
)


class _SampleRecord(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True, allow_inf_nan=False)

    __pydantic_extra__: dict[str, JsonValue] = Field(init=False)
    run_id: str = Field(min_length=1)
    point: str | None
    at: AwareDatetime
    source: str = Field(min_length=1)
    dev_value: float
    dev_unit: Literal["A", "V", "native"]

    @field_validator("at")
    @classmethod
    def utc_time(cls, value: datetime) -> datetime:
        if value.utcoffset() != timedelta(0):
            raise ValueError("at must be UTC")
        return value


class JsonlSampleTable:
    """Store sample history at an exact UTF-8 JSONL path.

    Each row has a nonempty run_id, nullable point, UTC at, nonempty source,
    finite dev_value and dev_unit (A, V or native). Other fields are JSON values
    declared by the producer; this store neither names nor derives quantities.
    Read returns the last complete row for each run_id, not a field-wise merge.
    No files are created until append. I/O and validation failures propagate;
    incomplete/corrupt lines are never silently skipped. There is no multi-file
    transaction, concurrent-writer coordination or crash recovery guarantee.
    """

    def __init__(self, path: str | Path) -> None:
        """Bind an exact caller-selected JSONL file without reading or writing it."""
        self._path = Path(path)

    def append(
        self,
        *,
        run_id: str,
        dev_value: float,
        dev_unit: Literal["A", "V", "native"],
        point: str | None = None,
        source: str = "manual",
        **fields: JsonValue,
    ) -> None:
        """Validate and append one row, creating parent directories if necessary.

        run_id identifies the acquisition (or an explicit legacy synthetic key).
        dev_value is the raw device value in dev_unit; it is not converted.
        point names the acquisition point, or None if none is known. source is
        manual, an accepted event ID, or workflow:<run_id>; the caller owns that
        reference. at is generated as UTC now and cannot be supplied in fields.
        fields carries producer-defined JSON measurements, including nested
        values. Invalid or missing coordinates fail before touching the file.
        Existing rows are never rewritten, even for an already-present run_id.
        """
        record = _SampleRecord(
            run_id=run_id,
            point=point,
            at=datetime.now(timezone.utc),
            source=source,
            dev_value=dev_value,
            dev_unit=dev_unit,
            **fields,
        )
        encoded = record.model_dump_json() + "\n"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as stream:
            stream.write(encoded)

    def read(self) -> pd.DataFrame:
        """Return latest rows per run_id with a fresh RangeIndex.

        Rows retain their last occurrence's file order. An absent file returns
        an empty DataFrame. Invalid rows raise ValueError naming path and line;
        other I/O errors propagate. Nested measurement JSON remains nested.
        """
        try:
            stream = self._path.open(encoding="utf-8")
        except FileNotFoundError:
            if self._path.is_symlink():
                raise
            return pd.DataFrame()
        latest: dict[str, _SampleRecord] = {}
        with stream:
            for number, line in enumerate(stream, start=1):
                try:
                    if not line.endswith("\n"):
                        raise ValueError("incomplete tail: expected LF")
                    record = _SampleRecord.model_validate_json(line)
                except ValueError as exc:
                    raise ValueError(f"{self._path}:{number}: {exc}") from exc
                # Replace the whole row and keep the last occurrences' file order.
                latest.pop(record.run_id, None)
                latest[record.run_id] = record
        return pd.DataFrame([row.model_dump(mode="json") for row in latest.values()])

    def export_csv(self, path: str | Path, *, overwrite: bool = False) -> None:
        """Export latest rows to an exact CSV path without a pandas index.

        Refuse existing paths with FileExistsError unless overwrite=True.
        A destination identifying this table's own history raises ValueError
        regardless of overwrite (including symlinks and hardlinks). Only this
        source is protected; this check is not a concurrent-writer guarantee.
        Parent directories must already exist. CSV is not a history store;
        nested measurements use pandas' textual CSV representation. Read and
        I/O failures propagate, and export is not an atomic replacement.
        """
        destination = Path(path)
        if (
            destination.exists()
            and self._path.exists()
            and destination.samefile(self._path)
        ):
            raise ValueError(f"{destination}: CSV export cannot replace sample history")
        frame = self.read()
        with destination.open(
            "w" if overwrite else "x", encoding="utf-8", newline=""
        ) as stream:
            frame.to_csv(stream, index=False)
