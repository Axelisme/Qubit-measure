"""Append-only, entry-bound record rows; CSV is only an export format."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    field_validator,
)


class _RecordRow(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True, allow_inf_nan=False)

    __pydantic_extra__: dict[str, JsonValue] = Field(init=False)
    run_id: str = Field(min_length=1)
    point: str | None
    at: AwareDatetime
    source: str = Field(min_length=1)

    @field_validator("at")
    @classmethod
    def utc_time(cls, value: datetime) -> datetime:
        if value.utcoffset() != timedelta(0):
            raise ValueError("at must be UTC")
        return value


def _check_flatten_columns(record: dict[str, JsonValue]) -> None:
    """Reject distinct leaf paths that flatten to the same dotted column."""
    pending = list(record.items())
    columns: set[str] = set()
    while pending:
        column, value = pending.pop()
        if isinstance(value, dict):
            pending.extend((f"{column}.{key}", item) for key, item in value.items())
        elif column in columns:
            raise ValueError(f"Colliding flattened column {column!r}")
        else:
            columns.add(column)


class JsonlRecordTable:
    """Store record history at an exact UTF-8 JSONL path.

    Append requires a nonempty run_id, nullable point, UTC at and nonempty
    source, and finite JSON producer fields. Read only requires the nonempty
    run_id used to select each run's latest row; other fields remain opaque.
    This store neither derives quantities nor validates device units.
    Read returns the last complete row for each run_id, not a field-wise merge.
    No files are created until append. I/O and validation failures propagate;
    incomplete/corrupt lines are never silently skipped. There is no multi-file
    transaction, concurrent-writer coordination or crash recovery guarantee.
    """

    def __init__(self, path: str | Path) -> None:
        """Bind an exact caller-selected JSONL file without reading or writing it."""
        self._path = Path(path)

    @property
    def path(self) -> Path:
        """Return this table's exact bound JSONL file path without I/O.

        Binding a table does not create or read the file. The returned Path
        identifies this table's history, not its parent directory.
        """
        return self._path

    def append(
        self,
        *,
        run_id: str,
        point: str | None = None,
        source: str = "manual",
        at: datetime | None = None,
        **fields: JsonValue,
    ) -> dict[str, JsonValue]:
        """Validate/append one row and return its detached complete JSON value.

        run_id identifies the acquisition (or an explicit legacy synthetic key).
        point names the acquisition point, or None if none is known. source is
        manual, an accepted event ID, or workflow:<run_id>; the caller owns that
        reference. at is an aware UTC timestamp, or None to generate UTC now.
        fields carries producer-defined JSON measurements, including nested
        values and optional device fields, preserved without unit conversion.
        Invalid metadata or non-JSON fields fail before touching the file.
        Existing rows are never rewritten, even for an already-present run_id.
        """
        record = _RecordRow(
            run_id=run_id,
            point=point,
            at=datetime.now(timezone.utc) if at is None else at,
            source=source,
            **fields,
        )
        encoded = record.model_dump_json() + "\n"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as stream:
            stream.write(encoded)
        return record.model_dump(mode="json")

    def _latest_records(self) -> dict[str, dict[str, JsonValue]]:
        """Return last whole JSON rows per run_id in file-occurrence order.

        An absent file returns an empty mapping. Invalid rows raise ValueError
        naming path and line; other I/O errors propagate. Nested JSON remains nested.
        """
        try:
            stream = self._path.open(encoding="utf-8")
        except FileNotFoundError:
            if self._path.is_symlink():
                raise
            return {}
        latest: dict[str, dict[str, JsonValue]] = {}
        with stream:
            for number, line in enumerate(stream, start=1):
                try:
                    if not line.endswith("\n"):
                        raise ValueError("incomplete tail: expected LF")
                    record = TypeAdapter(dict[str, JsonValue]).validate_json(line)
                    run_id = record.get("run_id")
                    if not isinstance(run_id, str) or not run_id:
                        raise ValueError("run_id must be a nonempty string")
                except ValueError as exc:
                    raise ValueError(f"{self._path}:{number}: {exc}") from exc
                # Replace the whole row and keep the last occurrences' file order.
                latest.pop(run_id, None)
                latest[run_id] = record
        return latest

    def read(self, *, flatten: bool = False) -> pd.DataFrame:
        """Return each run's last whole JSONL row in last-occurrence order.

        Absent history returns an empty frame. Only JSON object syntax and the
        nonempty run_id needed for grouping are checked. Unused fields, NaN and
        metadata are preserved without schema validation. Corrupt tails or invalid
        run_id raise ValueError with path/line; I/O failures propagate.
        With flatten=True, nested object keys become dot-separated columns
        (devices.bias.value); lists remain cell values. Distinct source paths
        colliding in one dotted column raise ValueError with path, zero-based
        output row, run_id and column. The default keeps nested JSON objects as
        cells, including dotted keys. Neither form changes the source history.
        """
        rows = list(self._latest_records().values())
        if flatten:
            for index, row in enumerate(rows):
                try:
                    _check_flatten_columns(row)
                except ValueError as exc:
                    raise ValueError(
                        f"{self._path}: row {index} (run_id={row['run_id']!r}): {exc}"
                    ) from exc
            return pd.json_normalize(rows)
        return pd.DataFrame(rows)

    def latest(self, run_id: str) -> dict[str, JsonValue] | None:
        """Read one run's latest complete JSON row, or None if absent.

        run_id is a nonempty acquisition key. Validate the whole history like
        read(), checking only JSON objects and run_id, never skipping corrupt
        siblings or tails. Returned fields are detached and otherwise opaque.
        Missing history returns None; validation and I/O failures propagate.
        """
        run_id = TypeAdapter(str).validate_python(run_id, strict=True)
        if not run_id:
            raise ValueError("run_id must be nonempty")
        return self._latest_records().get(run_id)

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
            raise ValueError(f"{destination}: CSV export cannot replace record history")
        frame = self.read()
        with destination.open(
            "w" if overwrite else "x", encoding="utf-8", newline=""
        ) as stream:
            frame.to_csv(stream, index=False)
