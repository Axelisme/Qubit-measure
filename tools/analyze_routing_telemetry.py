"""Summarize acquire wall stages, family usage and CPU calibration from JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    Discriminator,
    StrictBool,
    StrictFloat,
    StrictStr,
    Tag,
    TypeAdapter,
    ValidationError,
)


class FamilyInput(BaseModel):
    """Persisted family counters; times are CPU seconds, cost is relative."""

    rows: int
    prepare_cpu_s: float
    apply_cpu_s: float
    cache_hits: int
    rejected_rows: int
    estimated_cost: float
    estimated_cpu_s: float | None


class AcquireInput(BaseModel):
    """Validated routing row and optional wall measurements for analysis.

    families maps family names to CPU/counter entries, even for unmeasured rows.
    complete is normal return (True), escaping error (False), or unknown (None).
    wall_s is total acquire wall seconds. compile_wall_s measures compilation;
    evolve_wall_s and readout_wall_s sum evolution and readout wall seconds.
    materialize_wall_s measures in-acquire raw materialization. acquire_id links
    the summary to deferred events. Both may be absent in older summaries.
    Missing or null wall fields mean unmeasured, not zero. Non-null complete
    must be a JSON boolean; non-null wall fields must be JSON numbers, not bools
    or numeric strings. Wrong types invalidate the entire row.
    """

    families: dict[str, FamilyInput]
    acquire_id: StrictStr | None = None
    materialize_wall_s: StrictFloat | None = None
    complete: StrictBool | None = None
    wall_s: StrictFloat | None = None
    compile_wall_s: StrictFloat | None = None
    evolve_wall_s: StrictFloat | None = None
    readout_wall_s: StrictFloat | None = None


class MaterializeInput(BaseModel):
    """Successful deferred raw work in wall seconds for an opaque acquire ID."""

    event: Literal["materialize"]
    acquire_id: StrictStr
    materialize_wall_s: StrictFloat


def _routing_tag(row: object) -> Literal["acquire", "materialize"]:
    if isinstance(row, MaterializeInput) or (
        isinstance(row, dict) and row.get("event") == "materialize"
    ):
        return "materialize"
    return "acquire"


_routing_record = TypeAdapter(
    Annotated[
        Annotated[AcquireInput, Tag("acquire")]
        | Annotated[MaterializeInput, Tag("materialize")],
        Discriminator(_routing_tag),
    ]
)


def decode_routing_record(line: str | bytes) -> AcquireInput | MaterializeInput:
    """Decode an acquire summary or deferred raw timing; invalid rows raise ValidationError."""
    return _routing_record.validate_json(line)


@dataclass
class AcquireWallTotals:
    """Measured-only totals and coverage counts for a caller's group of acquires.

    included counts records with all four wall measurements; excluded counts
    the others. wall_s is total acquire wall seconds. compile_wall_s,
    evolve_wall_s and readout_wall_s sum the corresponding stages in seconds.
    remainder_s is acquire wall seconds not accounted for by those stages.
    """

    included: int = 0
    excluded: int = 0
    wall_s: float = 0.0
    compile_wall_s: float = 0.0
    evolve_wall_s: float = 0.0
    readout_wall_s: float = 0.0
    remainder_s: float = 0.0

    def add(self, record: AcquireInput) -> None:
        """Count every valid row; sum only rows with all four wall measurements."""
        if (
            record.wall_s is None
            or record.compile_wall_s is None
            or record.evolve_wall_s is None
            or record.readout_wall_s is None
        ):
            self.excluded += 1
            return
        self.included += 1
        self.wall_s += record.wall_s
        self.compile_wall_s += record.compile_wall_s
        self.evolve_wall_s += record.evolve_wall_s
        self.readout_wall_s += record.readout_wall_s
        self.remainder_s += (
            record.wall_s
            - record.compile_wall_s
            - record.evolve_wall_s
            - record.readout_wall_s
        )


@dataclass
class FamilySummary:
    """Summed work, plus calibration against relative costs and CPU estimates."""

    rows: int = 0
    cache_hits: int = 0
    rejected_rows: int = 0
    prepare_cpu_s: float = 0.0
    apply_cpu_s: float = 0.0
    estimated_cost: float = 0.0
    estimated_cpu_s: float = 0.0
    covered_actual_cpu_s: float = 0.0
    estimated_records: int = 0
    usage_fraction: float = 0.0
    cpu_s_per_cost_unit: float | None = None
    cpu_relative_error: float | None = None

    def add(self, family: FamilyInput) -> None:
        """Accumulate one family's acquire entry; omit unknown CPU estimates."""
        self.rows += family.rows
        self.cache_hits += family.cache_hits
        self.rejected_rows += family.rejected_rows
        self.prepare_cpu_s += family.prepare_cpu_s
        self.apply_cpu_s += family.apply_cpu_s
        self.estimated_cost += family.estimated_cost
        if family.estimated_cpu_s is not None:
            self.estimated_cpu_s += family.estimated_cpu_s
            self.covered_actual_cpu_s += family.prepare_cpu_s + family.apply_cpu_s
            self.estimated_records += 1

    def finish(self, total_rows: int) -> None:
        """Calculate usage share and weighted actual-minus-estimate CPU error."""
        self.usage_fraction = self.rows / total_rows if total_rows else 0.0
        if self.estimated_cost > 0:
            self.cpu_s_per_cost_unit = (
                self.prepare_cpu_s + self.apply_cpu_s
            ) / self.estimated_cost
        if self.estimated_cpu_s > 0:
            self.cpu_relative_error = (
                self.covered_actual_cpu_s - self.estimated_cpu_s
            ) / self.estimated_cpu_s


def main(argv: Sequence[str] | None = None) -> int:
    """Print aggregate JSON for files/directories; skip malformed lines with warnings.

    Paths may name routing JSONL files or directories containing routing-*.jsonl.
    File read errors propagate. Validation failures are reported with file/line
    and excluded, including from family counters; skipped counts these rows.
    wall_by_complete groups normal_return (True, including partial Stop), raised
    (False) and unknown (missing/null complete). Only observed groups are emitted.
    Each group counts included/excluded rows and sums wall_s, compile_wall_s,
    evolve_wall_s, readout_wall_s and remainder_s (total minus the three stages).
    A row contributes times only when all four wall fields are measured.
    Non-null complete must be a JSON boolean; non-null wall fields must be JSON
    numbers (not booleans or strings). Wrong types skip the entire record.
    Only acquire summaries contribute counters; deferred raw events are skipped
    without warnings. Multiple inputs merge into one
    report without inferring simulator identity. No files are modified.
    Returns zero after a readable run. argparse handles invalid CLI arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        type=Path,
        nargs="+",
        help="routing JSONL files or directories containing routing-*.jsonl",
    )
    args = parser.parse_args(argv)
    files: set[Path] = set()
    for path in args.paths:
        files.update(path.glob("routing-*.jsonl") if path.is_dir() else (path,))
    families: dict[str, FamilySummary] = {}
    wall_by_complete: dict[str, AcquireWallTotals] = {}
    records = skipped = 0
    for path in sorted(files):
        with path.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                try:
                    record = decode_routing_record(line)
                except ValidationError as error:
                    print(
                        f"{path}:{number}: skipping invalid record: {error}",
                        file=sys.stderr,
                    )
                    skipped += 1
                    continue
                if isinstance(record, MaterializeInput):
                    continue
                records += 1
                group = (
                    "unknown"
                    if record.complete is None
                    else "normal_return"
                    if record.complete
                    else "raised"
                )
                wall_by_complete.setdefault(group, AcquireWallTotals()).add(record)
                for name, family in record.families.items():
                    families.setdefault(name, FamilySummary()).add(family)
    total_rows = sum(f.rows for f in families.values())
    for family in families.values():
        family.finish(total_rows)
    print(
        json.dumps(
            {
                "records": records,
                "skipped": skipped,
                "estimated_cost_unit": "relative scheduling units",
                "families": {name: asdict(f) for name, f in sorted(families.items())},
                "wall_by_complete": {
                    name: asdict(summary)
                    for name, summary in sorted(wall_by_complete.items())
                },
            },
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
