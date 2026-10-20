"""Summarize family usage, rejection counts and planner calibration from JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from pydantic import BaseModel, ValidationError


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
    """Only the routing families are needed for aggregate analysis."""

    families: dict[str, FamilyInput]


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
    and excluded; the JSON summary includes the number of skipped records.
    Returns zero after a readable run. argparse handles invalid CLI arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="+")
    args = parser.parse_args(argv)
    files: set[Path] = set()
    for path in args.paths:
        files.update(path.glob("routing-*.jsonl") if path.is_dir() else (path,))
    families: dict[str, FamilySummary] = {}
    records = skipped = 0
    for path in sorted(files):
        with path.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                try:
                    record = AcquireInput.model_validate_json(line)
                except ValidationError as error:
                    print(
                        f"{path}:{number}: skipping invalid record: {error}",
                        file=sys.stderr,
                    )
                    skipped += 1
                    continue
                records += 1
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
            },
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
