"""Best-effort local JSONL persistence for acquire-level routing reports."""

import json
import warnings
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import TypedDict

from zcu_tools.simulate.evolution import FamilyRouting, PathDiagnostics


class _Record(TypedDict):
    timestamp: str
    points: int
    rounds: int
    wall_s: float
    estimated_cost_unit: str
    families: dict[str, dict[str, int | float | None]]


def _families(
    paths: Sequence[PathDiagnostics],
) -> dict[str, dict[str, int | float | None]]:
    grouped: dict[str, list[FamilyRouting]] = {}
    for report in paths:
        for name, family in report.routing.items():
            grouped.setdefault(name, []).append(family)
    result = {}
    for name, uses in grouped.items():
        seconds = [f.estimated_cpu_s for f in uses]
        total = FamilyRouting(
            rows=sum(f.rows for f in uses),
            prepare_cpu_s=sum(f.prepare_cpu_s for f in uses),
            apply_cpu_s=sum(f.apply_cpu_s for f in uses),
            cache_hits=sum(f.cache_hits for f in uses),
            rejected_rows=sum(f.rejected_rows for f in uses),
            estimated_cost=sum(f.estimated_cost for f in uses),
            estimated_cpu_s=(
                sum(s for s in seconds if s is not None)
                if all(s is not None for s in seconds)
                else None
            ),
        )
        result[name] = {**asdict(total), "actual_cpu_s": total.actual_cpu_s}
    return result


class RoutingTelemetry:
    """Write one UTF-8 JSON line per acquire and bound owned routing files.

    directory is the local output directory, created lazily on write. max_bytes
    is the positive total routing-*.jsonl budget (default 50 MiB). Before append,
    delete oldest owned files by mtime/name until the incoming line fits, including
    today's file if necessary. Unrelated files are never removed. A single line
    larger than the budget is not written. Files use UTC dates and timestamps.
    Each instance serializes its writes. Filesystem/serialization failures emit
    at most one RuntimeWarning per instance and never escape, even under warning
    filters that turn warnings into exceptions. No numerical inputs are retained.
    """

    def __init__(self, directory: Path, *, max_bytes: int = 50 * 1024**2) -> None:
        if max_bytes < 1:
            raise ValueError("routing telemetry max_bytes must be positive")
        self.directory = directory
        self.max_bytes = max_bytes
        self._warned = False
        self._lock = Lock()

    def write(
        self,
        *,
        points: int,
        wall_s: float,
        rounds: int,
        paths: Sequence[PathDiagnostics],
    ) -> None:
        """Append an acquire summary; paths contains this acquire's round work.

        points counts the request grid once, not shots or software rounds. wall_s
        measures admitted acquire wall time including pacing, excluding this I/O.
        paths may be empty after a failure before evolution. Templates currently
        carry no experiment/program identifier, so no name is synthesized.
        """
        with self._lock:
            try:
                now = datetime.now(timezone.utc)
                record: _Record = {
                    "timestamp": now.isoformat(),
                    "points": points,
                    "rounds": rounds,
                    "wall_s": wall_s,
                    "estimated_cost_unit": "relative scheduling units",
                    "families": _families(paths),
                }
                line = (
                    json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n"
                ).encode("utf-8")
                if len(line) > self.max_bytes:
                    raise ValueError("routing record exceeds the directory budget")
                self.directory.mkdir(parents=True, exist_ok=True)
                files = sorted(
                    (
                        (p.stat().st_mtime_ns, p.name, p, p.stat().st_size)
                        for p in self.directory.glob("routing-*.jsonl")
                    ),
                )
                total = sum(size for _, _, _, size in files)
                for _, _, path, size in files:
                    if total + len(line) <= self.max_bytes:
                        break
                    path.unlink()
                    total -= size
                with (self.directory / f"routing-{now:%Y%m%d}.jsonl").open(
                    "ab"
                ) as stream:
                    stream.write(line)
            except (OSError, ValueError, TypeError) as error:
                if not self._warned:
                    self._warned = True
                    # Optional diagnostics remain harmless under -W error.
                    with suppress(Warning):
                        warnings.warn(
                            f"routing telemetry could not be written: {error}",
                            RuntimeWarning,
                            stacklevel=2,
                        )
