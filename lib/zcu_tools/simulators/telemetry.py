"""Best-effort local JSONL persistence for acquire-level routing reports."""

from __future__ import annotations

import json
import warnings
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, TypedDict

from pydantic import JsonValue

from zcu_tools.resources.simulation_schema import Values

if TYPE_CHECKING:
    from zcu_tools.simulate.evolution import PathDiagnostics


class TelemetrySettings(Values):
    """Local routing logging policy, separate from numerical accuracy.

    enabled: write one acquire summary, default True. directory: optional output
    path; None uses project-root .sim_telemetry. Relative paths resolve against
    the working directory when the environment is built. No I/O occurs here.
    """

    enabled: bool = True
    directory: Path | None = None


class AcquireTelemetry(TypedDict):
    """Required caller-supplied summary for one acquire, without numerical inputs.

    points: request grid size excluding shots and rounds.
    rounds: number of captured attempts, including failed attempts.
    wall_s: admitted acquire wall seconds including pacing, excluding writer I/O.
    compile_wall_s: compilation wall seconds, through failure if it raises.
    evolve_wall_s: sum of Engine.evolve wall seconds, including failed calls.
    readout_wall_s: summed ADC/readout wall seconds, including failed calls.
    Stages not entered contribute 0.0; no stage may be unmeasured.
    complete: True if the acquire body and cleanup returned normally, including
    a partial Stop result; False if an exception escaped. It does not mean all
    requested rounds completed.
    extras: simulator-prefixed keys (for example ftf.projection_outcomes) mapped
    to JSON values; pass {} when absent. Consumers index required keys directly.
    """

    points: int
    rounds: int
    wall_s: float
    compile_wall_s: float
    evolve_wall_s: float
    readout_wall_s: float
    complete: bool
    extras: dict[str, JsonValue]


class RoutingRecord(AcquireTelemetry):
    """Persisted acquire summary with writer-generated routing fields.

    timestamp: UTC ISO timestamp.
    frontend_cpu_s: calling-thread geometry and work-plan CPU seconds.
    estimated_cost_unit: label for relative family scheduling costs.
    families: family name to summed Engine routing counters and CPU seconds;
    estimated_cpu_s is None when any contributing estimate is unavailable.
    Caller-supplied fields retain their AcquireTelemetry meanings.
    """

    timestamp: str
    frontend_cpu_s: float
    estimated_cost_unit: str
    families: dict[str, dict[str, int | float | None]]


def _families(
    paths: Sequence[PathDiagnostics],
) -> dict[str, dict[str, int | float | None]]:
    # Keep configuration imports independent of the numerical runtime.
    from zcu_tools.simulate.evolution import FamilyRouting

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
            prepare_builds=sum(f.prepare_builds for f in uses),
            admission_rows=sum(f.admission_rows for f in uses),
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
        summary: AcquireTelemetry,
        *,
        paths: Sequence[PathDiagnostics],
    ) -> None:
        """Append one summary without modifying it or persisting extra caller keys.

        summary supplies every AcquireTelemetry field, including {} for no extras.
        paths contains this acquire's round work; it may be empty after a failure
        before evolution. Fields and aggregation are described by RoutingRecord.
        Templates carry no experiment/program identifier, so none is synthesized.
        Filesystem/serialization failures follow this class's warning-only policy.
        """
        with self._lock:
            try:
                now = datetime.now(timezone.utc)
                record: RoutingRecord = {
                    "timestamp": now.isoformat(),
                    "points": summary["points"],
                    "rounds": summary["rounds"],
                    "wall_s": summary["wall_s"],
                    "frontend_cpu_s": sum(p.frontend_cpu_s for p in paths),
                    "estimated_cost_unit": "relative scheduling units",
                    "families": _families(paths),
                    "compile_wall_s": summary["compile_wall_s"],
                    "evolve_wall_s": summary["evolve_wall_s"],
                    "readout_wall_s": summary["readout_wall_s"],
                    "complete": summary["complete"],
                    "extras": summary["extras"],
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
