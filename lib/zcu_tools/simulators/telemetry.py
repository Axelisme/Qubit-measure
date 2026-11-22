"""Best-effort local JSONL persistence for acquire routing and raw-shot timing."""

from __future__ import annotations

import json
import warnings
from collections.abc import Generator, Iterable, Sequence
from contextlib import contextmanager, suppress
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Literal, TypedDict

from pydantic import JsonValue, StrictBool, StrictStr

from zcu_tools.resources.simulation_schema import Values

if TYPE_CHECKING:
    from zcu_tools.simulate.evolution import PathDiagnostics


class TelemetrySettings(Values):
    """Local routing logging policy, separate from numerical accuracy.

    enabled: write one acquire summary, default True. directory: optional output
    path; None uses project-root .sim_telemetry. Relative paths resolve against
    the working directory when the environment is built. commit is the checkout
    revision supplied by the capture caller. dirty reports whether that checkout
    has uncommitted changes. Either may be None if not supplied. No I/O occurs here.
    """

    enabled: bool = True
    directory: Path | None = None
    commit: StrictStr | None = None
    dirty: StrictBool | None = None


class AcquireTelemetry(TypedDict):
    """Required caller-supplied summary for one acquire, without numerical inputs.

    acquire_id: opaque identity used to link deferred materialization to this row.
    points: request grid size excluding shots and rounds.
    rounds: number of captured attempts, including failed attempts.
    wall_s: admitted acquire wall seconds including pacing, excluding writer I/O.
    compile_wall_s: compilation wall seconds, through failure if it raises.
    evolve_wall_s: sum of Engine.evolve wall seconds, including failed calls.
    readout_wall_s: summed ADC/readout wall seconds, including failed calls.
    materialize_wall_s: summed raw-shot materialization wall seconds during the
    acquire, including failed calls; excludes subsequent shot processing.
    Each stage is None if never entered. A measured zero is 0.0.
    complete: True if the acquire body and cleanup returned normally, including
    a partial Stop result; False if an exception escaped. It does not mean all
    requested rounds completed.
    extras: simulator-prefixed keys (for example ftf.projection_outcomes) mapped
    to JSON values; pass {} when absent. Consumers index required keys directly.
    """

    acquire_id: str
    points: int
    rounds: int
    wall_s: float
    compile_wall_s: float | None
    evolve_wall_s: float | None
    readout_wall_s: float | None
    materialize_wall_s: float | None
    complete: bool
    extras: dict[str, JsonValue]
    capture_file: str | None
    """Published capture basename; None when recording is off or publication fails."""


class RoutingRecord(AcquireTelemetry):
    """Persisted acquire summary with writer-generated routing fields.

    timestamp: UTC ISO timestamp.
    commit: caller-supplied checkout revision or None if unknown.
    dirty: caller-supplied checkout modification flag or None if unknown.
    frontend_cpu_s: calling-thread geometry and work-plan CPU seconds.
    estimated_cost_unit: label for relative family scheduling costs.
    families: family name to merged acquire routing counters, CPU seconds and
    optional OpenCL detection error. estimated_cpu_s is None when any
    contributing estimate is unavailable. Engine tableau diagnostics are not
    persisted.
    Unentered stages are None for complete=False and 0.0 for complete=True
    (the stage was not needed). Measured times, including 0.0, are unchanged.
    Other caller-supplied fields retain their AcquireTelemetry meanings.
    """

    timestamp: str
    commit: str | None
    dirty: bool | None
    frontend_cpu_s: float
    estimated_cost_unit: str
    families: dict[str, dict[str, int | float | str | None]]


class MaterializeRecord(TypedDict):
    """Successful deferred raw work; not an acquire or part of its wall_s.

    event: record discriminator, always "materialize".
    acquire_id: the originating acquire row's opaque identity.
    materialize_wall_s: elapsed seconds for this raw materialization only.
    timestamp: UTC ISO timestamp of supplemental publication.
    commit: caller-supplied checkout revision or None if unknown.
    dirty: caller-supplied checkout modification flag or None if unknown.
    """

    event: Literal["materialize"]
    acquire_id: str
    materialize_wall_s: float
    timestamp: str
    commit: str | None
    dirty: bool | None


def _families(
    paths: Sequence[PathDiagnostics],
) -> dict[str, dict[str, int | float | str | None]]:
    # Keep configuration imports independent of the numerical runtime.
    from zcu_tools.simulate.evolution import merge_path_diagnostics

    # Declare the JSONL fields here so new Engine diagnostics do not expand the schema.
    return {
        name: {
            "rows": report.rows,
            "prepare_cpu_s": report.prepare_cpu_s,
            "apply_cpu_s": report.apply_cpu_s,
            "cache_hits": report.cache_hits,
            "rejected_rows": report.rejected_rows,
            "estimated_cost": report.estimated_cost,
            "estimated_cpu_s": report.estimated_cpu_s,
            "prepare_builds": report.prepare_builds,
            "admission_rows": report.admission_rows,
            "duration_ns": report.duration_ns,
            "opencl_rows": report.opencl_rows,
            "opencl_detection_error": report.opencl_detection_error,
            "actual_cpu_s": report.actual_cpu_s,
        }
        for name, report in merge_path_diagnostics(paths).routing.items()
    }


def sum_stage_seconds(spans: Iterable[float | None]) -> float | None:
    """Sum measured wall seconds; ignore None spans and return None if none ran.

    Pass timings in seconds from the same acquire stage, in any order. A timed
    span of 0.0 counts as a measurement. Errors from the input iterable propagate.
    """
    measured = [seconds for seconds in spans if seconds is not None]
    return sum(measured) if measured else None


def _stage_seconds(seconds: float | None, *, complete: bool) -> float | None:
    return 0.0 if seconds is None and complete else seconds


class RoutingTelemetry:
    """Write bounded UTF-8 JSONL for acquire summaries and deferred raw timing.

    directory is the local output directory, created lazily on write. max_bytes
    is the positive total routing-*.jsonl budget (default 50 MiB). Before append,
    delete oldest owned files by mtime/name until the incoming line fits, including
    today's file if necessary. Unrelated files are never removed. A single line
    larger than the budget is not written. A budget below one raises ValueError.
    Files use UTC dates and timestamps. Supply the checkout revision (commit)
    and modification flag (dirty) for the capture. Both are copied to every row;
    unspecified values remain null. The writer does not inspect Git.
    Each instance serializes its writes. Filesystem/serialization failures emit
    at most one RuntimeWarning per instance and never escape, even under warning
    filters that turn warnings into exceptions. No numerical inputs are retained.
    """

    def __init__(
        self,
        directory: Path,
        *,
        max_bytes: int = 50 * 1024**2,
        commit: str | None = None,
        dirty: bool | None = None,
    ) -> None:
        if max_bytes < 1:
            raise ValueError("routing telemetry max_bytes must be positive")
        self.directory = directory
        self.max_bytes = max_bytes
        self._commit = commit
        self._dirty = dirty
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
        before evolution. Unentered stages supplied as None stay null on failed
        acquires and become 0.0 on normal return. Measured zeros stay numeric.
        Templates carry no experiment/program identifier, so none is synthesized.
        Filesystem/serialization failures follow this class's warning-only policy.
        """
        now = datetime.now(timezone.utc)
        complete = summary["complete"]
        with self._writing():
            record: RoutingRecord = {
                "timestamp": now.isoformat(),
                "commit": self._commit,
                "dirty": self._dirty,
                "acquire_id": summary["acquire_id"],
                "points": summary["points"],
                "rounds": summary["rounds"],
                "wall_s": summary["wall_s"],
                "frontend_cpu_s": sum(p.frontend_cpu_s for p in paths),
                "estimated_cost_unit": "relative scheduling units",
                "families": _families(paths),
                "compile_wall_s": _stage_seconds(
                    summary["compile_wall_s"], complete=complete
                ),
                "evolve_wall_s": _stage_seconds(
                    summary["evolve_wall_s"], complete=complete
                ),
                "readout_wall_s": _stage_seconds(
                    summary["readout_wall_s"], complete=complete
                ),
                "materialize_wall_s": _stage_seconds(
                    summary["materialize_wall_s"], complete=complete
                ),
                "complete": complete,
                "extras": summary["extras"],
                "capture_file": summary["capture_file"],
            }
            self._append(now, record)

    def write_materialize(self, acquire_id: str, elapsed_s: float) -> None:
        """Append successful deferred raw work for an existing acquire identity.

        elapsed_s is the raw materialization's elapsed wall seconds, separate
        from the original row's wall_s. The original row is never rewritten.
        This append uses the same budget and warning-only policy as write().
        """
        now = datetime.now(timezone.utc)
        with self._writing():
            self._append(
                now,
                {
                    "event": "materialize",
                    "acquire_id": acquire_id,
                    "materialize_wall_s": elapsed_s,
                    "timestamp": now.isoformat(),
                    "commit": self._commit,
                    "dirty": self._dirty,
                },
            )

    def _append(self, now: datetime, record: RoutingRecord | MaterializeRecord) -> None:
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
        with (self.directory / f"routing-{now:%Y%m%d}.jsonl").open("ab") as stream:
            stream.write(line)

    @contextmanager
    def _writing(self) -> Generator[None]:
        with self._lock:
            try:
                yield
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
