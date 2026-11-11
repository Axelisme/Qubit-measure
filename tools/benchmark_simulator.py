"""Measure public simulator acquires; imports and setup are outside the timers."""

import argparse
import hashlib
import json
import logging
import platform
import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from importlib.metadata import version
from pathlib import Path
from statistics import median
from typing import Literal

from benchmark_simulator_workloads import (
    AcquireSession,
    Workload,
    make_workloads,
    open_session,
)
from pydantic import BaseModel

SessionFactory = Callable[[Workload], AbstractContextManager[AcquireSession]]
Mode = Literal["first_acquire", "warm_clear_history", "warm_retain_history"]


class TimingSample(BaseModel):
    """Elapsed wall and process CPU seconds for one successful acquire only."""

    wall_seconds: float
    cpu_seconds: float


class TimingSummary(BaseModel):
    """Median and median absolute deviation (MAD), both in seconds."""

    median_seconds: float
    mad_seconds: float


class ModeResult(BaseModel):
    """One history mode, its ordered raw samples and wall/CPU summaries."""

    mode: Mode
    samples: list[TimingSample]
    wall: TimingSummary
    cpu: TimingSummary


class WorkloadResult(BaseModel):
    """Measured workload inputs, SHA-256 fingerprints and per-mode timings."""

    workload: Workload
    workload_sha256: str
    configuration_sha256: str
    modes: list[ModeResult]


def run_workload(
    workload: Workload,
    session_factory: SessionFactory,
    *,
    wall_clock: Callable[[], float],
    cpu_clock: Callable[[], float],
) -> WorkloadResult:
    """Measure five acquires per mode using fresh or fixed-warmup sessions.

    Factory/setup, cleanup, two warmup acquires and history.clear are untimed.
    First mode uses five independent environments, not five fresh processes.
    Each warm mode uses its own environment and two identical warmup acquires.
    Factory/acquire/cleanup failures propagate; no partial result is returned.
    Clocks return monotonic seconds (wall) and process CPU seconds respectively.
    """
    modes: list[ModeResult] = []
    for mode in ("first_acquire", "warm_clear_history", "warm_retain_history"):
        samples: list[TimingSample] = []
        if mode == "first_acquire":
            for _ in range(5):
                with session_factory(workload) as session:
                    samples.append(_measure(session, wall_clock, cpu_clock))
        else:
            with session_factory(workload) as session:
                for _ in range(2):
                    session.acquire()
                for _ in range(5):
                    if mode == "warm_clear_history":
                        session.clear_history()
                    samples.append(_measure(session, wall_clock, cpu_clock))
        modes.append(
            ModeResult(
                mode=mode,
                samples=samples,
                wall=_summarize([s.wall_seconds for s in samples]),
                cpu=_summarize([s.cpu_seconds for s in samples]),
            )
        )
    return WorkloadResult(
        workload=workload,
        workload_sha256=_digest(workload.model_dump(mode="json")),
        configuration_sha256=_digest(workload.configuration),
        modes=modes,
    )


class BenchmarkResult(BaseModel):
    """Complete successful run: method contract, library versions and workload results.

    schema_version identifies the JSON layout. measurement records repeat/warmup
    counts and mode semantics; libraries maps distribution names to versions.
    workloads preserves caller order. Failed acquires never produce this record.
    """

    schema_version: int = 1
    measurement: dict[str, str | int]
    libraries: dict[str, str]
    workloads: list[WorkloadResult]


def run_benchmarks(
    workloads: list[Workload],
    session_factory: SessionFactory = open_session,
    *,
    wall_clock: Callable[[], float] = time.perf_counter,
    cpu_clock: Callable[[], float] = time.process_time,
) -> BenchmarkResult:
    """Measure caller workloads in order; record library versions and mode semantics.

    Default factory builds synthetic public-acquire sessions. Injection replaces
    only acquisition/environment effects, not benchmark mode or timing logic.
    Any setup/acquire/cleanup failure propagates without a partial result.
    """
    return BenchmarkResult(
        measurement={
            "version": 1,
            "repetitions": 5,
            "warmup_acquires": 2,
            "first_acquire": "new environment per sample; same process; disk cache unchanged",
            "warm_clear_history": "one environment; two untimed acquires; clear before every sample",
            "warm_retain_history": "one environment; two untimed acquires; retain history",
            "timers": "perf_counter wall seconds and process_time CPU seconds around public acquire only",
            "dispersion": "median absolute deviation (MAD), seconds",
        },
        libraries={
            "python": platform.python_version(),
            **{
                name: version(name)
                for name in (
                    "zcu-tools",
                    "numpy",
                    "scipy",
                    "numba",
                    "qick",
                    "scqubits",
                    "pydantic",
                )
            },
        },
        workloads=[
            run_workload(
                workload, session_factory, wall_clock=wall_clock, cpu_clock=cpu_clock
            )
            for workload in workloads
        ],
    )


class ComparisonRow(BaseModel):
    """One workload/mode/clock pair: old/new medians in seconds and percent change.

    Positive percent_change means slower; None means the old median was zero.
    metric is wall or cpu. Comparison never applies a performance pass threshold.
    """

    workload: str
    mode: Mode
    metric: Literal["wall", "cpu"]
    before_seconds: float
    after_seconds: float
    percent_change: float | None


def compare_results(
    before: BenchmarkResult, after: BenchmarkResult
) -> list[ComparisonRow]:
    """Compare compatible complete runs; raise ValueError for input/method mismatch.

    Workload names, versions, configurations, fingerprints, measurement contract,
    schema and history modes must match. Library versions may differ. Each row
    reports both medians and (after/before - 1)*100; zero baseline yields None.
    """
    if (
        before.schema_version != after.schema_version
        or before.measurement != after.measurement
    ):
        raise ValueError("incompatible schema or measurement modes")
    if [w.workload for w in before.workloads] != [w.workload for w in after.workloads]:
        raise ValueError("incompatible workloads or configurations")
    rows: list[ComparisonRow] = []
    for old, new in zip(before.workloads, after.workloads, strict=True):
        if (old.workload_sha256, old.configuration_sha256) != (
            new.workload_sha256,
            new.configuration_sha256,
        ):
            raise ValueError("incompatible workload fingerprints")
        if [m.mode for m in old.modes] != [m.mode for m in new.modes]:
            raise ValueError("incompatible measurement modes")
        for old_mode, new_mode in zip(old.modes, new.modes, strict=True):
            for metric in ("wall", "cpu"):
                old_median = getattr(old_mode, metric).median_seconds
                new_median = getattr(new_mode, metric).median_seconds
                rows.append(
                    ComparisonRow(
                        workload=old.workload.name,
                        mode=old_mode.mode,
                        metric=metric,
                        before_seconds=old_median,
                        after_seconds=new_median,
                        percent_change=100 * (new_median / old_median - 1)
                        if old_median
                        else None,
                    )
                )
    return rows


def main(
    argv: list[str] | None = None,
    *,
    session_factory: SessionFactory = open_session,
    workloads: list[Workload] | None = None,
) -> int:
    """Write a complete JSON run; return nonzero on acquire or comparison failure.

    argv is CLI arguments without the executable. Optional workloads/factory
    replace synthetic acquisition inputs for in-process callers. Failures leave
    the requested output untouched and print their cause to stderr.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", required=True, type=Path, help="Complete benchmark JSON destination"
    )
    parser.add_argument(
        "--compare",
        type=Path,
        help="Previous compatible JSON; print median differences",
    )
    args = parser.parse_args(argv)
    try:
        result = run_benchmarks(
            make_workloads() if workloads is None else workloads, session_factory
        )
        if args.compare is not None:
            before = BenchmarkResult.model_validate_json(
                args.compare.read_text(encoding="utf-8")
            )
            for row in compare_results(before, result):
                change = (
                    "n/a (zero baseline)"
                    if row.percent_change is None
                    else f"{row.percent_change:+.2f}%"
                )
                print(
                    f"{row.workload} {row.mode} {row.metric}: {row.before_seconds:.6f}s -> {row.after_seconds:.6f}s ({change})"
                )
        args.output.write_text(
            result.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
    except Exception:
        # CLI isolation: preserve the cause and never publish a partial run.
        logging.exception("Benchmark failed; no complete result was written")
        return 1
    return 0


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _measure(
    session: AcquireSession,
    wall_clock: Callable[[], float],
    cpu_clock: Callable[[], float],
) -> TimingSample:
    wall_start = wall_clock()
    cpu_start = cpu_clock()
    session.acquire()
    cpu_elapsed = cpu_clock() - cpu_start
    wall_elapsed = wall_clock() - wall_start
    return TimingSample(wall_seconds=wall_elapsed, cpu_seconds=cpu_elapsed)


def _summarize(values: list[float]) -> TimingSummary:
    center = median(values)
    return TimingSummary(
        median_seconds=center, mad_seconds=median([abs(v - center) for v in values])
    )


if __name__ == "__main__":
    raise SystemExit(main())
