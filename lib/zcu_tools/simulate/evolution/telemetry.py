"""Routing reports, evolve-local CPU accounting and validity warnings."""

import time
import warnings
from collections.abc import Generator, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Lock
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .secular import StarkOrderInversion


@dataclass(frozen=True)
class FamilyRouting:
    """One kernel family's observed work in a single evolve/acquire report.

    rows: logical dispatched segments, including adopted shared prefixes.
    duration_ns: summed physical segment lengths in ns, with the same logical
        consumers as rows. Shared prefixes count once per adopting timeline,
        unlike CPU, which counts executed work once. Reuse preserves duration
        even with zero CPU. Rejected-only families have zero rows and duration.
    prepare_cpu_s: summed calling-thread CPU seconds in guards, preparation,
        and input binding (including rejected work).
    apply_cpu_s: summed calling-thread CPU seconds in kernel apply calls.
    cache_hits: logical segment consumers served by a caller-retained complete
        preparation. Evolve itself always executes the numerical work.
    rejected_rows: rows actually rejected by this family's numerical guards;
        no threshold/reason detail and no cost-preference skips.
    estimated_cost: sum of relative scheduling units for executed segments.
    estimated_cpu_s: selected-plan CPU seconds, or None when any selected
        segment lacks a seconds estimate. Not inferred from scheduling units.
    prepare_builds: coefficient products built by common_frame, or exact-key
        frame construction attempts by floquet (including invalid frames).
        Zero for families without product instrumentation.
    prepare_key_builds: key slots passed to scheduled prepare calls or returned
        in guard preparations. Rebuilding a key counts again. Rejected guard
        products count if returned; guard-only checks without products do not.
        For multitone_gate, counts completed numerical builds, not subsequent
        prepare calls reusing a key within the same evolve.
    admission_rows: common_frame counts row indicator evaluations; multitone_rwa
        counts sampled rows with a completed RWA/full-drive spot comparison,
        whether accepted or rejected. Neither counts apply or reuse hits.
        For multitone_rwa, row proxies, cost skips and memo hits do not count.
        Other families leave this count zero.
    probe_memo_hits: sampled multitone_rwa probes served by the preparation memo.
    probe_integration_cpu_s: calling-thread CPU seconds in both sampled probe
        integrators, excluding payload construction and memo lookups. This is
        measured within prepare_cpu_s, not added to it. Both probe fields are
        zero for other families and on complete-result reuse.
    opencl_rows: physical endpoints executed by OpenCL within this family.
        Zero for CPU-only calls and reuse hits. The backend does not rename
        the family or change the logical row counts.
    opencl_detection_error: reason automatic DOP853 device detection or the
        active dimension's compilation failed, or None when neither failed.
        Populated only for DOP853. Merged reports retain the first nonempty
        reason in input order.
    dp54_rows: physical endpoints executed by DP5(4) within dop853, across devices.
    dop853_rows: physical endpoints executed by DOP853 within dop853, across
        devices. Both tableau counts are zero for reuse and other families;
        execution of a shared prefix counts once, not once per consumer.
    CPU values exclude dispatch, geometry, ADC and I/O. All values are
    nonnegative; actual_cpu_s is the sum of the two measured phases.
    """

    rows: int = 0
    prepare_cpu_s: float = 0.0
    apply_cpu_s: float = 0.0
    cache_hits: int = 0
    rejected_rows: int = 0
    estimated_cost: float = 0.0
    estimated_cpu_s: float | None = None
    prepare_builds: int = 0
    admission_rows: int = 0
    duration_ns: float = 0.0
    opencl_rows: int = 0
    opencl_detection_error: str | None = None
    dp54_rows: int = 0
    dop853_rows: int = 0
    prepare_key_builds: int = 0
    probe_memo_hits: int = 0
    probe_integration_cpu_s: float = 0.0

    @property
    def actual_cpu_s(self) -> float:
        """Measured prepare plus apply calling-thread CPU seconds."""
        return self.prepare_cpu_s + self.apply_cpu_s


def merge_family(parts: Sequence[FamilyRouting]) -> FamilyRouting:
    """Combine observed work for the same family across evolves."""
    return FamilyRouting(
        rows=sum(part.rows for part in parts),
        duration_ns=sum(part.duration_ns for part in parts),
        prepare_cpu_s=sum(part.prepare_cpu_s for part in parts),
        apply_cpu_s=sum(part.apply_cpu_s for part in parts),
        cache_hits=sum(part.cache_hits for part in parts),
        rejected_rows=sum(part.rejected_rows for part in parts),
        estimated_cost=sum(part.estimated_cost for part in parts),
        estimated_cpu_s=(
            sum(
                part.estimated_cpu_s
                for part in parts
                if part.estimated_cpu_s is not None
            )
            if all(part.estimated_cpu_s is not None for part in parts)
            else None
        ),
        prepare_builds=sum(part.prepare_builds for part in parts),
        prepare_key_builds=sum(part.prepare_key_builds for part in parts),
        admission_rows=sum(part.admission_rows for part in parts),
        probe_memo_hits=sum(part.probe_memo_hits for part in parts),
        probe_integration_cpu_s=sum(part.probe_integration_cpu_s for part in parts),
        opencl_rows=sum(part.opencl_rows for part in parts),
        dp54_rows=sum(part.dp54_rows for part in parts),
        dop853_rows=sum(part.dop853_rows for part in parts),
        opencl_detection_error=first_opencl_detection_error(parts),
    )


def reused_routing(
    segments: Mapping[str, int],
    durations: Mapping[str, float],
    previous: Mapping[str, FamilyRouting],
) -> dict[str, FamilyRouting]:
    """Report complete-result reuse without counting any kernel work.

    segments and durations give the logical row counts and ns by family.
    Each segment family must have a duration, otherwise raises KeyError.
    previous supplies any recorded device detection errors, or can be empty.
    """
    return {
        name: FamilyRouting(
            rows=count,
            cache_hits=count,
            estimated_cpu_s=0.0,
            duration_ns=durations[name],
            opencl_detection_error=first_opencl_detection_error((previous.get(name),)),
        )
        for name, count in segments.items()
    }


def warn_stark_order(inversions: Sequence[StarkOrderInversion]) -> None:
    """Report one UserWarning per reversed pair in a standalone evolve."""
    # ADR-0078: report the model limit without rejecting experiment parameters.
    by_pair: dict[tuple[int, int], list[StarkOrderInversion]] = {}
    for entry in inversions:
        by_pair.setdefault(entry.states, []).append(entry)
    for (lower, upper), entries in by_pair.items():
        first = entries[0]
        start, end = first.span_ns
        points = len({entry.point for entry in entries})
        limit = min(entry.max_shift_difference_ghz for entry in entries)
        warnings.warn(
            f"Stark shift reverses energy ordering of states {lower} and {upper} "
            f"(affected points: {points}, affected segments: {len(entries)}); first at "
            f"point {first.point} segment [{start:g}, {end:g}] ns. "
            f"Lower-minus-upper Stark shift difference upper limit: {limit} GHz. "
            "Results above this limit are outside the dispersive model's validity.",
            UserWarning,
            stacklevel=3,
        )


def first_opencl_detection_error(parts: Iterable[FamilyRouting | None]) -> str | None:
    """Return the first recorded device detection error in report order."""
    return next(
        (
            part.opencl_detection_error
            for part in parts
            if part is not None and part.opencl_detection_error
        ),
        None,
    )


@dataclass
class _Totals:
    prepare_cpu_s: float = 0.0
    apply_cpu_s: float = 0.0
    rejected_rows: int = 0
    estimated_cost: float = 0.0
    estimated_cpu_s: float = 0.0
    missing_estimate: bool = False
    prepare_builds: int = 0
    admission_rows: int = 0
    opencl_rows: int = 0
    dp54_rows: int = 0
    dop853_rows: int = 0
    prepare_key_builds: int = 0
    probe_memo_hits: int = 0
    probe_integration_cpu_s: float = 0.0


class RoutingCollector:
    """Evolve-local accounting shared by dispatch and its existing workers.

    enabled=False avoids clocks, locks and retained row identities. Measurements
    use time.thread_time in the thread executing each kernel call, not process
    CPU or wall time. snapshot is called only after workers have joined.
    """

    def __init__(self, *, enabled: bool = False) -> None:
        self.enabled = enabled
        self._lock = Lock()
        self._totals: dict[str, _Totals] = {}
        self._expected: dict[tuple[str, int, int], float] = {}
        self._tables: dict[int, object] = {}

    @contextmanager
    def measure(
        self, name: str, phase: Literal["prepare", "apply", "probe_integration"]
    ) -> Generator[None]:
        """Account a kernel call, including work completed before an exception."""
        if not self.enabled:
            yield
            return
        started = time.thread_time()
        try:
            yield
        finally:
            elapsed = time.thread_time() - started
            with self._lock:
                total = self._totals.setdefault(name, _Totals())
                if phase == "prepare":
                    total.prepare_cpu_s += elapsed
                elif phase == "apply":
                    total.apply_cpu_s += elapsed
                else:
                    total.probe_integration_cpu_s += elapsed

    def built(self, name: str, count: int) -> None:
        """Count coefficient products built by instrumented numerical owners."""
        if self.enabled and count:
            with self._lock:
                self._totals.setdefault(name, _Totals()).prepare_builds += count

    def prepared_keys(self, name: str, count: int) -> None:
        """Count prepared key slots, including repeated builds of the same key."""
        if self.enabled and count:
            with self._lock:
                self._totals.setdefault(name, _Totals()).prepare_key_builds += count

    def admitted(self, name: str, count: int) -> None:
        """Count actual numerical admission work, not apply or result hits."""
        if self.enabled and count:
            with self._lock:
                self._totals.setdefault(name, _Totals()).admission_rows += count

    def hit_probe_memo(self, name: str, count: int) -> None:
        """Count sampled guard comparisons skipped by the preparation memo."""
        if self.enabled and count:
            with self._lock:
                self._totals.setdefault(name, _Totals()).probe_memo_hits += count

    def ran_opencl(self, name: str, count: int, *, dp54: bool) -> None:
        """Record completed OpenCL endpoints for the selected tableau."""
        if self.enabled and count:
            with self._lock:
                total = self._totals.setdefault(name, _Totals())
                total.opencl_rows += count
                if dp54:
                    total.dp54_rows += count
                else:
                    total.dop853_rows += count

    def ran_cpu_dop853(self, name: str, rows: int, dp54_rows: int) -> None:
        """Count completed CPU endpoints: dp54_rows of rows used DP5(4)."""
        if self.enabled and rows:
            with self._lock:
                total = self._totals.setdefault(name, _Totals())
                total.dp54_rows += dp54_rows
                total.dop853_rows += rows - dp54_rows

    def reject(self, name: str, count: int) -> None:
        """Count actual numerical rejections, never cost-based skips."""
        if self.enabled and count:
            with self._lock:
                self._totals.setdefault(name, _Totals()).rejected_rows += count

    def expect(
        self,
        name: str,
        table: object,
        rows: NDArray[np.int64],
        seconds: NDArray[np.float64],
    ) -> None:
        """Retain existing seconds estimates before guards choose a family."""
        if self.enabled:
            with self._lock:
                self._tables[id(table)] = table
                for row, value in zip(rows, seconds, strict=True):
                    self._expected[name, id(table), int(row)] = float(value)

    def planned(
        self, table: object, names: NDArray[np.str_], costs: NDArray[np.float64]
    ) -> None:
        """Sum final selected costs once per scheduled physical segment."""
        if self.enabled:
            with self._lock:
                for name in np.unique(names[names != ""]):
                    rows = np.flatnonzero(names == name)
                    total = self._totals.setdefault(str(name), _Totals())
                    total.estimated_cost += float(np.sum(costs[rows]))
                    for row in rows:
                        value = self._expected.get((str(name), id(table), int(row)))
                        if value is None:
                            total.missing_estimate = True
                        else:
                            total.estimated_cpu_s += value

    def snapshot(
        self, segments: Mapping[str, int], durations: Mapping[str, float]
    ) -> dict[str, FamilyRouting]:
        """Return immutable family values after workers join; disabled returns {}."""
        if not self.enabled:
            return {}
        result = {}
        for name in self._totals.keys() | segments.keys():
            total = self._totals.get(name, _Totals())
            result[name] = FamilyRouting(
                rows=segments.get(name, 0),
                duration_ns=durations.get(name, 0.0),
                prepare_builds=total.prepare_builds,
                prepare_key_builds=total.prepare_key_builds,
                admission_rows=total.admission_rows,
                probe_memo_hits=total.probe_memo_hits,
                probe_integration_cpu_s=total.probe_integration_cpu_s,
                opencl_rows=total.opencl_rows,
                dp54_rows=total.dp54_rows,
                dop853_rows=total.dop853_rows,
                prepare_cpu_s=total.prepare_cpu_s,
                apply_cpu_s=total.apply_cpu_s,
                rejected_rows=total.rejected_rows,
                estimated_cost=total.estimated_cost,
                estimated_cpu_s=None
                if total.missing_estimate or not segments.get(name, 0)
                else total.estimated_cpu_s,
            )
        return result
