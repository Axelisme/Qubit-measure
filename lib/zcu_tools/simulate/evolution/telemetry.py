"""Named routing reports and evolve-local, thread-CPU accounting; no I/O."""

import time
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Lock
from typing import Literal

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class FamilyRouting:
    """One kernel family's observed work in a single evolve/acquire report.

    rows: logical dispatched segments, including adopted shared prefixes.
    prepare_cpu_s: summed calling-thread CPU seconds in guards, preparation,
        cached admission, extension and input binding (including rejected work).
    apply_cpu_s: summed calling-thread CPU seconds in kernel apply calls.
    cache_hits: segment consumers served by persistent prepared or exact-result
        caches. A shared prefix is measured once, not per adopting timeline.
    rejected_rows: rows actually rejected by this family's numerical guards;
        no threshold/reason detail and no cost-preference skips.
    estimated_cost: sum of relative scheduling units for executed segments.
    estimated_cpu_s: selected-plan CPU seconds, or None when any selected
        segment lacks a seconds estimate. Not inferred from scheduling units.
    prepare_builds: coefficient products built by instrumented owners (currently
        common_frame); zero for families without product instrumentation.
    admission_rows: common-frame row indicator evaluations, excluding apply and
        exact-result hits. Other families currently leave this count zero.
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

    @property
    def actual_cpu_s(self) -> float:
        """Measured prepare plus apply calling-thread CPU seconds."""
        return self.prepare_cpu_s + self.apply_cpu_s


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
        self._hits: dict[str, set[tuple[int, int]]] = {}
        self._expected: dict[tuple[str, int, int], float] = {}
        self._tables: dict[int, object] = {}

    @contextmanager
    def measure(self, name: str, phase: Literal["prepare", "apply"]) -> Generator[None]:
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
                else:
                    total.apply_cpu_s += elapsed

    def built(self, name: str, count: int) -> None:
        """Count coefficient products built by instrumented numerical owners."""
        if self.enabled and count:
            with self._lock:
                self._totals.setdefault(name, _Totals()).prepare_builds += count

    def admitted(self, name: str, count: int) -> None:
        """Count row indicator evaluations, not apply calls or result hits."""
        if self.enabled and count:
            with self._lock:
                self._totals.setdefault(name, _Totals()).admission_rows += count

    def reject(self, name: str, count: int) -> None:
        """Count actual numerical rejections, never cost-based skips."""
        if self.enabled and count:
            with self._lock:
                self._totals.setdefault(name, _Totals()).rejected_rows += count

    def hit(self, name: str, table: object, rows: NDArray[np.int64]) -> None:
        """Count persistent prepared consumers once across admission and binding."""
        if self.enabled:
            with self._lock:
                self._tables[id(table)] = table
                self._hits.setdefault(name, set()).update(
                    (id(table), int(row)) for row in rows
                )

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

    def snapshot(self, segments: Mapping[str, int]) -> dict[str, FamilyRouting]:
        """Return immutable family values after workers join; disabled returns {}."""
        if not self.enabled:
            return {}
        result = {}
        for name in self._totals.keys() | segments.keys():
            total = self._totals.get(name, _Totals())
            result[name] = FamilyRouting(
                rows=segments.get(name, 0),
                prepare_builds=total.prepare_builds,
                admission_rows=total.admission_rows,
                prepare_cpu_s=total.prepare_cpu_s,
                apply_cpu_s=total.apply_cpu_s,
                cache_hits=len(self._hits.get(name, ())),
                rejected_rows=total.rejected_rows,
                estimated_cost=total.estimated_cost,
                estimated_cpu_s=None
                if total.missing_estimate or not segments.get(name, 0)
                else total.estimated_cpu_s,
            )
        return result
