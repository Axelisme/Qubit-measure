"""Typed evolve-local facts, algorithm groups and resource ownership.

The planner publishes groups before workers start. Facts and resources may be
requested concurrently; builders run outside inventory locks. Engine closes the
plan only after its worker pool drains. Nothing here changes numerical routing.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable
from concurrent.futures import Future
from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass
from threading import RLock
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .kernels._carrier_reuse import CarrierWorkspace

import numpy as np


@dataclass(frozen=True)
class FactKey[T]:
    """Complete evolve-local identity and runtime type of an immutable product.

    name includes all inputs affecting the product, including table identity for
    row-indexed facts. value_type is the concrete result class, not a typing alias.
    Builders may return scalars, arrays, tuples, frozensets and frozen dataclasses
    recursively containing those types. Mutable objects raise TypeError.
    """

    name: Hashable
    value_type: type[T]


@dataclass(frozen=True)
class GainBuildPlan:
    """One gain's fixed builder and anchor dependencies, in original work units.

    gain is signed port-scaled amplitude; kind chooses the existing builder;
    work_limit is its CF4 work ceiling, not seconds; anchors are fixed gains.
    """

    gain: float
    kind: Literal["dop853", "cf4", "sambe"]
    work_limit: float
    anchors: tuple[float, ...]


@dataclass(frozen=True)
class GainGroupPlan:
    """Complete gain inventory for one physical model, port and signed carrier.

    model_identity identifies the numeric model at quantized level coordinates;
    port indexes its drives; frequency_ghz is signed cycles/ns. gains is sorted
    and unique; builders aligns with gains, independently of row admission.
    """

    model_identity: str
    level: tuple[int, ...]
    port: int
    frequency_ghz: float
    gains: tuple[float, ...]
    builders: tuple[GainBuildPlan, ...]


@dataclass(frozen=True)
class CarrierFamilyPlan:
    """Physical carrier group with fixed peak bound and subperiod resolution.

    model_identity/level select the numeric model; port indexes its drives;
    bound is the existing peak gain bound; subdivisions is its subperiod count.
    carriers_ghz is the sorted unique signed inventory. rows names planned
    candidates in the complete evolve table, not an apply batch or prefix slice.
    """

    model_identity: str
    level: tuple[int, ...]
    port: int
    bound: float
    subdivisions: int
    carriers_ghz: tuple[float, ...]
    rows: tuple[int, ...]


@dataclass(frozen=True)
class RampFamilyPlan:
    """Rise/fall consumers sharing a spectral family, before numeric admission.

    model_identity/level select the numeric model; port indexes its drives;
    frequency_ghz is signed cycles/ns; bound and subdivisions retain the existing
    amplitude/resolution rules. rows names complete-table planned consumers,
    whose durations, slopes and prepare keys may differ.
    """

    model_identity: str
    level: tuple[int, ...]
    port: int
    frequency_ghz: float
    bound: float
    subdivisions: int
    rows: tuple[int, ...]


def _freeze(value: object) -> None:
    """Validate a copied immutable value tree and freeze its owned arrays."""
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("facts cannot contain object arrays")
        value.setflags(write=False)
    elif isinstance(value, (tuple, frozenset)):
        for item in value:
            _freeze(item)
    elif is_dataclass(value) and not isinstance(value, type):
        params = getattr(type(value), "__dataclass_params__", None)
        if params is None or not params.frozen:
            raise TypeError("facts require frozen dataclasses")
        for item in fields(value):
            _freeze(getattr(value, item.name))
    elif value is not None and not isinstance(
        value, (str, bytes, int, float, complex, bool, np.generic)
    ):
        raise TypeError("unsupported mutable fact value")


class _Resource:
    def __init__(
        self, planned: frozenset[Hashable], *, retain_failure: bool = False
    ) -> None:
        self.planned = planned
        self.retain_failure = retain_failure
        self.actual: set[Hashable] = set()
        self.delivered: set[Hashable] = set()
        self.sealed = False
        self.released = False
        self.future: Future[object] | None = None
        self.lock = RLock()

    def get(self, build: Callable[[], object]) -> object:
        with self.lock:
            if self.released:
                raise RuntimeError("resource is released")
            producer = self.future is None
            if producer:
                self.future = Future()
            future = self.future
        if future is None:
            raise RuntimeError("resource publication is missing")
        if producer:
            try:
                value = deepcopy(build())
                _freeze(value)
            except BaseException as error:
                # Publish the failure to waiters, then drop this owner's refs.
                future.set_exception(error)
                if not self.retain_failure:
                    self.close()
                raise
            else:
                future.set_result(value)
        return future.result()

    def close(self) -> None:
        with self.lock:
            self.released = True
            self.future = None

    def release_if_finished(self) -> None:
        if self.sealed and self.actual == self.delivered:
            self.close()


class ResourceHandle[T]:
    """Typed view of one shared resource with incremental admission and sealing.

    planned_consumers fixes numerical coordinates. actual_consumers records
    admitted key builds; remaining_consumers excludes completed handoffs. An open
    handle retains its product for later admission, even with no remaining users.
    A sealed handle releases its product after the final handoff. released get
    raises RuntimeError; active references already handed out remain valid.
    """

    def __init__(self, state: _Resource, value_type: type[T]) -> None:
        self._state = state
        self._value_type = value_type

    @property
    def planned_consumers(self) -> frozenset[Hashable]:
        """Immutable full candidate inventory."""
        return self._state.planned

    @property
    def actual_consumers(self) -> frozenset[Hashable]:
        """All admitted consumers, including completed handoffs."""
        with self._state.lock:
            return frozenset(self._state.actual)

    @property
    def remaining_consumers(self) -> frozenset[Hashable]:
        """Admitted consumers not yet handed their result."""
        with self._state.lock:
            return frozenset(self._state.actual - self._state.delivered)

    @property
    def sealed(self) -> bool:
        """Whether admission is closed permanently."""
        with self._state.lock:
            return self._state.sealed

    @property
    def released(self) -> bool:
        """Whether this owner has dropped its numeric product."""
        with self._state.lock:
            return self._state.released

    def admit(self, consumer: Hashable) -> None:
        """Admit a planned consumer before seal; repeat admission is idempotent.

        Unplanned, already handed-off, sealed or released admissions raise
        ValueError. No numerical choices depend on actual consumer membership.
        """
        state = self._state
        with state.lock:
            if (
                state.sealed
                or state.released
                or consumer not in state.planned
                or consumer in state.delivered
            ):
                raise ValueError("consumer cannot be admitted to this resource")
            state.actual.add(consumer)

    def seal(self) -> None:
        """Close admission, releasing immediately if all actual users handed off.

        Idempotent. Deferred groups stay open until WorkPlan.close instead.
        """
        with self._state.lock:
            self._state.sealed = True
            self._state.release_if_finished()

    def get(self, build: Callable[[], T]) -> T:
        """Build once, copy and publish immutable data, or propagate its failure.

        Concurrent callers share one publication. Builders cannot recursively
        request the same key. Different resources may build concurrently.
        """

        def checked_build() -> T:
            value = build()
            if not isinstance(value, self._value_type):
                raise TypeError("resource builder returned the wrong fact type")
            return value

        value = self._state.get(checked_build)
        if not isinstance(value, self._value_type):
            raise TypeError("resource builder returned the wrong fact type")
        return value

    def handoff(self, consumer: Hashable) -> None:
        """Complete one actual consumer; unknown or duplicate IDs raise ValueError."""
        state = self._state
        with state.lock:
            if consumer not in state.actual or consumer in state.delivered:
                raise ValueError("consumer has no pending resource handoff")
            state.delivered.add(consumer)
            state.release_if_finished()


class WorkPlan:
    """One evolve's immutable publications and group/resource ownership.

    Groups start empty and each family is published once before workers start.
    fact/resource are thread-safe. close is idempotent and runs after workers
    drain on success, cancellation or failure. Acquiring after close raises
    RuntimeError. Model-only resident cache ownership is separate.
    """

    def __init__(self) -> None:
        self._facts: dict[Hashable, _Resource] = {}
        self._resources: dict[Hashable, _Resource] = {}
        self._carrier_workspace: CarrierWorkspace | None = None
        self._published: set[str] = set()
        self._gain_groups: tuple[GainGroupPlan, ...] = ()
        self._carrier_families: tuple[CarrierFamilyPlan, ...] = ()
        self._ramp_families: tuple[RampFamilyPlan, ...] = ()
        self._lock = RLock()
        self._closed = False

    @property
    def gain_groups(self) -> tuple[GainGroupPlan, ...]:
        """Caller-published gain inventories; empty until publication."""
        return self._gain_groups

    @property
    def carrier_families(self) -> tuple[CarrierFamilyPlan, ...]:
        """Caller-published physical carrier inventories."""
        return self._carrier_families

    @property
    def ramp_families(self) -> tuple[RampFamilyPlan, ...]:
        """Caller-published rise/fall family inventories."""
        return self._ramp_families

    def carrier_workspace(self) -> CarrierWorkspace:
        """Return mutable physical-block reuse shared within this evolve only.

        The first request builds carrier grids from published carrier_families
        under the inventory lock. Prefix and timeline consumers share the same
        workspace until close drops ownership after workers drain. Requests
        after close raise RuntimeError. Numerical tile builds use its own lock.
        """
        from .kernels._carrier_reuse import CarrierWorkspace

        with self._lock:
            self._require_open()
            if self._carrier_workspace is None:
                self._carrier_workspace = CarrierWorkspace(self._carrier_families)
            return self._carrier_workspace

    def fact[T](self, key: FactKey[T], build: Callable[[], T]) -> T:
        """Return this key's single immutable publication until evolve exit.

        Complete identity and supported value types are described by FactKey.
        The first builder's exception is shared with waiters and never retried.
        """
        with self._lock:
            self._require_open()
            state = self._facts.get(key)
            if state is None:
                state = _Resource(frozenset(), retain_failure=True)
                self._facts[key] = state
        return ResourceHandle(state, key.value_type).get(build)

    def resource[T](
        self, key: FactKey[T], *, planned_consumers: frozenset[Hashable]
    ) -> ResourceHandle[T]:
        """Obtain a shared resource's typed handle, before incremental admission.

        All callers must supply identical planned inventory or ValueError is
        raised. Returned views share publication, admission and release state.
        """
        with self._lock:
            self._require_open()
            state = self._resources.get(key)
            if state is None:
                state = _Resource(planned_consumers)
                self._resources[key] = state
            if state.planned != planned_consumers:
                raise ValueError("conflicting planned consumers")
        return ResourceHandle(state, key.value_type)

    def publish_gain_groups(self, groups: tuple[GainGroupPlan, ...]) -> None:
        """Publish fixed gain groups once; repeated publication raises ValueError."""
        self._publish("gain")
        self._gain_groups = groups

    def publish_carrier_families(self, groups: tuple[CarrierFamilyPlan, ...]) -> None:
        """Publish physical carrier groups once before workers start."""
        self._publish("carrier")
        self._carrier_families = groups

    def publish_ramp_families(self, groups: tuple[RampFamilyPlan, ...]) -> None:
        """Publish rise/fall spectral groups once before workers start."""
        self._publish("ramp")
        self._ramp_families = groups

    def _publish(self, name: str) -> None:
        with self._lock:
            self._require_open()
            if name in self._published:
                raise ValueError("group family is already published")
            self._published.add(name)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("work plan is closed")

    def close(self) -> None:
        """Release facts, groups and open resources after workers drain; idempotent."""
        with self._lock:
            self._closed = True
            for state in (*self._facts.values(), *self._resources.values()):
                state.close()
            self._facts.clear()
            self._resources.clear()
            self._carrier_workspace = None
            self._published.clear()
            self._gain_groups = ()
            self._carrier_families = ()
            self._ramp_families = ()
