"""Evolve-local publication and ownership through injected kernel collaborators."""

from contextlib import closing
from dataclasses import dataclass
from threading import Barrier, Event, Lock

import numpy as np
import pytest
from zcu_tools.simulate.evolution import Cancelled, Engine
from zcu_tools.simulate.evolution.cache import PrepareCache, PreparedPayload
from zcu_tools.simulate.evolution.kernel import KernelContext

from ._engine import Payload, RecordingKernel, batch, source


@dataclass(frozen=True)
class Bounds:
    values: np.ndarray


class ContextKernel(RecordingKernel):
    """Identity kernel that consumes evolve-local facts and group resources."""

    def __init__(self, *, outcome="success", parallel=False):
        super().__init__("t4")
        self.outcome = outcome
        self.parallel = parallel
        self.barrier = Barrier(2)
        self.lock = Lock()
        self.builds = 0
        self.values = []
        self.handles = []
        self.plans = []
        self.original = np.array([2.0, 3.0])
        self.event = Event()

    def guard(self, table, rows, context, *legacy):
        assert hasattr(context, "work_plan"), "Engine must supply KernelContext"
        from zcu_tools.simulate.evolution.workplan import FactKey

        self.plans.append(context.work_plan)
        handle = context.work_plan.resource(
            FactKey("anchors", Bounds), planned_consumers=frozenset({"early", "late"})
        )
        handle.admit("early")
        value = handle.get(lambda: Bounds(self.original))
        handle.handoff("early")
        assert not handle.released, "open resources must allow deferred admission"
        handle.admit("late")
        assert handle.get(lambda: Bounds(np.array([-1.0]))) is value
        self.handles.append(handle)
        return np.ones(len(rows), dtype=bool)

    def prepare_keys(self, table, rows, context: KernelContext):
        return table.timeline[rows].copy().view("V8")

    def apply(self, prepared, key_index, table, rows, states, context):
        from zcu_tools.simulate.evolution.workplan import FactKey

        assert context.work_plan is self.plans[0]
        if self.parallel:
            self.barrier.wait(timeout=10)

        def build():
            with self.lock:
                self.builds += 1
            return Bounds(self.original)

        value = context.work_plan.fact(FactKey("bounds", Bounds), build)
        self.values.append(value)
        if self.outcome == "failure":
            raise RuntimeError("numerical failure")
        if self.outcome == "cancel":
            self.event.set()
        return np.zeros(len(rows), dtype=bool)


def test_workers_share_one_immutable_fact_and_evolve_does_not_retain_it():
    kernel = ContextKernel(parallel=True)
    with closing(Engine(2, 0, kernels=(kernel,), chunks_per_worker=1)) as engine:
        result = engine.evolve(source(), (batch(2),), cancel=kernel.event)
    assert result.paths.segments_by_kernel == {"t4": 2}
    assert kernel.builds == 1
    assert kernel.values[0] is kernel.values[1]
    kernel.original[:] = 99
    np.testing.assert_array_equal(kernel.values[0].values, [2.0, 3.0])
    with pytest.raises(ValueError, match="read-only"):
        kernel.values[0].values[0] = 5
    with pytest.raises(RuntimeError, match="closed"):
        kernel.plans[0].fact(None, lambda: 1)


@pytest.mark.parametrize("outcome", ["success", "cancel", "failure"])
def test_evolve_releases_open_resources_on_every_exit(outcome):
    kernel = ContextKernel(outcome=outcome)
    with closing(Engine(1, 0, kernels=(kernel,))) as engine:
        if outcome == "failure":
            with pytest.raises(RuntimeError, match="numerical failure"):
                engine.evolve(source(), (batch(1),), cancel=kernel.event)
        elif outcome == "cancel":
            with pytest.raises(Cancelled):
                engine.evolve(source(), (batch(1),), cancel=kernel.event)
        else:
            engine.evolve(source(), (batch(1),), cancel=kernel.event)
    assert kernel.handles[0].released
    with pytest.raises(RuntimeError, match="released"):
        kernel.handles[0].get(lambda: Bounds(np.zeros(1)))


def test_sealed_resource_releases_after_last_actual_handoff():
    class SealingKernel(ContextKernel):
        def guard(self, table, rows, context, *legacy):
            accepted = super().guard(table, rows, context, *legacy)
            handle = self.handles[0]
            handle.seal()
            assert not handle.released
            handle.handoff("late")
            assert handle.released
            assert handle.actual_consumers == frozenset({"early", "late"})
            assert not handle.remaining_consumers
            return accepted

    kernel = SealingKernel()
    with closing(Engine(1, 0, kernels=(kernel,))) as engine:
        engine.evolve(source(), (batch(1),), cancel=kernel.event)


def test_cache_admits_singleton_whole_allocation_and_preserves_active_reference():
    cache = PrepareCache(32)
    namespace = ("synthetic", "source", 0.001)
    singleton = PreparedPayload(Payload(2), (b"one",))
    cache.add(namespace, singleton)
    active = cache.find(namespace, {b"one"})
    assert active == [singleton]
    assert cache.bytes == 32
    cache.add(namespace, PreparedPayload(Payload(3), (b"oversize",)))
    assert cache.bytes == 32
    assert cache.find(namespace, {b"one"}) == [singleton]
    cache.add(namespace, PreparedPayload(Payload(1), (b"replacement",)))
    assert cache.find(namespace, {b"one"}) == []
    assert active[0].prepared.nbytes == 32


def test_declined_cached_admission_keeps_guard_build_handoff():
    from zcu_tools.simulate.evolution.kernel import GuardPreparation, GuardResult

    class Builder(ContextKernel):
        guard_builds = True

        def guard_prepared(self, table, rows, context, *legacy):
            self.guard(table, rows, context, *legacy)
            assert self.admit_prepared(Payload(1), table, rows, context) is None
            keys, first = np.unique(
                self.prepare_keys(table, rows, context), return_index=True
            )
            prepared = self.prepare(keys, table, rows[first], context)
            return GuardResult(
                np.ones(len(rows), dtype=bool), (GuardPreparation(keys, prepared),)
            )

    kernel = Builder()
    with closing(Engine(1, 0, kernels=(kernel,))) as engine:
        result = engine.evolve(source(), (batch(1),), cancel=kernel.event)
    assert result.paths.segments_by_kernel == {"t4": 1}
    assert kernel.preparations == 1


def test_carrier_workspace_is_shared_until_close_and_isolated_between_evolves():
    import gc
    import weakref
    from concurrent.futures import ThreadPoolExecutor

    from zcu_tools.simulate.evolution.workplan import WorkPlan

    plan = WorkPlan()
    plan.publish_carrier_families(())
    assert hasattr(plan, "carrier_workspace"), "WorkPlan must own carrier reuse"
    with ThreadPoolExecutor(max_workers=2) as pool:
        workspaces = list(pool.map(lambda _: plan.carrier_workspace(), range(2)))
    assert workspaces[0] is workspaces[1]
    other = WorkPlan()
    other.publish_carrier_families(())
    assert other.carrier_workspace() is not workspaces[0]
    reference = weakref.ref(workspaces[0])
    del workspaces
    plan.close()
    gc.collect()
    assert reference() is None
    with pytest.raises(RuntimeError, match="closed"):
        plan.carrier_workspace()
    other.close()
