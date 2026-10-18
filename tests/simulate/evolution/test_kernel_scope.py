"""Engine scopes optional numeric workspaces without knowing kernel families."""

import gc
import weakref
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import dataclass
from threading import Barrier, Event

import pytest
from zcu_tools.simulate.evolution import Engine

from ._engine import RecordingKernel, batch, source


@dataclass
class Workspace:
    reads: list[int]

    @property
    def nbytes(self):
        self.reads.append(48)
        return 48


class ScopedKernel(RecordingKernel):
    def __init__(self, scopes, reads, *, barrier=None, fail=False):
        super().__init__("t4")
        self.scopes = scopes
        self.reads = reads
        self.barrier = barrier
        self.fail = fail
        self.workspace = None

    def evolve_scope(self, table, source):
        workspace = Workspace(self.reads)
        self.scopes.append(weakref.ref(workspace))
        scoped = ScopedKernel(
            self.scopes, self.reads, barrier=self.barrier, fail=self.fail
        )
        scoped.workspace = workspace
        return scoped, workspace

    def apply(self, prepared, key_index, table, rows, states, budget):
        if self.barrier is not None:
            self.barrier.wait(timeout=10)
        if self.fail:
            raise RuntimeError("synthetic numeric failure")
        return super().apply(prepared, key_index, table, rows, states, budget)


@pytest.mark.parametrize("fail", [False, True])
def test_engine_registers_fresh_scope_and_releases_on_success_or_failure(fail):
    scopes, reads = [], []
    kernel = ScopedKernel(scopes, reads, fail=fail)
    with closing(Engine(1, 0, kernels=[kernel])) as engine:
        for _ in range(2):
            if fail:
                with pytest.raises(RuntimeError, match="synthetic numeric failure"):
                    engine.evolve(source(), (batch(1),), cancel=Event())
            else:
                engine.evolve(source(), (batch(1),), cancel=Event())
            gc.collect()
            assert len(scopes) > 0, "Engine must invoke the optional evolve scope"
            assert all(ref() is None for ref in scopes)
    assert len(scopes) == 2
    assert reads == [48, 48], "Engine must register each live byte charge"
    assert kernel.workspace is None


def test_concurrent_evolves_do_not_share_workspace_even_with_one_kernel_registry():
    scopes, reads = [], []
    kernel = ScopedKernel(scopes, reads, barrier=Barrier(2))

    def evolve():
        with closing(Engine(1, 0, kernels=[kernel])) as engine:
            return engine.evolve(source(), (batch(1),), cancel=Event())

    with ThreadPoolExecutor(max_workers=2) as pool:
        a, b = list(pool.map(lambda _: evolve(), range(2)))
    assert len(scopes) == 2, "each concurrent evolve needs its own scope"
    assert scopes[0] is not scopes[1]
    gc.collect()
    assert all(ref() is None for ref in scopes)
    assert reads == [48, 48]
    assert a.densities.shape == b.densities.shape
