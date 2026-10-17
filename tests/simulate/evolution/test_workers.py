"""Process-wide BLAS lifetime and serial fast paths through Engine.evolve."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event, current_thread

import joblib.parallel
import numpy as np
import pytest
from joblib.parallel import ThreadingBackend
from threadpoolctl import threadpool_info, threadpool_limits
from zcu_tools.simulate.evolution import Engine

from .test_engine import RecordingKernel, batch, source


def blas_widths():
    return {
        pool["prefix"]: pool["num_threads"]
        for pool in threadpool_info()
        if pool["user_api"] == "blas"
    }


@pytest.mark.parametrize("failure", [False, True])
def test_overlapping_evolves_hold_blas_one_until_last_exit(failure):
    entered = [Event(), Event()]
    release = [Event(), Event()]
    observations = []

    class ObservedKernel(RecordingKernel):
        def __init__(self, index):
            super().__init__("t4")
            self.index = index

        def guard(self, *args):
            observations.append(blas_widths())
            return super().guard(*args)

        def prepare(self, *args):
            observations.append(blas_widths())
            return super().prepare(*args)

        def apply(self, *args):
            observations.append(blas_widths())
            entered[self.index].set()
            assert release[self.index].wait(5), "test failed to release evolve"
            if failure and self.index == 0:
                raise RuntimeError("injected failure")
            return super().apply(*args)

    def evolve(index):
        engine = Engine(1, 0, kernels=(ObservedKernel(index),))
        try:
            return engine.evolve(source(), (batch(1),), cancel=Event())
        finally:
            engine.close()

    # A known non-unit caller setting proves limiting and exact restoration,
    # even when the test runner normally starts BLAS with one thread.
    with threadpool_limits(limits=2, user_api="blas"):
        original = blas_widths()
        assert original and set(original.values()) == {2}
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(evolve, 0)
            second = None
            try:
                assert entered[0].wait(5)
                second = executor.submit(evolve, 1)
                assert entered[1].wait(5)
                release[0].set()
                if failure:
                    with pytest.raises(RuntimeError, match="injected failure"):
                        first.result(timeout=5)
                else:
                    first.result(timeout=5)
                during_overlap = blas_widths()
            finally:
                for event in release:
                    event.set()
            if second is not None:
                second.result(timeout=5)
        assert observations and all(
            set(observation.values()) == {1} for observation in observations
        ), "guard, prepare and apply must all run under BLAS=1"
        assert set(during_overlap.values()) == {1}, "first exit must not restore BLAS"
        assert blas_widths() == original, "last exit must restore caller's BLAS state"


@pytest.mark.parametrize("case", ["empty", "zero-duration", "single-timeline"])
def test_serial_fast_path_does_not_open_joblib_pool(case, monkeypatch):
    configured = []
    threads = []

    class ObservedBackend(ThreadingBackend):
        def configure(self, *args, **kwargs):
            configured.append(True)
            return super().configure(*args, **kwargs)

    class ObservedKernel(RecordingKernel):
        def apply(self, *args):
            threads.append(current_thread())
            return super().apply(*args)

    batches = {
        "empty": (),
        "zero-duration": (batch(3, duration=0),),
        "single-timeline": (batch(1),),
    }[case]
    # The named backend is a neutral scheduler seam, not an engine patch.
    monkeypatch.setitem(joblib.parallel.BACKENDS, "threading", ObservedBackend)
    engine = Engine(2, 0, kernels=(ObservedKernel("t4"),))
    try:
        actual = engine.evolve(source(), batches, cancel=Event())
    finally:
        engine.close()
    assert not configured, "no-work and single-timeline evolves must not open a pool"
    assert all(thread is current_thread() for thread in threads)
    n = sum(len(item.points) for item in batches)
    np.testing.assert_array_equal(
        actual.densities, np.repeat(source().initial_density[None], n, axis=0)
    )
    if case == "single-timeline":
        engine = Engine(2, 0, kernels=(RecordingKernel("t4"),))
        try:
            pooled = engine.evolve(source(), (batch(2),), cancel=Event())
        finally:
            engine.close()
        np.testing.assert_array_equal(actual.densities[0], pooled.densities[0])
