import numpy as np
from zcu_tools.experiment.v2.runner.state import TaskState
from zcu_tools.experiment.v2.runner.task import Task


def _measure(state, hook):
    return np.array([1.0 + 2.0j])


def test_task_run_writes_signal_into_state():
    t = Task(
        measure_fn=_measure,
        raw2signal_fn=lambda raw: raw,
        result_shape=(1,),
        dtype=np.complex128,
        pbar_n=1,
    )
    init = t.get_default_result()
    assert init.shape == (1,) and np.isnan(init).all()

    state = TaskState(root_data=init, cfg={})
    t.init(state, dynamic_pbar=False)
    t.run(state)
    t.cleanup()

    assert init[0] == 1.0 + 2.0j


def test_task_set_pbar_n_updates_total():
    t = Task(measure_fn=_measure, result_shape=(1,), pbar_n=3)
    t.init(TaskState(root_data=t.get_default_result(), cfg={}))
    t.set_pbar_n(10)
    assert t.avg_pbar is not None
    assert t.avg_pbar.total == 10
    t.cleanup()


def test_get_default_result_nan_filled():
    t = Task(measure_fn=_measure, result_shape=(3, 2), dtype=np.complex128)
    r = t.get_default_result()
    assert r.shape == (3, 2)
    assert np.isnan(r).all()
