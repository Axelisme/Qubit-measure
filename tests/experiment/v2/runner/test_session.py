from typing import Any

import numpy as np
import pytest
from zcu_tools.experiment.v2.runner import MeasureSession, MeasureStep
from zcu_tools.experiment.v2.runner.state import TaskState

from .conftest import DictCfg


def _extras(cfg: DictCfg) -> dict[str, Any]:
    extra = cfg.model_extra
    assert extra is not None
    return extra


def test_single_buffer_measure_writes_final_and_partial_update():
    updates: list[np.ndarray] = []

    def measure_fn(ctx: TaskState, hook):
        hook(1, np.array([1.0]))
        return np.array([2.0])

    with MeasureSession(DictCfg(), update_interval=None) as run:
        buffer = run.buffer(
            (1,),
            dtype=np.float64,
            on_update=lambda data: updates.append(data.copy()),
        )
        buffer.measure(
            measure_fn,
            raw2signal_fn=lambda raw: raw,
            pbar_n=2,
        )

        assert np.allclose(buffer.array, [2.0])
        assert len(updates) >= 2
        assert np.allclose(updates[0], [1.0])
        assert np.allclose(updates[-1], [2.0])


def test_session_on_update_receives_full_snapshot():
    updates: list[np.ndarray] = []

    def measure_fn(ctx: TaskState, hook):
        return np.array([2.0])

    with MeasureSession(
        DictCfg(),
        on_update=lambda snap: updates.append(snap.root_data.copy()),
        update_interval=None,
    ) as run:
        buffer = run.buffer((1,), dtype=np.float64)
        buffer.measure(measure_fn, raw2signal_fn=lambda raw: raw)

    assert len(updates) == 1
    assert np.allclose(updates[0], [2.0])


def test_session_deepcopies_initial_cfg():
    cfg = DictCfg.model_validate({"payload": {"value": 1}})

    with MeasureSession(cfg) as run:
        _extras(run.cfg)["payload"]["value"] = 2

        assert run.cfg is not cfg
        assert _extras(cfg)["payload"]["value"] == 1


def test_scan_cfg_isolation_and_buffer_at_step_uses_step_cfg():
    cfg = DictCfg.model_validate({"payload": [], "marker": "root"})
    seen_payload_lengths: list[int] = []
    seen_markers: list[int] = []

    with MeasureSession(cfg) as run:
        buffer = run.buffer((2,), dtype=np.float64)

        for step in run.scan("gain", [10, 20]):
            step_extra = _extras(step.cfg)
            seen_payload_lengths.append(len(step_extra["payload"]))
            step_extra["payload"].append(step.value)
            step_extra["marker"] = step.value

            def measure_fn(ctx: TaskState, hook):
                marker = _extras(ctx.cfg)["marker"]
                seen_markers.append(marker)
                return np.asarray(marker, dtype=np.float64)

            buffer[step].measure(
                measure_fn,
                raw2signal_fn=lambda raw: raw,
            )

        assert seen_payload_lengths == [0, 0]
        assert seen_markers == [10, 20]
        assert np.allclose(buffer.array, [10.0, 20.0])
        assert _extras(run.cfg)["marker"] == "root"


def test_buffer_at_mixed_plain_index_and_step_uses_step_cfg():
    cfg = DictCfg.model_validate({"marker": "root"})
    seen_markers: list[int] = []

    with MeasureSession(cfg) as run:
        buffer = run.buffer((1, 2), dtype=np.float64)

        for rep in run.repeat("round", 1):
            for step in rep.scan("gain", [10, 20]):
                _extras(step.cfg)["marker"] = step.value

                def measure_fn(ctx: TaskState, hook):
                    marker = _extras(ctx.cfg)["marker"]
                    seen_markers.append(marker)
                    return np.asarray(marker, dtype=np.float64)

                buffer[rep.index, step].measure(
                    measure_fn,
                    raw2signal_fn=lambda raw: raw,
                )

        assert seen_markers == [10, 20]
        assert np.allclose(buffer.array, [[10.0, 20.0]])
        assert _extras(run.cfg)["marker"] == "root"


def test_scan_stop_short_circuits_later_steps():
    seen_values: list[int] = []

    with MeasureSession(DictCfg()) as run:
        for step in run.scan("x", [1, 2, 3]):
            seen_values.append(step.value)
            run.set_stop()

    assert seen_values == [1]


def test_repeat_sets_repeat_idx_in_shared_env():
    seen_idx: list[int] = []

    with MeasureSession(DictCfg()) as run:
        for rep in run.repeat("round", 3, interval=0.0):
            seen_idx.append(rep.env["repeat_idx"])

    assert seen_idx == [0, 1, 2]


def test_leaf_retry_succeeds_after_failure_with_scalar_slot():
    attempts = {"n": 0}

    def measure_fn(ctx: TaskState, hook):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("transient")
        return 7.0

    with MeasureSession(DictCfg()) as run:
        buffer = run.buffer((1,), dtype=np.float64)
        buffer[0].measure(
            measure_fn,
            raw2signal_fn=lambda raw: np.asarray(raw, dtype=np.float64),
            retry=1,
        )

        assert attempts["n"] == 2
        assert np.allclose(buffer.array, [7.0])


def test_keyboard_interrupt_stops_without_retry_and_prevents_later_scan_steps():
    attempts = {"n": 0}
    seen_steps: list[int] = []

    def measure_fn(ctx: TaskState, hook):
        attempts["n"] += 1
        raise KeyboardInterrupt

    with MeasureSession(DictCfg()) as run:
        buffer = run.buffer((3,), dtype=np.float64)
        for step in run.scan("x", [0, 1, 2]):
            assert isinstance(step.index, int)
            seen_steps.append(step.index)
            buffer[step].measure(
                measure_fn,
                raw2signal_fn=lambda raw: np.asarray(raw, dtype=np.float64),
                retry=3,
            )

        assert attempts["n"] == 1
        assert seen_steps == [0]
        assert run.is_stop()
        assert np.isnan(buffer.array[1:]).all()


def test_batch_runs_child_callables_sequentially_and_returns_results():
    order: list[str] = []

    def child_a(job: MeasureStep):
        order.append("a")
        buffer = job.buffer((1,), dtype=np.float64)
        buffer.measure(
            lambda ctx, hook: np.array([1.0]),
            raw2signal_fn=lambda raw: raw,
        )
        return buffer.array.copy()

    def child_b(job: MeasureStep):
        order.append("b")
        return np.array([2.0])

    with MeasureSession(DictCfg()) as run:
        results = run.batch({"a": child_a, "b": child_b})

        assert order == ["a", "b"]
        assert np.allclose(results["a"], [1.0])
        assert np.allclose(results["b"], [2.0])
        assert np.allclose(run.root_data["a"], [1.0])
        assert np.allclose(run.root_data["b"], [2.0])


def test_batch_return_value_triggers_update():
    updates: list[dict[str, Any]] = []

    def child(job: MeasureStep):
        return np.array([3.0])

    with MeasureSession(
        DictCfg(),
        on_update=lambda snap: updates.append(dict(snap.root_data)),
        update_interval=None,
    ) as run:
        run.batch({"a": child})

    assert len(updates) == 1
    assert np.allclose(updates[0]["a"], [3.0])


def test_retry_batch_retries_per_child():
    attempts = {"a": 0, "b": 0}

    def child_a(job: MeasureStep):
        attempts["a"] += 1
        if attempts["a"] == 1:
            raise RuntimeError("transient")
        return "a-ok"

    def child_b(job: MeasureStep):
        attempts["b"] += 1
        return "b-ok"

    with MeasureSession(DictCfg()) as run:
        results = run.batch({"a": child_a, "b": child_b}, retry=1)

        assert attempts == {"a": 2, "b": 1}
        assert results == {"a": "a-ok", "b": "b-ok"}


def test_batch_child_fast_fails_on_named_then_unnamed_buffer_mix():
    def child(job: MeasureStep):
        job.buffer((1,), dtype=np.float64, name="signal")
        job.buffer((1,), dtype=np.float64)

    with MeasureSession(DictCfg()) as run:
        with pytest.raises(ValueError, match="unnamed child buffer after named"):
            run.batch({"a": child})


def test_custom_raw2signal_applies_to_partial_and_final_raw():
    updates: list[np.ndarray] = []
    raw_seen: list[float] = []

    def raw2signal(raw):
        raw_seen.append(float(raw[0]))
        return raw * 2.0

    def measure_fn(ctx: TaskState, hook):
        hook(1, np.array([1.5]))
        return np.array([2.5])

    with MeasureSession(DictCfg(), update_interval=None) as run:
        buffer = run.buffer(
            (1,),
            dtype=np.float64,
            on_update=lambda data: updates.append(data.copy()),
        )
        buffer.measure(
            measure_fn,
            raw2signal_fn=raw2signal,
            pbar_n=2,
        )

        assert raw_seen == [1.5, 2.5]
        assert np.allclose(updates[0], [3.0])
        assert np.allclose(updates[-1], [5.0])
        assert np.allclose(buffer.array, [5.0])


def test_multiple_unnamed_buffers_fast_fail():
    with MeasureSession(DictCfg()) as run:
        run.buffer((1,), dtype=np.float64)
        with pytest.raises(ValueError, match="one unnamed root buffer"):
            run.buffer((1,), dtype=np.float64)


def test_named_root_buffers_share_mapping_root():
    with MeasureSession(DictCfg()) as run:
        i_data = run.buffer((1,), dtype=np.float64, name="i")
        q_data = run.buffer((1,), dtype=np.float64, name="q")

        assert run.root_data["i"] is i_data.array
        assert run.root_data["q"] is q_data.array
