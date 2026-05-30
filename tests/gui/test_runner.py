"""Runner / RunWorker (Qt, FakeAdapter)."""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest
from qtpy.QtCore import QCoreApplication
from zcu_tools.experiment.v2_gui.adapters.fake import FakeAdapter, FakeAnalyzeParams
from zcu_tools.gui.adapter import AnalyzeRequest, CfgSchema, ExpContext, RunRequest
from zcu_tools.gui.runner import (
    AnalyzeRunner,
    AnalyzeWorker,
    Runner,
    RunWorker,
    SaveDataRunner,
    SaveDataWorker,
)
from zcu_tools.meta_tool import MetaDict, ModuleLibrary

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx():
    from unittest.mock import MagicMock

    return ExpContext(
        md=MetaDict(), ml=ModuleLibrary(), soc=MagicMock(), soccfg=MagicMock()
    )


def _make_run_req() -> RunRequest:
    ctx = _make_ctx()
    return RunRequest(md=ctx.md, ml=ctx.ml, soc=ctx.soc, soccfg=ctx.soccfg)


def _simple_schema() -> CfgSchema:
    return FakeAdapter().make_default_cfg(_make_ctx())


def _analyze_req() -> AnalyzeRequest[Any, FakeAnalyzeParams]:
    adapter = FakeAdapter()
    ctx = _make_ctx()
    schema = adapter.make_default_cfg(ctx)
    result = adapter.run(
        RunRequest(md=ctx.md, ml=ctx.ml, soc=ctx.soc, soccfg=ctx.soccfg), schema
    )
    return AnalyzeRequest(
        run_result=result,
        analyze_params=FakeAnalyzeParams(threshold=0.0),
        md=ctx.md,
        ml=ctx.ml,
        predictor=ctx.predictor,
    )


def _wait_for(condition, timeout_ms: int = 3000, step_ms: int = 10) -> bool:
    """Spin the Qt event loop until condition() is True or timeout."""
    app = QCoreApplication.instance()
    assert app is not None
    deadline = time.monotonic() + timeout_ms / 1000
    while time.monotonic() < deadline:
        app.processEvents()
        if condition():
            return True
        time.sleep(step_ms / 1000)
    return False


# ---------------------------------------------------------------------------
# RunWorker
# ---------------------------------------------------------------------------


def test_runworker_emits_run_finished(qapp):
    adapter = FakeAdapter()
    schema = _simple_schema()

    results = []
    running_at_notification = []
    worker = RunWorker(adapter, _make_run_req(), schema)
    worker.run_finished.connect(
        lambda r: (
            results.append(r),
            running_at_notification.append(worker.isRunning()),
        )
    )

    worker.start()
    assert _wait_for(lambda: len(results) > 0), "run_finished not emitted in time"
    assert len(results) == 1
    import numpy as np

    assert isinstance(results[0].data, np.ndarray)
    assert running_at_notification == [False]


def test_runworker_cancel_before_start_still_finishes(qapp):
    """cancel() before start should cause adapter to finish early (FakeAdapter ignores stop)."""
    adapter = FakeAdapter()
    schema = _simple_schema()

    finished = []
    worker = RunWorker(adapter, _make_run_req(), schema)
    worker.run_finished.connect(lambda r: finished.append(r))
    worker.cancel()  # set stop flag before start
    worker.start()
    assert _wait_for(lambda: len(finished) > 0), "run_finished not emitted after cancel"


def test_runworker_emits_run_failed_on_exception(qapp):
    from unittest.mock import MagicMock

    adapter = MagicMock(spec=FakeAdapter)
    adapter.run.side_effect = RuntimeError("boom")

    errors: list[Exception] = []
    worker = RunWorker(adapter, _make_run_req(), _simple_schema())
    worker.run_failed.connect(lambda e: errors.append(e))
    worker.start()
    assert _wait_for(lambda: len(errors) > 0), "run_failed not emitted in time"
    assert isinstance(errors[0], RuntimeError)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def test_runner_start_run_emits_run_finished(qapp):
    runner = Runner()
    adapter = FakeAdapter()

    finished = []
    runner.run_finished.connect(lambda tid, r: finished.append((tid, r)))

    runner.start_run("tab1", adapter, _make_run_req(), _simple_schema())
    assert _wait_for(lambda: len(finished) > 0), "runner run_finished not emitted"
    assert finished[0][0] == "tab1"


def test_runner_run_finished_clears_is_running(qapp):
    runner = Runner()
    adapter = FakeAdapter()

    done = []
    runner.run_finished.connect(lambda *_: done.append(True))

    runner.start_run("tab1", adapter, _make_run_req(), _simple_schema())
    assert runner.is_running or _wait_for(lambda: len(done) > 0)
    _wait_for(lambda: len(done) > 0)
    assert not runner.is_running


def test_runner_duplicate_start_raises(qapp):
    """Starting a second run while one is active should raise RuntimeError."""
    from unittest.mock import MagicMock

    # Use a slow adapter so the first run is still running when we try the second
    slow_adapter = MagicMock(spec=FakeAdapter)
    event = threading.Event()
    slow_adapter.run.side_effect = lambda *a, **kw: event.wait()

    runner = Runner()
    runner.start_run("tab1", slow_adapter, _make_run_req(), _simple_schema())

    assert runner.is_running
    with pytest.raises(RuntimeError, match="already active"):
        runner.start_run("tab2", FakeAdapter(), _make_run_req(), _simple_schema())

    # cleanup — unblock the slow worker
    event.set()
    runner.cancel()
    _wait_for(lambda: not runner.is_running, timeout_ms=2000)


def test_runner_cancel_stops_active_run(qapp):
    from unittest.mock import MagicMock

    slow_adapter = MagicMock(spec=FakeAdapter)
    stop_seen = []
    event = threading.Event()

    def slow_run(ctx, schema):
        from zcu_tools.experiment.v2.runner.base import _current_stop_flag

        if _current_stop_flag is not None:
            _current_stop_flag.wait(timeout=2)
            stop_seen.append(_current_stop_flag.is_set())
        event.set()

    slow_adapter.run.side_effect = slow_run

    finished = []
    runner = Runner()
    runner.run_finished.connect(lambda *_: finished.append(True))

    runner.start_run("tab1", slow_adapter, _make_run_req(), _simple_schema())
    runner.cancel()
    assert _wait_for(lambda: len(finished) > 0, timeout_ms=3000)
    assert stop_seen and stop_seen[0]  # stop flag was set when run() saw it


def test_analyzeworker_emits_analyze_finished(qapp):
    adapter = FakeAdapter()
    results = []
    running_at_notification = []
    worker = AnalyzeWorker(adapter, _analyze_req())
    worker.analyze_finished.connect(
        lambda r: (
            results.append(r),
            running_at_notification.append(worker.isRunning()),
        )
    )

    worker.start()
    assert _wait_for(lambda: len(results) > 0), "analyze_finished not emitted in time"
    assert len(results) == 1
    assert results[0].figure is not None
    assert running_at_notification == [False]


def test_savedataworker_emits_save_finished(qapp):
    adapter = FakeAdapter()
    req = _analyze_req()
    from zcu_tools.gui.adapter import SaveDataRequest

    finished = []
    running_at_notification = []
    worker = SaveDataWorker(
        adapter,
        SaveDataRequest(
            run_result=req.run_result,
            data_path="/tmp/fake_data",
            md=req.md,
            ml=req.ml,
            chip_name="chip",
            qub_name="qubit",
            res_name="res",
            active_label="ctx001",
        ),
    )
    worker.save_finished.connect(
        lambda: (
            finished.append(True),
            running_at_notification.append(worker.isRunning()),
        )
    )

    worker.start()
    assert _wait_for(lambda: len(finished) > 0), "save_finished not emitted in time"
    assert running_at_notification == [False]


def test_analyze_runner_emits_finished(qapp):
    runner = AnalyzeRunner()
    adapter = FakeAdapter()
    finished = []
    runner.analyze_finished.connect(lambda tid, r: finished.append((tid, r)))

    runner.start_analyze("tab1", adapter, _analyze_req())
    assert _wait_for(lambda: len(finished) > 0), "runner analyze_finished not emitted"
    assert finished[0][0] == "tab1"


def test_save_data_runner_emits_finished(qapp):
    from zcu_tools.gui.adapter import SaveDataRequest

    runner = SaveDataRunner()
    adapter = FakeAdapter()
    analyze_req = _analyze_req()
    finished = []
    runner.save_finished.connect(lambda tid: finished.append(tid))

    runner.start_save(
        "tab1",
        adapter,
        SaveDataRequest(
            run_result=analyze_req.run_result,
            data_path="/tmp/fake_data",
            md=analyze_req.md,
            ml=analyze_req.ml,
            chip_name="chip",
            qub_name="qubit",
            res_name="res",
            active_label="ctx001",
        ),
    )
    assert _wait_for(lambda: len(finished) > 0), "runner save_finished not emitted"
    assert finished[0] == "tab1"


def test_save_data_runner_does_not_have_save_both(qapp):
    """SaveDataRunner no longer has start_save_both; save_both is handled by SaveService."""
    runner = SaveDataRunner()
    assert not hasattr(runner, "start_save_both")
    assert not hasattr(runner, "save_both_finished")


# ---------------------------------------------------------------------------
# use_pbar_factory round-trip
# ---------------------------------------------------------------------------


def test_use_pbar_factory_restores_after_block():
    from zcu_tools.progress_bar import make_pbar
    from zcu_tools.progress_bar.interface import _pbar_factory, use_pbar_factory

    before = _pbar_factory.get()
    sentinel = []

    def fake_factory(*a, **kw):
        sentinel.append(True)
        from zcu_tools.progress_bar.backend.tqdm import TQDMProgressBar

        return TQDMProgressBar(*a, **kw)

    with use_pbar_factory(fake_factory):
        assert _pbar_factory.get() is fake_factory
        make_pbar(total=1)

    assert _pbar_factory.get() is before
    assert sentinel  # factory was actually called


def test_use_pbar_factory_nested_restores_outer():
    from zcu_tools.progress_bar.backend.tqdm import TQDMProgressBar
    from zcu_tools.progress_bar.interface import _pbar_factory, use_pbar_factory

    outer_factory = lambda *a, **kw: TQDMProgressBar(*a, **kw)  # noqa: E731
    inner_factory = lambda *a, **kw: TQDMProgressBar(*a, **kw)  # noqa: E731

    with use_pbar_factory(outer_factory):
        with use_pbar_factory(inner_factory):
            assert _pbar_factory.get() is inner_factory
        assert _pbar_factory.get() is outer_factory
