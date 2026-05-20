"""Phase 6 tests — Runner / RunWorker (Qt, FakeAdapter)."""

from __future__ import annotations

import sys
import threading
import time
from typing import Any

import pytest
from qtpy.QtCore import QCoreApplication, QEventLoop, QTimer
from qtpy.QtWidgets import QApplication
from zcu_tools.experiment.v2_gui.adapters.fake import FakeAdapter
from zcu_tools.gui.adapter import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    ExpContext,
    ScalarSpec,
    ScalarValue,
)  # noqa: F401
from zcu_tools.gui.runner import Runner, RunWorker

# ---------------------------------------------------------------------------
# QApplication singleton — must exist before any QObject is created
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx():
    from unittest.mock import MagicMock

    ctx = MagicMock(spec=ExpContext)
    ctx.ml = MagicMock()
    ctx.ml.get_module.side_effect = lambda name, override=None: {"name": name}
    return ctx


def _simple_schema() -> CfgSchema:
    spec = CfgSectionSpec(fields={"reps": ScalarSpec(label="Reps", type=int)})
    value = CfgSectionValue(fields={"reps": ScalarValue(10)})
    return CfgSchema(spec=spec, value=value)


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
    ctx = _make_ctx()
    schema = _simple_schema()

    results = []
    worker = RunWorker(adapter, ctx, schema, {})
    worker.run_finished.connect(lambda r: results.append(r))

    worker.start()
    assert _wait_for(lambda: len(results) > 0), "run_finished not emitted in time"
    assert len(results) == 1
    import numpy as np

    assert isinstance(results[0], np.ndarray)


def test_runworker_cancel_before_start_still_finishes(qapp):
    """cancel() before start should cause adapter to finish early (FakeAdapter ignores stop)."""
    adapter = FakeAdapter()
    ctx = _make_ctx()
    schema = _simple_schema()

    finished = []
    worker = RunWorker(adapter, ctx, schema, {})
    worker.run_finished.connect(lambda r: finished.append(r))
    worker.cancel()  # set stop flag before start
    worker.start()
    assert _wait_for(lambda: len(finished) > 0), "run_finished not emitted after cancel"


def test_runworker_emits_run_failed_on_exception(qapp):
    from unittest.mock import MagicMock

    adapter = MagicMock(spec=FakeAdapter)
    adapter.run.side_effect = RuntimeError("boom")

    errors: list[Exception] = []
    worker = RunWorker(adapter, _make_ctx(), _simple_schema(), {})
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

    runner.start_run("tab1", adapter, _make_ctx(), _simple_schema(), {})
    assert _wait_for(lambda: len(finished) > 0), "runner run_finished not emitted"
    assert finished[0][0] == "tab1"


def test_runner_run_finished_clears_is_running(qapp):
    runner = Runner()
    adapter = FakeAdapter()

    done = []
    runner.run_finished.connect(lambda *_: done.append(True))

    runner.start_run("tab1", adapter, _make_ctx(), _simple_schema(), {})
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
    runner.start_run("tab1", slow_adapter, _make_ctx(), _simple_schema(), {})

    assert runner.is_running
    with pytest.raises(RuntimeError, match="already active"):
        runner.start_run("tab2", FakeAdapter(), _make_ctx(), _simple_schema(), {})

    # cleanup — unblock the slow worker
    event.set()
    runner.cancel()
    _wait_for(lambda: not runner.is_running, timeout_ms=2000)


def test_runner_cancel_stops_active_run(qapp):
    from unittest.mock import MagicMock

    slow_adapter = MagicMock(spec=FakeAdapter)
    stop_seen = []
    event = threading.Event()

    def slow_run(ctx, schema, **kw):
        from zcu_tools.experiment.v2.runner.base import _current_stop_flag

        if _current_stop_flag is not None:
            _current_stop_flag.wait(timeout=2)
            stop_seen.append(_current_stop_flag.is_set())
        event.set()

    slow_adapter.run.side_effect = slow_run

    finished = []
    runner = Runner()
    runner.run_finished.connect(lambda *_: finished.append(True))

    runner.start_run("tab1", slow_adapter, _make_ctx(), _simple_schema(), {})
    runner.cancel()
    assert _wait_for(lambda: len(finished) > 0, timeout_ms=3000)
    assert stop_seen and stop_seen[0]  # stop flag was set when run() saw it


# ---------------------------------------------------------------------------
# use_pbar_factory round-trip
# ---------------------------------------------------------------------------


def test_use_pbar_factory_restores_after_block():
    from zcu_tools.progress_bar import make_pbar
    from zcu_tools.progress_bar.interface import _pbar_factory, use_pbar_factory

    before = _pbar_factory
    sentinel = []

    def fake_factory(*a, **kw):
        sentinel.append(True)
        from zcu_tools.progress_bar.backend.tqdm import TQDMProgressBar

        return TQDMProgressBar(*a, **kw)

    with use_pbar_factory(fake_factory):
        from zcu_tools.progress_bar.interface import _pbar_factory as during

        assert during is fake_factory
        make_pbar(total=1)

    from zcu_tools.progress_bar.interface import _pbar_factory as after

    assert after is before
    assert sentinel  # factory was actually called


def test_use_pbar_factory_nested_restores_outer():
    from zcu_tools.progress_bar.backend.tqdm import TQDMProgressBar
    from zcu_tools.progress_bar.interface import _pbar_factory, use_pbar_factory

    outer_factory = lambda *a, **kw: TQDMProgressBar(*a, **kw)  # noqa: E731
    inner_factory = lambda *a, **kw: TQDMProgressBar(*a, **kw)  # noqa: E731

    with use_pbar_factory(outer_factory):
        with use_pbar_factory(inner_factory):
            from zcu_tools.progress_bar.interface import _pbar_factory as innermost

            assert innermost is inner_factory
        from zcu_tools.progress_bar.interface import _pbar_factory as mid

        assert mid is outer_factory
