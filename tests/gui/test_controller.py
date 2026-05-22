"""Phase 7 tests — Controller skeleton (tab + run flow)."""

from __future__ import annotations

import sys
import threading
import time
from unittest.mock import MagicMock

import pytest
from qtpy.QtCore import QCoreApplication
from qtpy.QtWidgets import QApplication
from zcu_tools.experiment.v2_gui.adapters.fake import FakeAdapter
from zcu_tools.experiment.v2_gui.registry import register_all
from zcu_tools.gui.adapter import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    ExpContext,
    ScalarSpec,
    ScalarValue,
)
from zcu_tools.gui.controller import Controller
from zcu_tools.gui.device_manager import DeviceManager
from zcu_tools.gui.io_manager import IOManager
from zcu_tools.gui.registry import Registry
from zcu_tools.gui.runner import Runner
from zcu_tools.gui.state import State

# ---------------------------------------------------------------------------
# Session-scoped QApplication
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_ctx() -> ExpContext:
    return ExpContext(
        md=MagicMock(),
        ml=MagicMock(),
        soc=MagicMock(),  # simulate connected soc
        soccfg=MagicMock(),
    )


def _make_view() -> MagicMock:
    view = MagicMock()
    view.refresh_tab = MagicMock()
    view.refresh_run_state = MagicMock()
    view.refresh_context_panel = MagicMock()
    view.refresh_config_panels = MagicMock()
    view.refresh_predictor_panel = MagicMock()
    view.show_status_message = MagicMock()
    view.make_pbar_factory = MagicMock(return_value=None)
    return view


class ControllerFixture:
    """Holds all objects to prevent premature GC during tests."""

    def __init__(self) -> None:
        self.state = State(_make_ctx())
        self.runner = Runner()
        self.registry = Registry()
        register_all(self.registry)
        self.view = _make_view()
        io_manager = IOManager()
        io_manager._em = MagicMock()  # simulate a project being set up
        from zcu_tools.gui.event_bus import EventBus
        self.bus = EventBus()
        self.ctrl = Controller(
            state=self.state,
            runner=self.runner,
            registry=self.registry,
            io_manager=io_manager,
            device_manager=DeviceManager(),
            view=self.view,
            bus=self.bus,
        )


@pytest.fixture()
def cf(qapp) -> ControllerFixture:  # noqa: ARG001
    return ControllerFixture()


def _simple_schema() -> CfgSchema:
    spec = CfgSectionSpec(fields={"reps": ScalarSpec(label="Reps", type=int)})
    value = CfgSectionValue(fields={"reps": ScalarValue(10)})
    return CfgSchema(spec=spec, value=value)


def _wait_for(condition, timeout_ms: int = 3000, step_ms: int = 10) -> bool:
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
# new_tab / close_tab
# ---------------------------------------------------------------------------


def test_new_tab_adds_entry_to_state(cf):
    tab_id = cf.ctrl.new_tab("fake")
    assert tab_id in cf.state.tabs
    assert isinstance(cf.state.get_tab(tab_id).adapter, FakeAdapter)


def test_new_tab_unknown_adapter_raises(cf):
    with pytest.raises(KeyError):
        cf.ctrl.new_tab("no_such_adapter")


def test_new_tab_returns_unique_ids(cf):
    t1 = cf.ctrl.new_tab("fake")
    t2 = cf.ctrl.new_tab("fake")
    assert t1 != t2


def test_close_tab_removes_from_state(cf):
    tab_id = cf.ctrl.new_tab("fake")
    cf.ctrl.close_tab(tab_id)
    assert tab_id not in cf.state.tabs


# ---------------------------------------------------------------------------
# start_run / run_finished flow
# ---------------------------------------------------------------------------


def test_start_run_sets_is_running(cf):
    tab_id = cf.ctrl.new_tab("fake")
    cf.ctrl.start_run(tab_id, _simple_schema(), {})
    assert cf.state.is_running
    _wait_for(lambda: not cf.state.is_running)  # cleanup


def test_start_run_calls_refresh_run_state_true(cf):
    tab_id = cf.ctrl.new_tab("fake")
    cf.ctrl.start_run(tab_id, _simple_schema(), {})
    cf.view.refresh_run_state.assert_called_with(True)
    _wait_for(lambda: not cf.state.is_running)


def test_run_finished_updates_tab_state(cf):
    tab_id = cf.ctrl.new_tab("fake")
    cf.ctrl.start_run(tab_id, _simple_schema(), {})
    assert _wait_for(lambda: not cf.state.is_running)
    assert cf.state.get_tab(tab_id).last_result is not None


def test_run_finished_calls_refresh_run_state_false(cf):
    tab_id = cf.ctrl.new_tab("fake")
    cf.ctrl.start_run(tab_id, _simple_schema(), {})
    assert _wait_for(lambda: not cf.state.is_running)
    cf.view.refresh_run_state.assert_called_with(False)


def test_run_finished_calls_refresh_tab(cf):
    tab_id = cf.ctrl.new_tab("fake")
    cf.ctrl.start_run(tab_id, _simple_schema(), {})
    assert _wait_for(lambda: not cf.state.is_running)
    cf.view.refresh_tab.assert_called_with(tab_id)


# ---------------------------------------------------------------------------
# run_failed flow
# ---------------------------------------------------------------------------


def test_run_failed_shows_status_message(cf):
    bad_adapter = MagicMock(spec=FakeAdapter)
    bad_adapter.run.side_effect = RuntimeError("boom")
    tab_id = cf.ctrl.new_tab("fake")
    cf.state.get_tab(tab_id).adapter = bad_adapter

    cf.ctrl.start_run(tab_id, _simple_schema(), {})
    assert _wait_for(lambda: not cf.state.is_running)
    assert cf.view.show_status_message.called
    msg = cf.view.show_status_message.call_args[0][0]
    assert "boom" in msg


def test_run_failed_clears_is_running(cf):
    bad_adapter = MagicMock(spec=FakeAdapter)
    bad_adapter.run.side_effect = RuntimeError("oops")
    tab_id = cf.ctrl.new_tab("fake")
    cf.state.get_tab(tab_id).adapter = bad_adapter

    cf.ctrl.start_run(tab_id, _simple_schema(), {})
    assert _wait_for(lambda: not cf.state.is_running)
    assert not cf.state.is_running


# ---------------------------------------------------------------------------
# Duplicate start_run guard
# ---------------------------------------------------------------------------


def test_start_run_while_running_raises(cf):
    slow = MagicMock(spec=FakeAdapter)
    ev = threading.Event()
    slow.run.side_effect = lambda *a, **kw: ev.wait()

    tab_id = cf.ctrl.new_tab("fake")
    cf.state.get_tab(tab_id).adapter = slow
    cf.ctrl.start_run(tab_id, _simple_schema(), {})

    assert cf.state.is_running
    with pytest.raises(RuntimeError, match="already active"):
        cf.ctrl.start_run(tab_id, _simple_schema(), {})

    # cleanup
    ev.set()
    _wait_for(lambda: not cf.state.is_running, timeout_ms=2000)


# ---------------------------------------------------------------------------
# View getter methods
# ---------------------------------------------------------------------------


def test_get_tab_result_returns_last_result(cf):
    import numpy as np

    tab_id = cf.ctrl.new_tab("fake")
    cf.ctrl.start_run(tab_id, _simple_schema(), {})
    _wait_for(lambda: not cf.state.is_running)
    result = cf.ctrl.get_tab_result(tab_id)
    assert isinstance(result, np.ndarray)


def test_get_adapter_names_includes_fake(cf):
    assert "fake" in cf.ctrl.get_adapter_names()


# ---------------------------------------------------------------------------
# set_connection / set_predictor
# ---------------------------------------------------------------------------


def test_set_connection_updates_exp_context(cf):
    soc = object()
    soccfg = object()
    cf.ctrl.set_connection(soc, soccfg)
    assert cf.state.exp_context.soc is soc
    assert cf.state.exp_context.soccfg is soccfg


def test_set_predictor_updates_exp_context(cf):
    pred = object()
    cf.ctrl.set_predictor(pred)
    assert cf.state.exp_context.predictor is pred
