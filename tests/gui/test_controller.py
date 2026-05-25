"""Phase 7 tests — Controller skeleton (tab + run flow)."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest
from qtpy.QtCore import QCoreApplication
from qtpy.QtWidgets import QLabel, QStackedWidget
from zcu_tools.experiment.v2_gui.adapters.fake import FakeAdapter
from zcu_tools.experiment.v2_gui.registry import register_all
from zcu_tools.gui.adapter import CfgSchema, ExpContext
from zcu_tools.gui.controller import Controller
from zcu_tools.gui.event_bus import (
    GuiEvent,
    RunLockChangedPayload,
    TabContentChangedPayload,
)
from zcu_tools.gui.io_manager import IOManager
from zcu_tools.gui.plot_host import FigureContainer
from zcu_tools.gui.plot_routing import has_current_container
from zcu_tools.gui.registry import Registry
from zcu_tools.gui.runner import Runner
from zcu_tools.gui.services.session_persistence import SessionPersistenceService
from zcu_tools.gui.state import State

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_ctx() -> ExpContext:
    return ExpContext(
        md=MagicMock(),
        ml=MagicMock(),
        soc=MagicMock(),  # simulate connected soc
        soccfg=MagicMock(),
        res_name="fake_res",
        result_dir="/tmp/zcu_result",
        database_path="/tmp/zcu_db/fake_chip/fake_qubit",
        active_label="ctx001",
    )


def _make_view() -> MagicMock:
    view = MagicMock()
    view.show_status_message = MagicMock()
    view.make_pbar_factory = MagicMock(return_value=None)
    view.make_live_container = MagicMock(return_value=None)
    return view


class ControllerFixture:
    """Holds all objects to prevent premature GC during tests."""

    def __init__(self) -> None:
        self.state = State(_make_ctx())
        self.runner = Runner()
        self.registry = Registry()
        register_all(self.registry)
        if not self.registry.has("fake"):
            self.registry.register("fake", FakeAdapter)
        self.view = _make_view()
        io_manager = IOManager()
        io_manager._em = MagicMock()  # simulate a project being set up
        from zcu_tools.gui.event_bus import EventBus

        self.bus = EventBus()
        self.bus.emit = MagicMock()
        self.ctrl = Controller(
            state=self.state,
            runner=self.runner,
            registry=self.registry,
            io_manager=io_manager,
            view=self.view,
            bus=self.bus,
        )


@pytest.fixture()
def cf(qapp) -> ControllerFixture:  # noqa: ARG001
    return ControllerFixture()


def _default_fake_schema(ctx: ExpContext) -> CfgSchema:
    return FakeAdapter().make_default_cfg(ctx)


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


def _make_figure_container() -> FigureContainer:
    stack = QStackedWidget()
    placeholder = QLabel("(placeholder)")
    stack.addWidget(placeholder)
    return FigureContainer(stack, placeholder)


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
    cf.ctrl.start_run(tab_id, _default_fake_schema(cf.state.exp_context))
    assert cf.state.is_tab_running(tab_id)
    _wait_for(lambda: not cf.state.is_tab_running(tab_id))  # cleanup


def test_start_run_emits_run_lock_changed(cf):
    tab_id = cf.ctrl.new_tab("fake")
    cf.ctrl.start_run(tab_id, _default_fake_schema(cf.state.exp_context))
    cf.bus.emit.assert_any_call(
        GuiEvent.RUN_LOCK_CHANGED, RunLockChangedPayload(running_tab_id=tab_id)
    )
    _wait_for(lambda: not cf.state.is_tab_running(tab_id))


def test_run_finished_updates_tab_state(cf):
    tab_id = cf.ctrl.new_tab("fake")
    cf.ctrl.start_run(tab_id, _default_fake_schema(cf.state.exp_context))
    assert _wait_for(lambda: not cf.state.is_tab_running(tab_id))
    assert cf.state.get_tab(tab_id).run_result is not None


def test_run_finished_emits_run_lock_release(cf):
    tab_id = cf.ctrl.new_tab("fake")
    cf.ctrl.start_run(tab_id, _default_fake_schema(cf.state.exp_context))
    assert _wait_for(lambda: not cf.state.is_tab_running(tab_id))
    cf.bus.emit.assert_any_call(
        GuiEvent.RUN_LOCK_CHANGED, RunLockChangedPayload(running_tab_id=None)
    )


def test_run_finished_calls_refresh_tab(cf):
    tab_id = cf.ctrl.new_tab("fake")
    cf.ctrl.start_run(tab_id, _default_fake_schema(cf.state.exp_context))
    assert _wait_for(lambda: not cf.state.is_tab_running(tab_id))
    cf.bus.emit.assert_any_call(
        GuiEvent.TAB_CONTENT_CHANGED, TabContentChangedPayload(tab_id=tab_id)
    )


# ---------------------------------------------------------------------------
# run_failed flow
# ---------------------------------------------------------------------------


def test_run_failed_shows_status_message(cf):
    bad_adapter = MagicMock(spec=FakeAdapter)
    bad_adapter.run.side_effect = RuntimeError("boom")
    tab_id = cf.ctrl.new_tab("fake")
    cf.state.get_tab(tab_id).adapter = bad_adapter

    cf.ctrl.start_run(tab_id, _default_fake_schema(cf.state.exp_context))
    assert _wait_for(lambda: not cf.state.is_tab_running(tab_id))
    assert cf.view.show_error_dialog.called
    msg = cf.view.show_error_dialog.call_args[0][1]
    assert "boom" in msg


def test_run_failed_clears_run_lock(cf):
    bad_adapter = MagicMock(spec=FakeAdapter)
    bad_adapter.run.side_effect = RuntimeError("oops")
    tab_id = cf.ctrl.new_tab("fake")
    cf.state.get_tab(tab_id).adapter = bad_adapter

    cf.ctrl.start_run(tab_id, _default_fake_schema(cf.state.exp_context))
    assert _wait_for(lambda: not cf.state.is_tab_running(tab_id))
    assert cf.state.is_run_active() is False


# ---------------------------------------------------------------------------
# Duplicate start_run guard
# ---------------------------------------------------------------------------


def test_start_run_while_running_raises(cf):
    slow = MagicMock(spec=FakeAdapter)
    ev = threading.Event()
    slow.run.side_effect = lambda *a, **kw: ev.wait()

    tab_id = cf.ctrl.new_tab("fake")
    cf.state.get_tab(tab_id).adapter = slow
    cf.ctrl.start_run(tab_id, _default_fake_schema(cf.state.exp_context))

    assert cf.state.is_tab_running(tab_id)
    other_tab_id = cf.ctrl.new_tab("fake")
    with pytest.raises(RuntimeError, match="Another run is already active"):
        cf.ctrl.start_run(other_tab_id, _default_fake_schema(cf.state.exp_context))

    # cleanup
    ev.set()
    _wait_for(lambda: not cf.state.is_tab_running(tab_id), timeout_ms=2000)


def test_start_run_while_device_setup_active_raises(cf):
    tab_id = cf.ctrl.new_tab("fake")
    cf.state.begin_device_setup("flux")

    with pytest.raises(RuntimeError, match="device setup is active"):
        cf.ctrl.start_run(tab_id, _default_fake_schema(cf.state.exp_context))

    cf.state.end_device_setup("flux")


def test_run_clears_active_figure_container_after_finish(cf):
    tab_id = cf.ctrl.new_tab("fake")
    cf.view.make_live_container.return_value = _make_figure_container()

    cf.ctrl.start_run(tab_id, _default_fake_schema(cf.state.exp_context))

    assert _wait_for(lambda: not cf.state.is_tab_running(tab_id))
    assert has_current_container() is False


# ---------------------------------------------------------------------------
# View getter methods
# ---------------------------------------------------------------------------


def test_get_tab_result_returns_last_result(cf):
    import numpy as np

    tab_id = cf.ctrl.new_tab("fake")
    cf.ctrl.start_run(tab_id, _default_fake_schema(cf.state.exp_context))
    _wait_for(lambda: not cf.state.is_tab_running(tab_id))
    result = cf.ctrl.get_tab_result(tab_id)
    assert isinstance(result.data, np.ndarray)


def test_get_adapter_names_includes_fake(cf):
    assert "fake" in cf.ctrl.get_adapter_names()


def test_persist_then_restore_tabs_session(cf, tmp_path):
    tab_id = cf.ctrl.new_tab("fake")
    schema = _default_fake_schema(cf.state.exp_context)
    cf.ctrl.update_tab_cfg(tab_id, schema)
    cf.ctrl.update_tab_save_paths(tab_id, "/tmp/a.h5", "/tmp/b.png")

    session_svc = SessionPersistenceService(cache_dir=tmp_path)
    cf.ctrl._session_svc = session_svc
    cf.ctrl.persist_tabs_session()

    cf_restored = ControllerFixture()
    cf_restored.ctrl._session_svc = SessionPersistenceService(cache_dir=tmp_path)
    cf_restored.ctrl.restore_tabs_from_session()

    assert len(cf_restored.state.tabs) == 1
    restored_tab = next(iter(cf_restored.state.tabs.values()))
    assert restored_tab.adapter_name == "fake"
    save_paths = cf_restored.state.get_effective_save_paths(
        next(iter(cf_restored.state.tabs.keys()))
    )
    assert save_paths is not None
    assert save_paths.data_path == "/tmp/a.h5"
    assert save_paths.image_path == "/tmp/b.png"


# ---------------------------------------------------------------------------
# Connection / predictor — handled by ConnectionService.
# (Direct controller setters are removed in Phase 62.2.)
# ---------------------------------------------------------------------------
