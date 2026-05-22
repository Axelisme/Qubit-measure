"""Phase 9 tests — Controller analyze / writeback / save flow."""

from __future__ import annotations

import sys
import time
from unittest.mock import MagicMock

import pytest
from qtpy.QtCore import QCoreApplication
from qtpy.QtWidgets import QApplication
from zcu_tools.experiment.v2_gui.adapters.fake import FakeAdapter
from zcu_tools.experiment.v2_gui.registry import register_all
from zcu_tools.gui.adapter import CfgSchema, CfgSectionSpec, CfgSectionValue, ExpContext
from zcu_tools.gui.controller import Controller
from zcu_tools.gui.device_manager import DeviceManager
from zcu_tools.gui.io_manager import IOManager
from zcu_tools.gui.registry import Registry
from zcu_tools.gui.runner import Runner
from zcu_tools.gui.state import State

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


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


def _simple_schema() -> CfgSchema:
    return CfgSchema(spec=CfgSectionSpec(), value=CfgSectionValue())


def _run_and_wait(cf: ControllerFixture, tab_id: str) -> None:
    """Start a run and block until it finishes."""
    cf.ctrl.start_run(tab_id, _simple_schema(), {})
    assert _wait_for(lambda: not cf.state.is_running)


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


def test_analyze_updates_tab_analyze_result(cf):
    tab_id = cf.ctrl.new_tab("fake")
    _run_and_wait(cf, tab_id)

    cf.ctrl.analyze(tab_id, {})

    tab = cf.state.get_tab(tab_id)
    assert tab.last_analyze_result is not None


def test_analyze_result_is_tuple_peak_fig(cf):
    tab_id = cf.ctrl.new_tab("fake")
    _run_and_wait(cf, tab_id)

    cf.ctrl.analyze(tab_id, {})

    tab = cf.state.get_tab(tab_id)
    peak, _fig = tab.last_analyze_result
    assert isinstance(peak, float)


def test_analyze_stores_figure_in_tab(cf):
    tab_id = cf.ctrl.new_tab("fake")
    _run_and_wait(cf, tab_id)

    cf.ctrl.analyze(tab_id, {})

    tab = cf.state.get_tab(tab_id)
    assert tab.last_figure is not None


def test_analyze_calls_refresh_tab(cf):
    tab_id = cf.ctrl.new_tab("fake")
    _run_and_wait(cf, tab_id)
    cf.view.refresh_tab.reset_mock()

    cf.ctrl.analyze(tab_id, {})

    cf.view.refresh_tab.assert_called_with(tab_id)


def test_analyze_without_result_raises(cf):
    tab_id = cf.ctrl.new_tab("fake")
    cf.ctrl.analyze(tab_id, {})
    cf.view.show_status_message.assert_called()


def test_analyze_exception_shows_status_message(cf):
    tab_id = cf.ctrl.new_tab("fake")
    _run_and_wait(cf, tab_id)

    bad_adapter = MagicMock(spec=FakeAdapter)
    bad_adapter.analyze.side_effect = ValueError("bad analysis")
    cf.state.get_tab(tab_id).adapter = bad_adapter

    cf.ctrl.analyze(tab_id, {})

    assert cf.view.show_status_message.called
    msg = cf.view.show_status_message.call_args[0][0]
    assert "bad analysis" in msg


def test_analyze_passes_user_params(cf):
    tab_id = cf.ctrl.new_tab("fake")
    _run_and_wait(cf, tab_id)

    spy_adapter = MagicMock(wraps=FakeAdapter())
    spy_adapter.get_figure = MagicMock(return_value=None)
    cf.state.get_tab(tab_id).adapter = spy_adapter

    cf.ctrl.analyze(tab_id, {"threshold": 0.99})

    call_kwargs = spy_adapter.analyze.call_args[1]
    assert call_kwargs.get("threshold") == 0.99


# ---------------------------------------------------------------------------
# get_tab_writeback_spec
# ---------------------------------------------------------------------------


def test_get_tab_writeback_spec_empty_before_analyze(cf):
    tab_id = cf.ctrl.new_tab("fake")
    _run_and_wait(cf, tab_id)

    assert cf.ctrl.get_tab_writeback_spec(tab_id) == []


def test_get_tab_writeback_spec_after_analyze(cf):
    tab_id = cf.ctrl.new_tab("fake")
    _run_and_wait(cf, tab_id)
    cf.ctrl.analyze(tab_id, {})

    spec = cf.ctrl.get_tab_writeback_spec(tab_id)
    assert len(spec) == 1
    assert spec[0].key == "fake_peak"


# ---------------------------------------------------------------------------
# apply_writeback
# ---------------------------------------------------------------------------


def test_apply_writeback_calls_adapter(cf):
    tab_id = cf.ctrl.new_tab("fake")
    _run_and_wait(cf, tab_id)
    cf.ctrl.analyze(tab_id, {})

    spy = MagicMock(wraps=cf.state.get_tab(tab_id).adapter)
    cf.state.get_tab(tab_id).adapter = spy

    cf.ctrl.apply_writeback(tab_id, ["fake_peak"])

    spy.apply_writeback.assert_called_once()
    call_args = spy.apply_writeback.call_args
    # positional: (ctx, analyze_result, selected_keys, overrides)
    keys = call_args[0][2]
    assert keys == ["fake_peak"]


def test_apply_writeback_calls_refresh_config_panels(cf):
    tab_id = cf.ctrl.new_tab("fake")
    _run_and_wait(cf, tab_id)
    cf.ctrl.analyze(tab_id, {})
    cf.view.refresh_config_panels.reset_mock()

    cf.ctrl.apply_writeback(tab_id, ["fake_peak"])

    cf.view.refresh_config_panels.assert_called_once()


def test_apply_writeback_without_analyze_raises(cf):
    tab_id = cf.ctrl.new_tab("fake")
    _run_and_wait(cf, tab_id)
    cf.ctrl.apply_writeback(tab_id, ["fake_peak"])
    cf.view.show_status_message.assert_called()


# ---------------------------------------------------------------------------
# save_data / save_image
# ---------------------------------------------------------------------------


def test_save_data_without_result_raises(cf):
    tab_id = cf.ctrl.new_tab("fake")
    cf.ctrl.save_data(tab_id, "/tmp/test")
    cf.view.show_status_message.assert_called()


def test_save_data_calls_adapter_save(cf):
    tab_id = cf.ctrl.new_tab("fake")
    _run_and_wait(cf, tab_id)

    spy = MagicMock(wraps=cf.state.get_tab(tab_id).adapter)
    cf.state.get_tab(tab_id).adapter = spy

    cf.ctrl.save_data(tab_id, "/tmp/fake_data")

    spy.save.assert_called_once()
    path_arg = spy.save.call_args[0][0]
    assert path_arg == "/tmp/fake_data"


def test_save_data_shows_success_message(cf):
    tab_id = cf.ctrl.new_tab("fake")
    _run_and_wait(cf, tab_id)

    cf.ctrl.save_data(tab_id, "/tmp/fake_data")

    msg = cf.view.show_status_message.call_args[0][0]
    assert "saved" in msg.lower()


def test_save_image_without_figure_raises(cf):
    tab_id = cf.ctrl.new_tab("fake")
    _run_and_wait(cf, tab_id)
    # no analyze yet → no figure
    cf.ctrl.save_image(tab_id, "/tmp/test.png")
    cf.view.show_status_message.assert_called()
    assert "No figure" in cf.view.show_status_message.call_args[0][0]


def test_save_image_calls_savefig(cf, tmp_path):
    tab_id = cf.ctrl.new_tab("fake")
    _run_and_wait(cf, tab_id)
    cf.ctrl.analyze(tab_id, {})

    out = str(tmp_path / "out.png")
    cf.ctrl.save_image(tab_id, out)

    import os

    assert os.path.exists(out)


# ---------------------------------------------------------------------------
# get_tab_analyze_params / get_tab_save_paths
# ---------------------------------------------------------------------------


def test_get_tab_analyze_params_returns_threshold(cf):
    tab_id = cf.ctrl.new_tab("fake")
    params = cf.ctrl.get_tab_analyze_params(tab_id)
    assert "threshold" in params


def test_get_tab_save_paths_returns_save_paths(cf):
    tab_id = cf.ctrl.new_tab("fake")
    paths = cf.ctrl.get_tab_save_paths(tab_id)
    assert hasattr(paths, "data_path")
    assert hasattr(paths, "image_path")
