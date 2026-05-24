"""Phase 9 tests — Controller analyze / writeback / save flow."""

from __future__ import annotations

import time
from unittest.mock import ANY, MagicMock

import pytest
from qtpy.QtCore import QCoreApplication
from qtpy.QtWidgets import QLabel, QStackedWidget
from zcu_tools.experiment.v2_gui.adapters.fake import FakeAdapter
from zcu_tools.experiment.v2_gui.registry import register_all
from zcu_tools.gui.adapter import CfgSchema, CfgSectionSpec, CfgSectionValue, ExpContext
from zcu_tools.gui.controller import Controller
from zcu_tools.gui.device_manager import DeviceManager
from zcu_tools.gui.event_bus import GuiEvent, MdChangedPayload, TabContentChangedPayload
from zcu_tools.gui.io_manager import IOManager
from zcu_tools.gui.plot_host import FigureContainer
from zcu_tools.gui.plot_routing import has_current_container
from zcu_tools.gui.registry import Registry
from zcu_tools.gui.runner import Runner
from zcu_tools.gui.state import State

# ---------------------------------------------------------------------------
# Fixtures
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
        self.bus.emit = MagicMock()
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


def _make_figure_container() -> FigureContainer:
    stack = QStackedWidget()
    placeholder = QLabel("(placeholder)")
    stack.addWidget(placeholder)
    return FigureContainer(stack, placeholder)


def _default_fake_schema(ctx: ExpContext) -> CfgSchema:
    return FakeAdapter().make_default_cfg(ctx)


def _run_and_wait(cf: ControllerFixture, tab_id: str) -> None:
    """Start a run and block until it finishes."""
    cf.ctrl.start_run(tab_id, _default_fake_schema(cf.state.exp_context))
    assert _wait_for(lambda: not cf.state.is_tab_running(tab_id))


def _default_analyze_params(cf: ControllerFixture, tab_id: str) -> dict[str, object]:
    params = cf.ctrl.get_tab_analyze_params(tab_id)
    return {param.key: param.default for param in params}


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


def test_analyze_updates_tab_analyze_result(cf):
    tab_id = cf.ctrl.new_tab("fake")
    _run_and_wait(cf, tab_id)

    cf.ctrl.analyze(tab_id, _default_analyze_params(cf, tab_id))
    assert _wait_for(lambda: cf.state.get_tab(tab_id).analyze_result is not None)

    tab = cf.state.get_tab(tab_id)
    assert tab.analyze_result is not None


def test_analyze_result_has_peak_and_figure(cf):
    tab_id = cf.ctrl.new_tab("fake")
    _run_and_wait(cf, tab_id)

    cf.ctrl.analyze(tab_id, _default_analyze_params(cf, tab_id))
    assert _wait_for(lambda: cf.state.get_tab(tab_id).analyze_result is not None)

    tab = cf.state.get_tab(tab_id)
    assert isinstance(tab.analyze_result.peak, float)
    assert tab.analyze_result.figure is not None


def test_analyze_stores_figure_in_tab(cf):
    tab_id = cf.ctrl.new_tab("fake")
    _run_and_wait(cf, tab_id)

    cf.ctrl.analyze(tab_id, _default_analyze_params(cf, tab_id))
    assert _wait_for(lambda: cf.state.get_tab(tab_id).figure is not None)

    tab = cf.state.get_tab(tab_id)
    assert tab.figure is not None


def test_analyze_calls_refresh_tab(cf):
    tab_id = cf.ctrl.new_tab("fake")
    _run_and_wait(cf, tab_id)
    cf.view.refresh_tab.reset_mock()

    cf.ctrl.analyze(tab_id, _default_analyze_params(cf, tab_id))
    assert _wait_for(lambda: cf.state.get_tab(tab_id).analyze_result is not None)

    cf.bus.emit.assert_any_call(
        GuiEvent.TAB_CONTENT_CHANGED, TabContentChangedPayload(tab_id=tab_id)
    )


def test_analyze_without_result_raises(cf):
    tab_id = cf.ctrl.new_tab("fake")
    with pytest.raises(RuntimeError, match="No run result"):
        cf.ctrl.analyze(tab_id, {})


def test_analyze_exception_shows_status_message(cf):
    tab_id = cf.ctrl.new_tab("fake")
    _run_and_wait(cf, tab_id)

    bad_adapter = MagicMock(spec=FakeAdapter)
    bad_adapter.analyze.side_effect = ValueError("bad analysis")
    cf.state.get_tab(tab_id).adapter = bad_adapter

    cf.ctrl.analyze(tab_id, _default_analyze_params(cf, tab_id))
    assert _wait_for(lambda: not cf.state.is_tab_analyzing(tab_id))

    assert cf.view.show_error_dialog.called
    msg = cf.view.show_error_dialog.call_args[0][1]
    assert "bad analysis" in msg


def test_analyze_passes_analyze_params(cf):
    tab_id = cf.ctrl.new_tab("fake")
    _run_and_wait(cf, tab_id)

    spy_adapter = MagicMock(wraps=FakeAdapter())
    cf.state.get_tab(tab_id).adapter = spy_adapter

    cf.ctrl.analyze(tab_id, {"threshold": 0.99})
    assert _wait_for(lambda: spy_adapter.analyze.called)

    call_args = spy_adapter.analyze.call_args[0]
    assert call_args[0].analyze_params == {"threshold": 0.99}


# ---------------------------------------------------------------------------
# get_tab_writeback_items
# ---------------------------------------------------------------------------


def test_get_tab_writeback_items_empty_before_analyze(cf):
    tab_id = cf.ctrl.new_tab("fake")
    _run_and_wait(cf, tab_id)

    assert cf.ctrl.get_tab_writeback_items(tab_id) == []


def test_get_tab_writeback_items_after_analyze(cf):
    tab_id = cf.ctrl.new_tab("fake")
    _run_and_wait(cf, tab_id)
    cf.ctrl.analyze(tab_id, _default_analyze_params(cf, tab_id))
    assert _wait_for(lambda: cf.state.get_tab(tab_id).analyze_result is not None)

    spec = cf.ctrl.get_tab_writeback_items(tab_id)
    assert len(spec) == 1
    assert spec[0].key == "fake_peak"


# ---------------------------------------------------------------------------
# apply_writeback_items
# ---------------------------------------------------------------------------


def test_apply_writeback_items_updates_md(cf):
    tab_id = cf.ctrl.new_tab("fake")
    _run_and_wait(cf, tab_id)
    cf.ctrl.analyze(tab_id, _default_analyze_params(cf, tab_id))
    assert _wait_for(lambda: cf.state.get_tab(tab_id).analyze_result is not None)
    items = cf.ctrl.get_tab_writeback_items(tab_id)

    applied = cf.ctrl.apply_writeback_items(tab_id, items)

    assert applied == ["fake_peak"]
    assert cf.state.exp_context.md.fake_peak is not None

    items_after = cf.ctrl.get_tab_writeback_items(tab_id)
    assert len(items_after) == 1
    assert items_after[0].selected is False


def test_apply_writeback_items_emits_md_changed(cf):
    tab_id = cf.ctrl.new_tab("fake")
    _run_and_wait(cf, tab_id)
    cf.ctrl.analyze(tab_id, _default_analyze_params(cf, tab_id))
    assert _wait_for(lambda: cf.state.get_tab(tab_id).analyze_result is not None)
    items = cf.ctrl.get_tab_writeback_items(tab_id)

    cf.ctrl.apply_writeback_items(tab_id, items)

    cf.bus.emit.assert_any_call(GuiEvent.MD_CHANGED, ANY)


def test_apply_writeback_items_without_analyze_raises(cf):
    tab_id = cf.ctrl.new_tab("fake")
    _run_and_wait(cf, tab_id)
    with pytest.raises(RuntimeError, match="No analyze result"):
        cf.ctrl.apply_writeback_items(tab_id, [])


# ---------------------------------------------------------------------------
# save_data / save_image
# ---------------------------------------------------------------------------


def test_save_data_without_result_raises(cf):
    tab_id = cf.ctrl.new_tab("fake")
    with pytest.raises(RuntimeError, match="No run result"):
        cf.ctrl.save_data(tab_id, "/tmp/test")


def test_save_data_calls_adapter_save(cf):
    tab_id = cf.ctrl.new_tab("fake")
    _run_and_wait(cf, tab_id)

    spy = MagicMock(wraps=cf.state.get_tab(tab_id).adapter)
    cf.state.get_tab(tab_id).adapter = spy

    cf.ctrl.save_data(tab_id, "/tmp/fake_data")
    assert _wait_for(lambda: spy.save.called)

    spy.save.assert_called_once()
    assert spy.save.call_args[0][0].data_path == "/tmp/fake_data"


def test_save_data_shows_success_message(cf):
    tab_id = cf.ctrl.new_tab("fake")
    _run_and_wait(cf, tab_id)

    cf.ctrl.save_data(tab_id, "/tmp/fake_data")
    assert _wait_for(lambda: cf.view.show_status_message.called)

    msg = cf.view.show_status_message.call_args[0][0]
    assert "saved" in msg.lower()


def test_save_image_without_figure_raises(cf):
    tab_id = cf.ctrl.new_tab("fake")
    _run_and_wait(cf, tab_id)
    # no analyze yet → no figure
    with pytest.raises(RuntimeError, match="No figure"):
        cf.ctrl.save_image(tab_id, "/tmp/test.png")


def test_save_image_calls_savefig(cf, tmp_path):
    tab_id = cf.ctrl.new_tab("fake")
    _run_and_wait(cf, tab_id)
    cf.ctrl.analyze(tab_id, _default_analyze_params(cf, tab_id))
    assert _wait_for(lambda: cf.state.get_tab(tab_id).figure is not None)

    out = str(tmp_path / "out.png")
    cf.ctrl.save_image(tab_id, out)

    import os

    assert os.path.exists(out)


# ---------------------------------------------------------------------------
# get_tab_analyze_params / get_tab_save_paths
# ---------------------------------------------------------------------------


def test_get_tab_analyze_params_returns_threshold(cf):
    tab_id = cf.ctrl.new_tab("fake")
    _run_and_wait(cf, tab_id)
    params = cf.ctrl.get_tab_analyze_params(tab_id)
    assert len(params) == 1
    assert params[0].key == "threshold"


def test_analyze_is_async_and_sets_busy_flag(cf):
    tab_id = cf.ctrl.new_tab("fake")
    _run_and_wait(cf, tab_id)

    slow_adapter = MagicMock(wraps=FakeAdapter())

    def slow_analyze(req):
        time.sleep(0.05)
        return FakeAdapter().analyze(req)

    slow_adapter.get_analyze_params.side_effect = lambda result, ctx: (
        FakeAdapter().get_analyze_params(result, ctx)
    )
    slow_adapter.analyze.side_effect = slow_analyze
    slow_adapter.get_writeback_items.side_effect = lambda req: (
        FakeAdapter().get_writeback_items(req)
    )
    cf.state.get_tab(tab_id).adapter = slow_adapter

    cf.ctrl.analyze(tab_id, {"threshold": 0.0})
    assert cf.state.is_tab_analyzing(tab_id) is True
    assert _wait_for(lambda: not cf.state.is_tab_analyzing(tab_id))


def test_analyze_clears_active_figure_container_after_finish(cf):
    tab_id = cf.ctrl.new_tab("fake")
    _run_and_wait(cf, tab_id)
    cf.view.make_live_container.return_value = _make_figure_container()

    cf.ctrl.analyze(tab_id, _default_analyze_params(cf, tab_id))

    assert _wait_for(lambda: not cf.state.is_tab_analyzing(tab_id))
    assert has_current_container() is False


def test_save_data_is_async_and_sets_busy_flag(cf):
    tab_id = cf.ctrl.new_tab("fake")
    _run_and_wait(cf, tab_id)

    slow_adapter = MagicMock(wraps=FakeAdapter())

    def slow_save(req):
        time.sleep(0.05)
        return FakeAdapter().save(req)

    slow_adapter.save.side_effect = slow_save
    cf.state.get_tab(tab_id).adapter = slow_adapter

    cf.ctrl.save_data(tab_id, "/tmp/fake_data")
    assert cf.state.is_tab_saving_data(tab_id) is True
    assert _wait_for(lambda: not cf.state.is_tab_saving_data(tab_id))


def test_save_both_reports_mixed_result(cf):
    tab_id = cf.ctrl.new_tab("fake")
    _run_and_wait(cf, tab_id)
    cf.ctrl.analyze(tab_id, _default_analyze_params(cf, tab_id))
    assert _wait_for(lambda: cf.state.get_tab(tab_id).figure is not None)

    spy_adapter = MagicMock(wraps=cf.state.get_tab(tab_id).adapter)
    cf.state.get_tab(tab_id).adapter = spy_adapter

    cf.ctrl.save_both(tab_id, "/tmp/fake_data", "/tmp/does_not_exist/out.png")
    assert _wait_for(lambda: spy_adapter.save.called)
    assert _wait_for(
        lambda: "image failed" in cf.view.show_status_message.call_args[0][0].lower()
    )


def test_save_both_reports_data_only_failure(cf, tmp_path):
    tab_id = cf.ctrl.new_tab("fake")
    _run_and_wait(cf, tab_id)
    cf.ctrl.analyze(tab_id, _default_analyze_params(cf, tab_id))
    assert _wait_for(lambda: cf.state.get_tab(tab_id).figure is not None)

    bad_adapter = MagicMock(wraps=cf.state.get_tab(tab_id).adapter)
    bad_adapter.save.side_effect = RuntimeError("boom")
    cf.state.get_tab(tab_id).adapter = bad_adapter

    image_path = str(tmp_path / "good.png")
    cf.ctrl.save_both(tab_id, "/tmp/fake_data", image_path)
    assert _wait_for(lambda: bad_adapter.save.called)
    assert _wait_for(
        lambda: "data failed" in cf.view.show_status_message.call_args[0][0].lower()
    )
    msg = cf.view.show_status_message.call_args[0][0].lower()
    assert "image saved" in msg


def test_save_both_reports_both_success(cf, tmp_path):
    tab_id = cf.ctrl.new_tab("fake")
    _run_and_wait(cf, tab_id)
    cf.ctrl.analyze(tab_id, _default_analyze_params(cf, tab_id))
    assert _wait_for(lambda: cf.state.get_tab(tab_id).figure is not None)

    image_path = str(tmp_path / "out.png")
    cf.ctrl.save_both(tab_id, "/tmp/fake_data", image_path)

    def _both_success() -> bool:
        args = cf.view.show_status_message.call_args
        if args is None:
            return False
        msg = args[0][0].lower()
        return "data saved" in msg and "image saved" in msg

    assert _wait_for(_both_success)


def test_save_both_reports_both_failures(cf):
    tab_id = cf.ctrl.new_tab("fake")
    _run_and_wait(cf, tab_id)
    cf.ctrl.analyze(tab_id, _default_analyze_params(cf, tab_id))
    assert _wait_for(lambda: cf.state.get_tab(tab_id).figure is not None)

    bad_adapter = MagicMock(wraps=cf.state.get_tab(tab_id).adapter)
    bad_adapter.save.side_effect = RuntimeError("boom")
    cf.state.get_tab(tab_id).adapter = bad_adapter

    cf.ctrl.save_both(tab_id, "/tmp/fake_data", "/tmp/does_not_exist/out.png")
    assert _wait_for(lambda: bad_adapter.save.called)
    msg_lower = lambda: cf.view.show_error_dialog.call_args[0][1].lower()  # noqa: E731
    assert _wait_for(
        lambda: "data failed" in msg_lower() and "image failed" in msg_lower()
    )


def test_start_run_blocked_while_analyzing(cf):
    tab_id = cf.ctrl.new_tab("fake")
    _run_and_wait(cf, tab_id)

    slow_adapter = MagicMock(wraps=FakeAdapter())

    def slow_analyze(req):
        time.sleep(0.05)
        return FakeAdapter().analyze(req)

    slow_adapter.get_analyze_params.side_effect = lambda result, ctx: (
        FakeAdapter().get_analyze_params(result, ctx)
    )
    slow_adapter.analyze.side_effect = slow_analyze
    cf.state.get_tab(tab_id).adapter = slow_adapter

    cf.ctrl.analyze(tab_id, {"threshold": 0.0})
    assert cf.state.is_tab_analyzing(tab_id) is True
    with pytest.raises(RuntimeError, match="Tab .* is busy"):
        cf.ctrl.start_run(tab_id, _default_fake_schema(cf.state.exp_context))
    assert _wait_for(lambda: not cf.state.is_tab_analyzing(tab_id))


def test_analyze_allowed_while_other_tab_saving_data(cf):
    tab_id = cf.ctrl.new_tab("fake")
    other_tab_id = cf.ctrl.new_tab("fake")
    _run_and_wait(cf, tab_id)
    _run_and_wait(cf, other_tab_id)

    slow_adapter = MagicMock(wraps=FakeAdapter())

    def slow_save(req):
        time.sleep(0.05)
        return FakeAdapter().save(req)

    slow_adapter.save.side_effect = slow_save
    cf.state.get_tab(other_tab_id).adapter = slow_adapter

    cf.ctrl.save_data(other_tab_id, "/tmp/fake_data")
    assert cf.state.is_tab_saving_data(other_tab_id) is True
    cf.ctrl.analyze(tab_id, {"threshold": 0.0})
    assert _wait_for(lambda: cf.state.get_tab(tab_id).analyze_result is not None)
    assert _wait_for(lambda: not cf.state.is_tab_saving_data(other_tab_id))


def test_close_busy_tab_shows_status_message(cf):
    tab_id = cf.ctrl.new_tab("fake")
    _run_and_wait(cf, tab_id)

    slow_adapter = MagicMock(wraps=FakeAdapter())

    def slow_analyze(req):
        time.sleep(0.05)
        return FakeAdapter().analyze(req)

    slow_adapter.get_analyze_params.side_effect = lambda result, ctx: (
        FakeAdapter().get_analyze_params(result, ctx)
    )
    slow_adapter.analyze.side_effect = slow_analyze
    cf.state.get_tab(tab_id).adapter = slow_adapter

    cf.ctrl.analyze(tab_id, {"threshold": 0.0})
    assert cf.state.is_tab_analyzing(tab_id) is True
    with pytest.raises(RuntimeError, match="busy tab"):
        cf.ctrl.close_tab(tab_id)
    assert tab_id in cf.state.tabs
    assert _wait_for(lambda: not cf.state.is_tab_analyzing(tab_id))


def test_unknown_tab_status_queries_raise_key_error(cf):
    with pytest.raises(KeyError):
        cf.ctrl.is_tab_running("missing")
    with pytest.raises(KeyError):
        cf.ctrl.is_tab_analyzing("missing")
    with pytest.raises(KeyError):
        cf.ctrl.is_tab_saving_data("missing")
    with pytest.raises(KeyError):
        cf.ctrl.is_tab_busy("missing")


def test_close_unknown_tab_raises_key_error(cf):
    with pytest.raises(KeyError):
        cf.ctrl.close_tab("missing")


def test_get_tab_save_paths_returns_save_paths(cf):
    tab_id = cf.ctrl.new_tab("fake")
    paths = cf.ctrl.get_tab_save_paths(tab_id)
    assert hasattr(paths, "data_path")
    assert hasattr(paths, "image_path")


def test_get_tab_save_paths_without_active_context_returns_none(cf):
    cf.state.exp_context = ExpContext(
        md=MagicMock(),
        ml=MagicMock(),
        soc=MagicMock(),
        soccfg=MagicMock(),
    )
    tab_id = cf.ctrl.new_tab("fake")

    assert cf.ctrl.get_tab_save_paths(tab_id) is None
