"""A history tab exposes factory failure and retries through the normal Reset."""

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from qtpy.QtWidgets import QApplication
from zcu_tools.cfg_model import PulseSpec
from zcu_tools.gui.app.measure.ui.main_window import MainWindow
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_types import (
    CommandOrigin,
    EntryRequest,
    PointCreateRequest,
)
from zcu_tools.program.v2 import make_mock_soccfg

from tests.gui._dialog_fakes import RecordingDialogPresenter
from tests.gui._user_input import ComponentDriver
from tests.gui.app.measure._history_form_support import history_controller
from tests.gui.app.measure._load_support import write_native_file


@pytest.fixture(params=["failure", "binding", "success"])
def history_ui(
    request: pytest.FixtureRequest, qapp: QApplication, tmp_path: Path
) -> Iterator[tuple[ComponentDriver, str]]:
    path = tmp_path / "history.h5"
    write_native_file(path, tag="test/saving")
    with history_controller(tmp_path) as ctrl:
        presenter = RecordingDialogPresenter(confirm_answers=[True])
        window = MainWindow(ctrl, templates=lambda: {}, dialog_presenter=presenter)
        ctrl.add_view(window)
        ctrl.setup_control.new_entry(EntryRequest("current"))
        ctrl.setup_control.new_point(PointCreateRequest("p", "", CommandOrigin("gui")))
        ctrl.session_services.soc_connection.install_soc(
            MagicMock(), make_mock_soccfg()
        )
        ctrl.context_control.add_component("Q1", "history/source", {})
        ctrl.context_control.set_alias_rows((AliasRow("history", "Q1"),))
        window.show()
        qapp.processEvents()
        ctrl.tab_control.open_tab_from_file("history", str(path))
        key = {
            "success": "Q1.pulses.history",
            "failure": "Q1.pulses.retry",
            "binding": "Q1.pulses.template",
        }[request.param]
        ctrl.context_control.set_module(
            ctrl.read_context().target,
            key,
            PulseSpec(ch=0, nqz=1, freq=100.0, gain=0.25, waveform="Q1.waveforms.flat"),
        )
        qapp.processEvents()
        yield ComponentDriver(window), request.param
        window.deleteLater()
        qapp.processEvents()


def test_history_form_failure_and_reset_are_visible(
    history_ui: tuple[ComponentDriver, str], tmp_path: Path
) -> None:
    ui, variant = history_ui
    reason = "factory: history_source is missing"
    assert ui.observe(object_name="history.form_error").text == reason
    run = ui.observe("Run")
    assert not run.enabled and reason in run.tooltip
    assert (
        "Historical entry differs"
        in ui.observe(object_name="history.analysis_source").text
    )
    ui.screenshot(tmp_path / f"history-{variant}-before.png")
    ui.click("Reset")
    if variant == "success":
        assert ui.observe(object_name="form.input.gain").text == "0.25"
        assert ui.observe("Run").enabled
    else:
        reason = (
            "Adapter 'history', pulse: Unknown template 'missing'"
            if variant == "binding"
            else "factory: replacement source is still missing"
        )
        assert ui.observe(object_name="history.form_error").text == reason
        assert not ui.observe("Run").enabled
        assert reason in ui.observe("Run").tooltip
    ui.screenshot(tmp_path / f"history-{variant}-after.png")
