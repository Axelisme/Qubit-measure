"""MainWindow Run input publication and admission contracts."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
from qtpy.QtWidgets import (
    QApplication,
    QLineEdit,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.gui.app.measure.services.tab_form import (
    AcceptedFormConfig,
    TabFormResource,
)
from zcu_tools.gui.app.measure.state import State
from zcu_tools.gui.app.measure.ui.exp_tab_widget import ExpTabWidget
from zcu_tools.gui.app.measure.ui.main_window import MainWindow
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.form_adapter import Form
from zcu_tools.gui.session.types import SessionEnv

from tests.gui._dialog_fakes import RecordingDialogPresenter
from tests.gui.app.measure._form_fakes import make_form
from tests.gui.app.measure.ui._main_window_fakes import (
    apply_window_defaults,
    make_tab_snapshot,
    mock_ctrl,
)
from tests.gui.widgets.form._text_input import commit_text


class ValueCfg(ExpCfgModel):
    """One float the Run panel edits and Run accepts."""

    value: float = 2.0


@dataclass
class RunFormFixture:
    """One shown Run panel over a caller-owned synthetic float form.

    window owns tab, whose value input is line. ctrl records commands; owner
    holds the published cfg input. runs records each accepted Run capture.
    """

    window: MainWindow
    ctrl: MagicMock
    owner: TabFormResource
    tab: ExpTabWidget
    line: QLineEdit
    runs: list[AcceptedFormConfig]


@pytest.fixture
def run_form(
    main_window_control_factory, qapp: QApplication, request: pytest.FixtureRequest
) -> Iterator[RunFormFixture]:
    ctrl = apply_window_defaults(mock_ctrl(main_window_control_factory))
    ctrl.bus = EventBus()
    snapshot = make_tab_snapshot(
        "pending-tab",
        supports_analysis=False,
        has_run_result=False,
        has_analyze_result=False,
        has_figure=False,
    )
    ctrl.get_tab_snapshot.return_value = snapshot
    sources = State(SessionEnv(parameters=None, soc=None, soccfg=None))
    owner = make_form(Form(ValueCfg, {}), state=sources)
    ctrl.cfg_resources.lookup.side_effect = lambda tab_id: owner
    runs: list[AcceptedFormConfig] = []

    def start(tab_id: str) -> None:
        runs.append(owner.accept())

    ctrl.start_run.side_effect = start
    window = MainWindow(
        ctrl, templates=lambda: {}, dialog_presenter=RecordingDialogPresenter()
    )
    window.add_tab_widget("pending-tab", "fake")
    window.resize(1100, 700)
    window.show()
    qapp.processEvents()
    tabs = window.findChildren(ExpTabWidget)
    assert len(tabs) == 1
    tab = tabs[0]
    line = tab.findChild(QLineEdit, "form.input.value")
    assert line is not None
    try:
        yield RunFormFixture(window, ctrl, owner, tab, line, runs)
    finally:
        window.remove_tab_widget("pending-tab")
        window.deleteLater()
        qapp.processEvents()


def edit_run_input(fx: RunFormFixture, text: str) -> None:
    """Replace the Value text and press Enter; the editor parses each commit."""
    fx.line.setFocus()
    commit_text(fx.line, text)


def test_run_uses_the_input_published_on_commit(
    run_form: RunFormFixture,
) -> None:
    fx = run_form
    edit_run_input(fx, "3.5")
    assert fx.tab.run_btn.isEnabled()
    fx.tab.run_btn.click()
    assert len(fx.runs) == 1
    assert fx.runs[0].values == {"value": 3.5}


@pytest.mark.parametrize("text", ["-", "nan"], ids=["incomplete", "nonfinite"])
def test_run_publishes_invalid_input_but_never_runs_old_values(
    run_form: RunFormFixture,
    text: str,
) -> None:
    fx = run_form
    edit_run_input(fx, text)
    fx.window.run_or_stop_tab("pending-tab")
    assert fx.runs == []
    assert fx.tab.cfg_invalid_reason() is not None
    assert fx.line.text() == text


def test_run_uses_the_external_update_shown_in_the_form(
    run_form: RunFormFixture,
) -> None:
    fx = run_form
    edit_run_input(fx, "3.51")
    fx.owner.write({("value",): 8.0})
    assert fx.line.text() == "8.0"
    fx.window.run_or_stop_tab("pending-tab")
    assert len(fx.runs) == 1
    assert fx.runs[0].values == {"value": 8.0}


def test_run_can_submit_a_repair_to_invalid_published_config(
    run_form: RunFormFixture,
) -> None:
    fx = run_form
    fx.owner.write({("value",): {"expr": "unfinished +"}})
    assert not fx.tab.run_btn.isEnabled()
    edit_run_input(fx, "4.5")
    assert fx.tab.run_btn.isEnabled()
    fx.tab.run_btn.click()
    assert len(fx.runs) == 1
    assert fx.runs[0].values == {"value": 4.5}


@pytest.mark.parametrize("block", ["busy", "context", "soc", "global-run"])
def test_valid_input_does_not_bypass_other_run_gates(
    run_form: RunFormFixture,
    block: str,
) -> None:
    fx = run_form
    fx.ctrl.get_tab_snapshot.return_value = make_tab_snapshot(
        "pending-tab",
        supports_analysis=False,
        has_run_result=False,
        has_analyze_result=False,
        has_figure=False,
        is_analyzing=block == "busy",
        has_active_context=block != "context",
        has_soc=block != "soc",
        global_run_active=block == "global-run",
    )
    edit_run_input(fx, "3.5")
    assert not fx.tab.run_btn.isEnabled()
    assert fx.runs == []


def test_stop_does_not_publish_or_run(run_form: RunFormFixture) -> None:
    fx = run_form
    edit_run_input(fx, "3.5")
    fx.ctrl.get_tab_snapshot.return_value = make_tab_snapshot(
        "pending-tab",
        supports_analysis=False,
        is_running=True,
    )
    fx.window.run_or_stop_tab("pending-tab")
    assert fx.runs == []
    fx.ctrl.cancel_run.assert_called_once_with()


def test_reset_replaces_published_input_with_defaults(
    run_form: RunFormFixture,
) -> None:
    fx = run_form
    dialogs = RecordingDialogPresenter(confirm_answers=[True])
    # Reopen through the public view lifecycle using the same cfg owner.
    fx.window.remove_tab_widget("pending-tab")
    fx.window.deleteLater()
    fx.window = MainWindow(fx.ctrl, templates=lambda: {}, dialog_presenter=dialogs)
    fx.window.add_tab_widget("pending-tab", "fake")
    tab = fx.window.findChildren(ExpTabWidget)[0]
    line = tab.findChild(QLineEdit, "form.input.value")
    assert line is not None
    commit_text(line, "3.5")
    assert fx.owner.accept().values == {"value": 3.5}
    tab.reset_btn.click()
    line = tab.findChild(QLineEdit, "form.input.value")
    assert line is not None and line.text() == "2.0"
    assert fx.owner.accept().values == {"value": 2.0}
