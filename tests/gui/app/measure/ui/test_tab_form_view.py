"""The measure tab's Run panel edits its TabFormResource through FormEditor."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from unittest.mock import MagicMock

import pytest
from pydantic import JsonValue
from qtpy.QtCore import QEvent
from qtpy.QtWidgets import (
    QApplication,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QWidget,
)
from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.gui.app.measure.adapter import AdapterCapabilities, AnalysisMode, Form
from zcu_tools.gui.app.measure.services import TabSnapshot
from zcu_tools.gui.app.measure.services.ports import (
    AnalysisPaneSnapshot,
    PostAnalysisPaneSnapshot,
    RunPaneSnapshot,
    SavePaneSnapshot,
)
from zcu_tools.gui.app.measure.services.tab_form import TabFormResource
from zcu_tools.gui.app.measure.state import State, TabInteractionState
from zcu_tools.gui.app.measure.ui.exp_tab_widget import ExpTabWidget
from zcu_tools.gui.control_identity import ControlIdentity
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.gui.widgets.form import FormEditor

from tests.gui._dialog_fakes import RecordingDialogPresenter
from tests.gui.app.measure._form_fakes import (
    LookupCfg,
    configure_form_lookup,
    make_form,
)
from tests.gui.app.measure.ui._artifact_snapshots import with_artifacts

CAPABILITIES = AdapterCapabilities(analysis=AnalysisMode.NONE, post_analysis=False)


def snapshot(*, is_running: bool = False, is_analyzing: bool = False) -> TabSnapshot:
    """A render snapshot of an idle or busy tab without results."""
    return with_artifacts(
        TabSnapshot(
            adapter_name="fake",
            cfg_input={},
            tab_id="tab-1",
            interaction=TabInteractionState(
                global_run_active=is_running,
                is_running=is_running,
                is_analyzing=is_analyzing,
                has_context=True,
                has_active_context=True,
                has_soc=True,
                has_run_result=False,
                has_analyze_result=False,
                has_figure=False,
                has_post_analyze_result=False,
            ),
            capabilities=CAPABILITIES,
            run=RunPaneSnapshot(result=None, source_path=None),
            analysis=AnalysisPaneSnapshot(
                params=None, result=None, figures=None, writeback_items=()
            ),
            post_analysis=PostAnalysisPaneSnapshot(
                params=None, result=None, figures=None, writeback_items=()
            ),
            save=SavePaneSnapshot(),
        )
    )


class TabView:
    """One attached tab widget, its resource and the recorded window actions."""

    def __init__(
        self,
        presenter: RecordingDialogPresenter,
        resource: TabFormResource | None = None,
    ) -> None:
        self.ctrl = MagicMock()
        self.ctrl.control_identity = ControlIdentity()
        configure_form_lookup(self.ctrl)
        if resource is not None:
            self.ctrl.cfg_resources.lookup.side_effect = lambda _tab_id: resource
        self.ctrl.get_left_panel_width.return_value = 500
        self.ctrl.get_tab_adapter_name.return_value = "fake"
        self.ctrl.get_adapter_guide.return_value = {}
        self.ctrl.progress_control.attach_progress.return_value = lambda: None
        self.ctrl.progress_control.progress_bars.return_value = []
        self.ctrl.get_tab_snapshot.return_value = snapshot()
        self.resource: TabFormResource = self.ctrl.cfg_resources.lookup("tab-1")
        self.actions = MagicMock()
        self.tab = ExpTabWidget(
            "tab-1", self.ctrl, CAPABILITIES, dialog_presenter=presenter
        )
        self.tab.attach(snapshot(), self.actions)
        self.tab.update_interaction_state(snapshot())

    def input(self, path: str) -> QLineEdit:
        return child(self.tab, QLineEdit, f"form.input.{path}")


def child[W: QWidget](root: QWidget, kind: type[W], name: str) -> W:
    """Find a documented public FormEditor child by its complete object name."""
    widget = root.findChild(kind, name)
    assert isinstance(widget, kind), name
    return widget


@pytest.fixture
def open_tab(qapp: QApplication) -> Iterator[Callable[..., TabView]]:
    """Open attached tab views over synthetic resources; tear them down after."""
    views: list[TabView] = []

    def make(
        presenter: RecordingDialogPresenter | None = None,
        resource: TabFormResource | None = None,
    ) -> TabView:
        view = TabView(presenter or RecordingDialogPresenter(), resource)
        views.append(view)
        return view

    yield make
    for view in views:
        view.tab.detach()
        view.tab.deleteLater()
    # Delete now, not whenever garbage collection reaches the views.
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qapp.processEvents()


def test_editor_input_reaches_the_resource_and_validity_gates_run(
    open_tab: Callable[..., TabView],
) -> None:
    view = open_tab()
    published: list[dict[str, JsonValue]] = []
    view.resource.subscribe(lambda: published.append(view.resource.input.data))
    assert view.input("gain").text() == "0.1"
    assert view.tab.run_btn.isEnabled()

    view.input("gain").setText("0.45")

    assert view.resource.input.data["gain"] == 0.45
    assert view.resource.result() == LookupCfg(gain=0.45)
    assert published == [{"reps": 100, "rounds": 10, "gain": 0.45}]

    view.input("reps").setText("many")
    view.actions.refresh_interaction.assert_called_with("tab-1")
    view.tab.update_interaction_state(snapshot())

    assert view.resource.input.data["reps"] == 100
    assert child(view.tab, QLabel, "form.error.reps").text() != ""
    assert not view.tab.run_btn.isEnabled()
    assert view.tab.run_btn.toolTip().startswith("Config invalid")
    reason = view.tab.cfg_invalid_reason()
    assert reason is not None and "reps" in reason

    view.input("reps").setText("7")
    view.tab.update_interaction_state(snapshot())

    assert view.resource.result() == LookupCfg(reps=7, gain=0.45)
    assert view.tab.cfg_invalid_reason() is None
    assert view.tab.run_btn.isEnabled()


def test_busy_tab_disables_editing_and_reset_until_idle(
    open_tab: Callable[..., TabView],
) -> None:
    view = open_tab()

    view.tab.update_interaction_state(snapshot(is_running=True))

    assert not view.input("gain").isEnabled()
    assert not view.tab.reset_btn.isVisibleTo(view.tab)

    view.tab.update_interaction_state(snapshot(is_analyzing=True))

    assert not view.input("gain").isEnabled()
    assert not view.tab.reset_btn.isEnabled()

    view.tab.update_interaction_state(snapshot())

    assert view.input("gain").isEnabled()
    assert view.tab.reset_btn.isEnabled()
    view.input("gain").setText("0.3")
    assert view.resource.input.data["gain"] == 0.3


class Pulse(ConfigBase):
    """A nested section the editor shows as a foldable group."""

    gain: float = 0.1
    length: float = 1.0


class SectionCfg(ExpCfgModel):
    """Top-level scalars and one foldable section."""

    reps: int = 100
    rounds: int = 10
    pulse: Pulse = Pulse()


def test_running_tab_locks_inputs_but_keeps_the_form_readable(
    qapp: QApplication, open_tab: Callable[..., TabView]
) -> None:
    state = State(SessionEnv(parameters=None, soc=None, soccfg=None))
    view = open_tab(resource=make_form(Form(SectionCfg, {}), state=state))
    view.tab.resize(900, 160)
    view.tab.show()
    qapp.processEvents()

    view.tab.update_interaction_state(snapshot(is_running=True))
    qapp.processEvents()

    for path in ("reps", "rounds", "pulse.gain", "pulse.length"):
        assert not view.input(path).isEnabled(), path
    scroll = next(
        area
        for area in view.tab.findChildren(QScrollArea)
        if isinstance(area.widget(), FormEditor)
    )
    editor = scroll.widget()
    assert editor is not None and editor.isEnabled() and scroll.isEnabled()
    bar = scroll.verticalScrollBar()
    assert bar is not None and bar.isEnabled() and bar.maximum() > 0
    bar.setValue(bar.maximum())
    assert bar.value() == bar.maximum()

    (fold,) = [b for b in editor.findChildren(QPushButton) if b.text() == "Collapse"]
    assert fold.isEnabled()
    gain = view.input("pulse.gain")
    fold.click()
    assert gain.isHidden() or not gain.isVisibleTo(editor)
    assert fold.text() == "Expand"
    fold.click()
    assert gain.isVisibleTo(editor)
    assert view.resource.input.data == {
        "reps": 100,
        "rounds": 10,
        "pulse": {"gain": 0.1, "length": 1.0},
    }


def test_reset_shows_fresh_defaults_and_later_edits_still_reach_the_resource(
    open_tab: Callable[..., TabView],
) -> None:
    presenter = RecordingDialogPresenter(confirm_answers=[True])
    view = open_tab(presenter)
    view.input("gain").setText("0.45")
    view.input("rounds").setText("3")

    view.tab.reset_btn.click()

    assert presenter.calls[-1].kind == "confirm"
    assert view.resource.input.data == {"reps": 100, "rounds": 10, "gain": 0.1}
    assert view.input("gain").text() == "0.1"
    assert view.input("rounds").text() == "10"
    view.actions.refresh_interaction.assert_called_with("tab-1")

    view.input("gain").setText("0.2")

    assert view.resource.input.data["gain"] == 0.2
    assert view.resource.result() == LookupCfg(gain=0.2)


def test_loaded_cfg_replaces_the_displayed_input(
    open_tab: Callable[..., TabView],
) -> None:
    view = open_tab()
    view.input("gain").setText("0.45")

    assert view.resource.load_cfg(LookupCfg(reps=5, rounds=2, gain=0.7)) is None

    assert view.input("reps").text() == "5"
    assert view.input("gain").text() == "0.7"
    view.input("rounds").setText("4")
    assert view.resource.result() == LookupCfg(reps=5, rounds=4, gain=0.7)


def test_detach_stops_following_the_resource(
    qapp: QApplication, open_tab: Callable[..., TabView]
) -> None:
    view = open_tab()
    view.tab.detach()

    view.resource.write({("gain",): 0.9})

    assert view.input("gain").text() == "0.1"
    assert not view.input("gain").isEnabled()
    view.tab.attach(snapshot(), view.actions)
