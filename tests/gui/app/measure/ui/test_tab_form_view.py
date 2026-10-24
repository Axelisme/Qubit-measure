"""The measure tab's Run panel edits its TabFormResource through FormEditor."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import JsonValue
from PyQt6.QtGui import QAction
from qtpy.QtCore import QEvent, Qt
from qtpy.QtWidgets import (
    QApplication,
    QLabel,
    QLineEdit,
    QScrollArea,
    QWidget,
)
from zcu_tools.cfg_model import (
    ConfigBase,
    ConstWaveformCfg,
    Expr,
    PulseCfg,
    PulseSpec,
    Range,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AdapterGuide,
    AnalysisMode,
)
from zcu_tools.gui.app.measure.services import TabSnapshot
from zcu_tools.gui.app.measure.services.ports import (
    AnalysisPaneSnapshot,
    PostAnalysisPaneSnapshot,
    RunPaneSnapshot,
    SavePaneSnapshot,
)
from zcu_tools.gui.app.measure.services.tab_form import (
    TabFormResource,
    TabFormSources,
)
from zcu_tools.gui.app.measure.state import State, TabInteractionState
from zcu_tools.gui.app.measure.ui.exp_tab_widget import ExpTabWidget
from zcu_tools.gui.app.measure.ui.main_window import MainWindow
from zcu_tools.gui.control_identity import ControlIdentity
from zcu_tools.gui.form import FormEnv
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.gui.widgets.form import FormEditor
from zcu_tools.resources.context.library import (
    ComponentLibrary,
    LibrarySnapshot,
    ModuleLibraryDocument,
)

from tests.gui._dialog_fakes import RecordingDialogPresenter
from tests.gui._qt_waits import qt_until
from tests.gui._user_input import ComponentDriver, key_click, type_text
from tests.gui.app.measure._form_fakes import (
    LookupCfg,
    configure_form_lookup,
    make_form,
)
from tests.gui.app.measure._synthetic import SampleAdapter, SampleCfg
from tests.gui.app.measure.artifacts._snapshots import with_artifacts
from tests.gui.app.measure.remote._helpers import Fixture
from tests.gui.widgets.form._field_errors import field_error
from tests.gui.widgets.form._text_input import commit_text

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
        self.ctrl.get_adapter_guide.return_value = AdapterGuide(
            markdown="# Synthetic guide\n", resource_dir=None
        )
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

    commit_text(view.input("gain"), "0.45")

    assert view.resource.input.data["gain"] == 0.45
    assert view.resource.result() == LookupCfg(gain=0.45)
    assert published == [{"reps": 100, "rounds": 10, "gain": 0.45}]

    commit_text(view.input("reps"), "many")
    view.actions.refresh_interaction.assert_called_with("tab-1")
    view.tab.update_interaction_state(snapshot())

    assert view.resource.input.data["reps"] == 100
    assert field_error(view.tab, "reps") != ""
    assert not view.tab.run_btn.isEnabled()
    assert view.tab.run_btn.toolTip().startswith("Config invalid")
    reason = view.tab.cfg_invalid_reason()
    assert reason is not None and "reps" in reason

    commit_text(view.input("reps"), "7")
    view.tab.update_interaction_state(snapshot())

    assert view.resource.result() == LookupCfg(reps=7, gain=0.45)
    assert view.tab.cfg_invalid_reason() is None
    assert view.tab.run_btn.isEnabled()


def test_busy_tab_locks_editing_and_reset_until_idle(
    open_tab: Callable[..., TabView],
) -> None:
    view = open_tab()
    gain = view.input("gain")
    original = gain.text()
    original_gain = view.resource.input.data["gain"]

    view.tab.update_interaction_state(snapshot(is_running=True))

    assert gain.isEnabled() and gain.isReadOnly()
    gain.selectAll()
    type_text(gain, "0.9")
    key_click(gain, Qt.Key.Key_Return)
    assert gain.selectedText() == original
    assert gain.text() == original
    assert view.resource.input.data["gain"] == original_gain
    assert not view.tab.reset_btn.isVisibleTo(view.tab)
    assert not view.tab.refresh_btn.isVisibleTo(view.tab)

    view.tab.update_interaction_state(snapshot(is_analyzing=True))

    assert gain.isEnabled() and gain.isReadOnly()
    assert not view.tab.reset_btn.isEnabled()
    assert not view.tab.refresh_btn.isEnabled()

    view.tab.update_interaction_state(snapshot())

    assert gain.isEnabled() and not gain.isReadOnly()
    assert view.tab.reset_btn.isEnabled()
    assert view.tab.refresh_btn.isEnabled()
    commit_text(gain, "0.3")
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
    view.tab.resize(900, 700)
    view.tab.show()
    qapp.processEvents()

    view.tab.update_interaction_state(snapshot(is_running=True))
    qapp.processEvents()

    driver = ComponentDriver(view.tab)
    driver.expand_section("form.fold.pulse")
    view.tab.resize(900, 160)
    qapp.processEvents()

    for path in ("reps", "rounds", "pulse.gain", "pulse.length"):
        field = view.input(path)
        assert field.isEnabled() and field.isReadOnly(), path
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

    view.tab.resize(900, 700)
    qapp.processEvents()
    fold = editor.findChild(QLabel, "form.fold.pulse")
    assert isinstance(fold, QLabel)
    assert fold.isEnabled()
    assert fold.text().startswith("\u25be ")
    gain = view.input("pulse.gain")
    driver.click(object_name="form.fold.pulse")
    assert gain.isHidden() or not gain.isVisibleTo(editor)
    assert fold.text().startswith("\u25b8 ")
    driver.expand_section("form.fold.pulse")
    assert gain.isVisibleTo(editor)
    assert fold.text().startswith("\u25be ")
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
    commit_text(view.input("gain"), "0.45")
    commit_text(view.input("rounds"), "3")

    view.tab.reset_btn.click()

    assert presenter.calls[-1].kind == "confirm"
    assert view.resource.input.data == {"reps": 100, "rounds": 10, "gain": 0.1}
    assert view.input("gain").text() == "0.1"
    assert view.input("rounds").text() == "10"
    view.actions.refresh_interaction.assert_called_with("tab-1")

    commit_text(view.input("gain"), "0.2")

    assert view.resource.input.data["gain"] == 0.2
    assert view.resource.result() == LookupCfg(gain=0.2)


def test_refresh_capture_failure_is_presented_and_keeps_visible_input(
    qapp: QApplication, open_tab: Callable[..., TabView]
) -> None:
    failed = False

    def capture(_aliases: object) -> TabFormSources:
        if failed:
            raise OSError("Source capture unavailable")
        return TabFormSources(FormEnv())

    resource = TabFormResource(
        lambda _ctx: Form(LookupCfg, {}),
        capture=capture,
        mutation_allowed=lambda: True,
    )
    presenter = RecordingDialogPresenter()
    view = open_tab(presenter, resource)
    view.tab.resize(1000, 750)
    view.tab.show()
    qapp.processEvents()
    driver = ComponentDriver(view.tab)
    driver.enter_text(object_name="form.input.gain", value="0.45")
    failed = True
    driver.click("Refresh")
    assert presenter.calls[-1].kind == "critical"
    assert presenter.calls[-1].message == "Source capture unavailable"
    assert driver.observe(object_name="form.input.gain").text == "0.45"
    assert driver.observe("Run").enabled


@pytest.fixture
def source_window(tmp_path, qapp: QApplication) -> Iterator[MainWindow]:
    """Open a synthetic tab, then publish a newer source before interaction."""

    class SourceAdapter(SampleAdapter):
        @staticmethod
        def make_form(ctx: FormContext) -> Form[SampleCfg]:
            del ctx
            return Form(
                SampleCfg,
                {
                    "gain": Expr("general.ext.amount"),
                    "sweep": Range(1, 2, expts=3),
                },
            )

    application = Fixture(headless=True, project_root=str(tmp_path / "project"))
    application.registry.register("source", SourceAdapter)
    ctrl = application.ctrl
    target = ctrl.context_control.read_context().target
    ctrl.context_control.set_parameter(target, "general.ext.amount", 0.2)
    window = MainWindow(ctrl, templates=lambda: {})
    ctrl.add_view(window)
    tab_id = ctrl.new_tab("source")
    ctrl.cfg_resources.lookup(tab_id).write(
        {("gain",): {"expr": "general.ext.amount * 1.5"}}
    )
    window.refresh_context_panel()
    # Setup publishes a newer source without changing the open tab's capture.
    ctrl.context_control.set_parameter(target, "general.ext.amount", 0.5)
    window.resize(1200, 850)
    window.show()
    try:
        yield window
    finally:
        application.stop()
        for tab_id in tuple(ctrl.list_tab_ids()):
            ctrl.close_tab(tab_id)
        window.deleteLater()
        QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        qapp.processEvents()


def test_tab_refresh_reset_edit_and_run_with_component_driver(
    qapp: QApplication, tmp_path: Path, source_window: MainWindow
) -> None:
    driver = ComponentDriver(source_window)
    images = tmp_path / "refresh-reset"
    driver.screenshot(images / "01-open.png")
    assert (
        driver.observe(object_name="form.input.gain").text == "general.ext.amount * 1.5"
    )
    assert driver.observe("Run").enabled
    assert any(
        item.text == "Refresh" and item.enabled for item in driver.visible().widgets
    ), "Tabs need an explicit Refresh action"
    driver.click("Refresh")
    driver.screenshot(images / "02-refreshed.png")
    assert (
        driver.observe(object_name="form.input.gain").text == "general.ext.amount * 1.5"
    )
    assert any(item.text == "= 0.75" for item in driver.visible().widgets)
    driver.enter_text(object_name="form.input.gain", value="general.ext.amount * 1.2")
    driver.screenshot(images / "03-edited.png")
    assert any(item.text == "= 0.6" for item in driver.visible().widgets)
    driver.click("Run")
    driver.select_tab("Analysis")
    qt_until(
        lambda: driver.observe("Analyze").enabled,
        label="source tab run complete",
    )
    driver.screenshot(images / "04-ran.png")
    driver.select_tab("Run")
    # Real dialog confirmation is delivered by a queued keyboard click.
    from qtpy.QtCore import QTimer
    from qtpy.QtWidgets import QMessageBox

    def confirm_reset() -> None:
        box = qapp.activeModalWidget()
        assert isinstance(box, QMessageBox)
        ComponentDriver(box).click("&Yes")

    QTimer.singleShot(0, confirm_reset)
    driver.click("Reset")
    driver.screenshot(images / "05-reset.png")
    assert driver.observe(object_name="form.input.gain").text == "0.5"
    assert driver.observe("Run").enabled


def test_detach_stops_following_the_resource(
    qapp: QApplication, open_tab: Callable[..., TabView]
) -> None:
    view = open_tab()
    view.tab.detach()

    view.resource.write({("gain",): 0.9})

    assert view.input("gain").text() == "0.1"
    assert not view.input("gain").isEnabled()
    view.tab.attach(snapshot(), view.actions)


class ModuleCore(ConfigBase):
    """One nullable pulse module the editor offers as library key or inline."""

    pulse: PulseCfg | None = None


def module_sources(_aliases: object) -> TabFormSources:
    """Capture one library pulse module, Q1.pulses.pi, at Open and Reset."""
    pulse = PulseSpec(
        ch=0, nqz=1, freq=100.0, gain=0.2, waveform=ConstWaveformCfg(length=0.1)
    )
    library = LibrarySnapshot(
        ModuleLibraryDocument(
            format="zcu.module-library",
            format_version="2.0",
            components={"Q1": ComponentLibrary(pulses={"pi": pulse})},
        )
    )
    return TabFormSources(FormEnv(library=library))


def test_option_command_runs_through_the_resource_and_locks_while_busy(
    open_tab: Callable[..., TabView],
) -> None:
    resource = TabFormResource(
        lambda _ctx: Form(ModuleCore, {}),
        capture=module_sources,
        mutation_allowed=lambda: True,
    )
    resource.write({("pulse",): {"kind": "ref", "name": "Q1.pulses.pi"}})
    view = open_tab(resource=resource)
    published: list[str] = []
    resource.subscribe(lambda: published.append("changed"))
    convert = view.tab.findChild(QAction, "form.command.pulse.to_inline")
    assert isinstance(convert, QAction)
    assert convert.text() == "Convert to inline"

    view.tab.update_interaction_state(snapshot(is_running=True))

    assert not convert.isEnabled()
    convert.trigger()
    assert resource.read_module_kind(("pulse",)) == "ref"
    assert published == []

    view.tab.update_interaction_state(snapshot())
    convert.trigger()

    # The resource publishes once, so the command used its guarded entry.
    assert published == ["changed"]
    assert resource.read_module_kind(("pulse",)) == "inline"
    assert resource.read_module(("pulse",)).value["freq"] == 100.0
