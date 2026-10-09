"""Tab role selectors, committed facts and historical identity through public Qt seams."""

from collections.abc import Iterator
from dataclasses import replace
from unittest.mock import MagicMock

import pytest
from qtpy.QtWidgets import QComboBox, QLabel, QPushButton, QTabWidget, QWidget
from zcu_tools.gui.app.measure.adapter import AdapterCapabilities, AnalysisMode
from zcu_tools.gui.app.measure.services.ports import (
    RunPaneSnapshot,
    SavePaneSnapshot,
    TabSnapshot,
)
from zcu_tools.gui.app.measure.state import TabInteractionState
from zcu_tools.gui.app.measure.ui.exp_tab_widget import ExpTabWidget
from zcu_tools.gui.app.measure.ui.main_window import MainWindow
from zcu_tools.gui.cfg import CfgSchema, CfgSectionSpec, CfgSectionValue
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.session.context_types import ComponentFocus, EntryDraft
from zcu_tools.gui.session.parameters import ParameterContext, RoleBinding
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.resources.context.library import LibrarySnapshot, ModuleLibraryDocument
from zcu_tools.resources.entry import ComponentSchema, RoleSpec, component_registry
from zcu_tools.resources.entry.schema import PointDocument

from tests.gui._dialog_fakes import RecordingDialogPresenter
from tests.gui.app.measure._cfg_fakes import configure_cfg_lookup


def _mock_ctrl() -> MagicMock:
    ctrl = MagicMock()
    configure_cfg_lookup(ctrl)
    ctrl.get_left_panel_width.return_value = 500
    return ctrl


@pytest.fixture
def focus_captures() -> Iterator[tuple[ParameterContext, ParameterContext]]:
    before = component_registry.kinds()
    component_registry.register("ui-focus/driver", ComponentSchema)
    component_registry.register("ui-focus/sensor", ComponentSchema)
    try:
        point = PointDocument.model_validate(
            {
                "format": "zcu.parameter-container",
                "format_version": "1.0",
                "general": {"created_at": "2026-10-08T00:00:00Z"},
                "components": {
                    "Q1": {"kind": "ui-focus/driver"},
                    "Q2": {"kind": "ui-focus/driver"},
                    "R1": {"kind": "ui-focus/sensor"},
                    "R2": {"kind": "ui-focus/sensor"},
                },
            }
        )
        library = LibrarySnapshot(
            ModuleLibraryDocument(format="zcu.module-library", format_version="1.0")
        )
        declarations = {
            "qubit": RoleSpec("ui-focus/driver"),
            "resonator": RoleSpec("ui-focus/sensor"),
        }
        captures = tuple(
            ParameterContext(
                point=point,
                library=library,
                target=EntryDraft(entry_id="entry"),
                roles=RoleBinding(
                    qubit, declarations, {"qubit": qubit, "resonator": resonator}
                ),
            )
            for qubit, resonator in (("Q1", "R1"), ("Q2", "R2"))
        )
        yield captures[0], captures[1]
    finally:
        component_registry.unregister("ui-focus/driver")
        component_registry.unregister("ui-focus/sensor")
        assert component_registry.kinds() == before


def _snapshot(tab_id: str) -> TabSnapshot:
    """Build an idle role-only render model without acquisition or save artifacts."""
    return TabSnapshot(
        adapter_name="fake",
        cfg_schema=CfgSchema(CfgSectionSpec(), CfgSectionValue()),
        tab_id=tab_id,
        capabilities=AdapterCapabilities(analysis=AnalysisMode.NONE),
        interaction=TabInteractionState(
            global_run_active=False,
            is_running=False,
            is_analyzing=False,
            has_context=True,
            has_active_context=True,
            has_soc=False,
            has_run_result=False,
            has_analyze_result=False,
            has_figure=False,
        ),
        run=RunPaneSnapshot(result=None, source_path=None),
        save=SavePaneSnapshot(),
    )


def _apply_window_defaults(ctrl: MagicMock) -> MagicMock:
    """Provide idle session reads for the top-level Qt host."""
    ctrl.active_operation_count.return_value = 0
    ctrl.get_active_context_label.return_value = None
    ctrl.has_draft_context.return_value = False
    ctrl.has_project.return_value = False
    ctrl.context_control.get_component_focus.return_value = ComponentFocus(
        None, None, ()
    )
    ctrl.get_session_env.return_value = SessionEnv(
        parameters=None, soc=None, soccfg=None
    )
    return ctrl


def test_role_rebind_ui_renders_capture_choices_and_emits_only_user_intent(
    qapp,
    focus_captures: tuple[ParameterContext, ParameterContext],
) -> None:
    ctrl = _mock_ctrl()
    q1, _q2 = focus_captures
    snapshot = replace(_snapshot("a"), parameters=q1)
    assert snapshot.capabilities is not None
    widget = ExpTabWidget("a", ctrl, snapshot.capabilities)
    widget.attach(snapshot, MagicMock())
    driver = widget.findChild(QComboBox, "role-qubit")
    sensor = widget.findChild(QComboBox, "role-resonator")
    reset = widget.findChild(QPushButton, "role-reset-qubit")
    assert driver is not None and sensor is not None and reset is not None
    assert {driver.itemData(index) for index in range(driver.count())} == {
        None,
        "Q1",
        "Q2",
    }
    assert {sensor.itemData(index) for index in range(sensor.count())} == {
        None,
        "R1",
        "R2",
    }
    assert driver.currentData() == "Q1" and sensor.currentData() == "R1"
    assert "Q1" in reset.text()
    ctrl.tab_control.set_role.assert_not_called()
    ctrl.tab_control.reset_role.assert_not_called()
    index = driver.findData("Q2")
    driver.setCurrentIndex(index)
    ctrl.tab_control.set_role.assert_not_called()
    driver.activated.emit(index)
    ctrl.tab_control.set_role.assert_called_once_with("a", "qubit", "Q2")
    ctrl.tab_control.set_role.reset_mock()
    clear = driver.findData(None)
    driver.setCurrentIndex(clear)
    driver.activated.emit(clear)
    ctrl.tab_control.set_role.assert_called_once_with("a", "qubit", None)
    reset.click()
    ctrl.tab_control.reset_role.assert_called_once_with("a", "qubit")
    widget.detach()


@pytest.mark.parametrize("busy", ["running", "analyzing"])
def test_role_rebind_ui_manual_clear_and_local_busy_are_projected_without_commands(
    qapp,
    focus_captures: tuple[ParameterContext, ParameterContext],
    busy: str,
) -> None:
    ctrl = _mock_ctrl()
    q1, _q2 = focus_captures
    assert q1.roles is not None
    rebound = ParameterContext(
        point=q1.document(),
        library=q1.library,
        target=q1.target,
        roles=RoleBinding(
            "Q1",
            q1.roles.declarations,
            {"qubit": "Q2"},
            {"qubit": "Q2", "resonator": None},
        ),
    )
    snapshot = replace(_snapshot("a"), parameters=rebound)
    assert snapshot.capabilities is not None and snapshot.interaction is not None
    widget = ExpTabWidget("a", ctrl, snapshot.capabilities)
    widget.attach(snapshot, MagicMock())
    widget.refresh_roles(snapshot)
    driver = widget.findChild(QComboBox, "role-qubit")
    sensor = widget.findChild(QComboBox, "role-resonator")
    manual = widget.findChild(QLabel, "role-manual-resonator")
    reset = widget.findChild(QPushButton, "role-reset-qubit")
    assert (
        driver is not None
        and sensor is not None
        and manual is not None
        and reset is not None
    )
    assert driver.currentData() == "Q2" and sensor.currentData() is None
    assert not manual.isHidden()
    interaction = replace(
        snapshot.interaction,
        is_running=busy == "running",
        is_analyzing=busy == "analyzing",
    )
    widget.update_interaction_state(replace(snapshot, interaction=interaction))
    assert not driver.isEnabled() and not sensor.isEnabled() and not reset.isEnabled()
    widget.update_interaction_state(
        replace(
            snapshot, interaction=replace(snapshot.interaction, global_run_active=True)
        )
    )
    assert driver.isEnabled() and sensor.isEnabled() and reset.isEnabled()
    ctrl.tab_control.set_role.assert_not_called()
    ctrl.tab_control.reset_role.assert_not_called()
    widget.detach()


def test_role_rebind_ui_roleless_capture_hides_row(qapp) -> None:
    ctrl = _mock_ctrl()
    snapshot = _snapshot("a")
    assert snapshot.capabilities is not None
    widget = ExpTabWidget("a", ctrl, snapshot.capabilities)
    widget.attach(snapshot, MagicMock())
    row = widget.findChild(QWidget, "tab-roles")
    assert row is not None and row.isHidden()
    widget.detach()


def test_role_rebind_ui_owner_fact_updates_complete_title_and_keeps_run_marker(
    qapp,
    focus_captures: tuple[ParameterContext, ParameterContext],
) -> None:
    ctrl = _apply_window_defaults(_mock_ctrl())
    bus = EventBus()
    ctrl.get_bus.return_value = bus
    ctrl.get_running_tab_id.return_value = "a"
    q1, _q2 = focus_captures
    assert q1.roles is not None
    snapshot = replace(_snapshot("a"), parameters=q1)
    ctrl.get_tab_snapshot.return_value = snapshot
    window = MainWindow(ctrl)
    window.add_tab_widget("a", "fake")
    tabs = window.findChild(QTabWidget)
    assert tabs is not None
    assert "●" in tabs.tabText(0)
    rebound = ParameterContext(
        point=q1.document(),
        library=q1.library,
        target=q1.target,
        roles=RoleBinding(
            "Q1",
            q1.roles.declarations,
            {"qubit": "Q2"},
            {"qubit": "Q2", "resonator": None},
        ),
    )
    ctrl.get_tab_snapshot.return_value = replace(snapshot, parameters=rebound)
    ctrl.get_tab_snapshot.reset_mock()
    from zcu_tools.gui.app.measure.events.tab import (
        TabContentChangedPayload,
        TabContentFact,
    )

    bus.emit(TabContentChangedPayload(tab_id="a", fact=TabContentFact.ROLES_COMMITTED))
    ctrl.get_tab_snapshot.assert_called_once_with("a")
    assert "●" in tabs.tabText(0)
    assert "qubit=Q2" in tabs.tabText(0)
    assert "resonator=" in tabs.tabText(0)
    assert "resonator=" in tabs.tabToolTip(0)
    widget = window.findChild(ExpTabWidget)
    assert widget is not None
    combo = widget.findChild(QComboBox, "role-qubit")
    assert combo is not None and combo.currentData() == "Q2"
    ctrl.tab_control.set_role.assert_not_called()
    ctrl.tab_control.reset_role.assert_not_called()


def test_role_rebind_ui_rejection_restores_committed_capture_without_retry(
    qapp,
    focus_captures: tuple[ParameterContext, ParameterContext],
) -> None:
    ctrl = _mock_ctrl()
    q1, _q2 = focus_captures
    snapshot = replace(_snapshot("a"), parameters=q1)
    ctrl.tab_control.get_tab_snapshot.return_value = snapshot
    failure = FailedPreconditionError("Tab is busy")
    ctrl.tab_control.set_role.side_effect = failure
    dialogs = RecordingDialogPresenter()
    assert snapshot.capabilities is not None
    widget = ExpTabWidget("a", ctrl, snapshot.capabilities, dialog_presenter=dialogs)
    widget.attach(snapshot, MagicMock())
    selector = widget.findChild(QComboBox, "role-qubit")
    assert selector is not None and selector.currentData() == "Q1"
    index = selector.findData("Q2")
    selector.setCurrentIndex(index)
    selector.activated.emit(index)
    assert selector.currentData() == "Q1"
    ctrl.tab_control.set_role.assert_called_once_with("a", "qubit", "Q2")
    assert len(dialogs.calls) == 1 and dialogs.calls[0].message == str(failure)
    widget.detach()
