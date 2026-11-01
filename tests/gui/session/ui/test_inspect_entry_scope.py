"""Entry destinations through real keyboard/mouse input and visible observations.

Persistence, cached revisions and fixed captures belong to the ContextService
seam tests; these scenarios only inspect rendered controls and dialogs.
"""

import os
from collections.abc import Generator
from pathlib import Path

import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication
from zcu_tools.cfg_model import ConstWaveformCfg, PulseSpec, TemplateDef
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_types import PointUseRequest
from zcu_tools.gui.session.services.context import ContextService
from zcu_tools.gui.session.state import SessionState
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.gui.session.ui.inspect_dialog import InspectDialog
from zcu_tools.resources.entry import ResultEntry
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase

from tests.gui._dialog_fakes import RecordingDialogPresenter
from tests.gui._user_input import ComponentDriver
from tests.resources.entry.fakes import register_fakes, registry_state, restore_registry


class _NoArb:
    def list_data_keys(self) -> list[str]:
        return []


@pytest.fixture(params=["global", "point"])
def entry_ui(
    qapp: QApplication, tmp_path: Path, request: pytest.FixtureRequest
) -> Generator[tuple[ComponentDriver, ComponentDriver, Path, str]]:
    before = registry_state()
    dialogs: list[InspectDialog] = []
    try:
        register_fakes()
        entry = ResultEntry.create(
            "entry",
            result_root=tmp_path / "results",
            database_root=tmp_path / "Database",
        )
        entry.setup.add_component("Q1", kind="fake/drive/a", rate=1200.0)
        entry.setup.add_component("Q2", kind="fake/drive/a", rate=6500.0)
        with entry.module_library().edit() as draft:
            draft.set_waveform("Q1.waveforms.base", ConstWaveformCfg(length=0.1))
        bus = BaseEventBus()
        context = ContextService(
            SessionState(SessionEnv(parameters=None, soc=None, soccfg=None)),
            bus,
            bind_assets=ArbWaveformDatabase.info_at,
        )
        context.install_entry(entry)
        context.apply_alias_rows((AliasRow("driver", "Q1"),))
        if request.param == "point":
            entry.new_point("working")
            target = context.use_point(PointUseRequest("working"))
            context.set_parameter(target, "Q1.rate", 4300.0)
            context.apply_alias_rows((AliasRow("driver", "Q2"),))
        templates = {
            "aliased": TemplateDef(
                PulseSpec,
                "Aliased pulse",
                {
                    "ch": 0,
                    "nqz": 1,
                    "freq": 120.0,
                    "gain": 0.3,
                    "waveform": "{driver}.waveforms.base",
                },
                default_name="Q1.pulses.created",
            )
        }
        for _ in range(2):
            dialog = InspectDialog(
                context,
                bus,
                templates=templates,
                arb_waveforms=_NoArb(),
                open_arb_waveforms=lambda: None,
                dialog_presenter=RecordingDialogPresenter(),
            )
            dialogs.append(dialog)
            dialog.resize(1000, 760)
            dialog.show()
        qapp.processEvents()
        evidence = Path(os.environ.get("INSPECT_ENTRY_EVIDENCE", str(tmp_path)))
        yield (
            ComponentDriver(dialogs[0]),
            ComponentDriver(dialogs[1]),
            evidence / request.param,
            request.param,
        )
    finally:
        for dialog in dialogs:
            dialog.close()
            dialog.deleteLater()
        qapp.processEvents()
        restore_registry(before)


def _parameter(ui: ComponentDriver) -> str:
    return next(
        row.columns[1] for row in ui.visible().tree_rows if row.path == ("Q1", "rate")
    )


def _parameters(ui: ComponentDriver) -> None:
    ui.select_tab("Parameters")
    ui.choose_option(
        object_name="inspect.parameters.destination", value="Entry template"
    )
    ui.enter_text(
        object_name="inspect.parameters.filter", value="Q1.rate", commit=False
    )


def _waveform(ui: ComponentDriver) -> None:
    ui.select_tab("Modules")
    ui.choose_option(object_name="inspect.library.destination", value="Entry template")
    ui.enter_text(
        object_name="inspect.library.filter", value="Q1.waveforms.base", commit=False
    )
    ui.select_tree_row(
        ("waveforms", "Q1", "waveforms", "base"), object_name="inspect.library"
    )


def test_entry_destinations_commit_and_notify_clean_inspectors(entry_ui) -> None:
    ui, observer, evidence, scope = entry_ui
    _parameters(ui)
    _parameters(observer)
    ui.screenshot(evidence / "01-parameters-entry.png")
    assert _parameter(ui) == "1200.0"
    ui.edit_tree_value(("Q1", "rate"), object_name="component_tree", value="1250")
    ui.screenshot(evidence / "02-parameters-saved.png")
    observer.screenshot(evidence / "03-parameters-observer.png")
    assert ui.observe(object_name="inspect.saved").text.startswith("Saved")
    assert observer.observe(object_name="inspect.saved").text.startswith("Saved")
    assert _parameter(observer) == "1250.0"
    ui.choose_option(
        object_name="inspect.parameters.destination", value="Active documents"
    )
    ui.screenshot(evidence / "04-parameters-active.png")
    assert _parameter(ui) == ("4300.0" if scope == "point" else "1250.0")

    _waveform(ui)
    _waveform(observer)
    ui.enter_text(object_name="form.input.length", value="0.35")
    ui.screenshot(evidence / "05-library-saved.png")
    observer.screenshot(evidence / "06-library-observer.png")
    assert ui.observe(object_name="inspect.saved").text.startswith("Saved")
    assert observer.observe(object_name="inspect.saved").text.startswith("Saved")
    assert observer.observe(object_name="form.input.length").text == "0.35"
    ui.click("New")
    ui.screenshot(evidence / "07-new-entry-capture.png")
    assert ui.observe(object_name="form.input.waveform").text == "Q1.waveforms.base"
    assert ui.observe("Create").enabled
    ui.click("Create")
    ui.enter_text(object_name="inspect.library.filter", value="created", commit=False)
    ui.screenshot(evidence / "08-new-entry-saved.png")
    assert ui.observe(object_name="inspect.entry.name").text == "Q1.pulses.created"

    for driver in (ui, observer):
        driver.select_tab("Aliases")
        driver.choose_option(
            object_name="inspect.aliases.destination", value="Entry template"
        )
    ui.edit_table_value("Q1", "Q2", object_name="inspect.aliases.table", column="Value")
    ui.screenshot(evidence / "09-aliases-saved.png")
    observer.screenshot(evidence / "10-aliases-observer.png")
    assert ui.observe(object_name="inspect.saved").text.startswith("Saved")
    assert observer.observe(object_name="inspect.saved").text.startswith("Saved")
    assert (
        observer.table_rows(object_name="inspect.aliases.table")[0].columns[1] == "Q2"
    )


def test_entry_notifications_preserve_dirty_alias_draft(entry_ui) -> None:
    ui, observer, evidence, _ = entry_ui
    for driver in (ui, observer):
        driver.select_tab("Aliases")
        driver.choose_option(
            object_name="inspect.aliases.destination", value="Entry template"
        )
    observer.edit_table_value(
        "driver", "invalid name", object_name="inspect.aliases.table", column="Alias"
    )
    observer.screenshot(evidence / "14-alias-dirty.png")
    ui.edit_table_value("Q1", "Q2", object_name="inspect.aliases.table", column="Value")
    observer.screenshot(evidence / "15-alias-dirty-notified.png")
    assert observer.table_rows(object_name="inspect.aliases.table")[0].columns[:2] == (
        "invalid name",
        "Q1",
    )
    observer.edit_table_value(
        "invalid name", "driver", object_name="inspect.aliases.table", column="Alias"
    )
    observer.screenshot(evidence / "16-alias-stale-rejected.png")
    assert (
        "revision" in observer.observe(object_name="inspect.aliases.error").text.lower()
    )


def test_entry_notifications_preserve_dirty_library_draft(entry_ui) -> None:
    ui, observer, evidence, _ = entry_ui
    _waveform(ui)
    _waveform(observer)
    observer.enter_text(object_name="form.input.length", value="invalid", commit=False)
    observer.screenshot(evidence / "11-library-dirty.png")
    ui.enter_text(object_name="form.input.length", value="0.45")
    observer.screenshot(evidence / "12-library-dirty-notified.png")
    assert observer.observe(object_name="form.input.length").text == "invalid"
    observer.enter_text(object_name="form.input.length", value="0.55", commit=False)
    observer.press_key(Qt.Key.Key_Return, object_name="form.input.length")
    observer.screenshot(evidence / "13-library-stale-rejected.png")
    assert (
        "revision" in observer.observe(object_name="inspect.entry.error").text.lower()
    )
