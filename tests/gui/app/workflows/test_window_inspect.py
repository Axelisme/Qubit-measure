"""Workflows composes the shared inspector over its live session owners.

Shared dialog tests own editing mechanics. These cases protect the host's
button, session writeback, run lock, fresh alias view and injected catalogs.
"""

from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QApplication,
    QComboBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTabWidget,
    QTreeWidget,
    QWidget,
)
from zcu_tools.cfg_model import ConstWaveformCfg, TemplateDef
from zcu_tools.gui.app.workflows.app import WorkflowsGuiBehavior
from zcu_tools.gui.app.workflows.composition import WorkflowComposition
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_types import (
    CommandOrigin,
    EntryRequest,
    LibraryChange,
    PointCreateRequest,
)
from zcu_tools.gui.session.ui.arb_waveform_dialog import ArbWaveformDialog
from zcu_tools.gui.session.ui.inspect_dialog import InspectDialog
from zcu_tools.gui.session.ui.new_entry_dialog import NewEntryDialog

from tests.gui.widgets.form._text_input import commit_text_on_focus_loss

from ._window_support import WindowSession, bind_point, install_soc, small_registration

WAVEFORM = "Q1.waveforms.base"


def child[W: QWidget](parent: QWidget, kind: type[W], name: str) -> W:
    found = parent.findChild(kind, name)
    assert found is not None, f"Missing Qt control {name}"
    return found


def button(parent: QWidget, text: str) -> QPushButton:
    return next(b for b in parent.findChildren(QPushButton) if b.text() == text)


def open_inspect(window: QWidget, qapp: QApplication) -> InspectDialog:
    child(window, QPushButton, "workflow.source.inspect").click()
    qapp.processEvents()
    shown = [d for d in window.findChildren(InspectDialog) if d.isVisible()]
    assert len(shown) == 1
    return shown[0]


def prepare(session: WindowSession, qapp: QApplication) -> None:
    bind_point(session.harness, session.root)
    install_soc(session.harness)
    context = session.harness.services.context
    context.apply_library_change(
        context.read_context(),
        LibraryChange(
            "waveform", "put", WAVEFORM, waveform=ConstWaveformCfg(length=0.1)
        ),
    )
    session.window.show()
    qapp.processEvents()


def select_waveform(dialog: InspectDialog) -> None:
    child(dialog, QTabWidget, "").setCurrentIndex(1)
    tree = child(dialog, QTreeWidget, "inspect.library")
    matches = tree.findItems(
        WAVEFORM, Qt.MatchFlag.MatchExactly | Qt.MatchFlag.MatchRecursive, 0
    )
    assert len(matches) == 1
    tree.setCurrentItem(matches[0])


def test_inspect_opens_once_and_edits_the_current_session(
    window_session: WindowSession, qapp: QApplication
) -> None:
    prepare(window_session, qapp)
    window = window_session.window
    dialog = open_inspect(window, qapp)
    assert dialog.parent() is window
    assert open_inspect(window, qapp) is dialog
    tabs = child(dialog, QTabWidget, "")
    assert [tabs.tabText(i) for i in range(tabs.count())] == [
        "Parameters",
        "Modules",
        "Aliases",
    ]
    assert any(
        tree.findItems("Q1", Qt.MatchFlag.MatchRecursive)
        for tree in dialog.findChildren(QTreeWidget)
    )
    select_waveform(dialog)
    commit_text_on_focus_loss(child(dialog, QLineEdit, "form.input.length"), "0.25")
    child(dialog, QPushButton, "inspect.entry.apply").click()
    context = window_session.harness.services.context.read_context()
    assert context.library.waveform_spec(WAVEFORM) == ConstWaveformCfg(length=0.25)
    assert child(dialog, QLabel, "inspect.entry.status").text() == "Saved"


def test_inspect_locks_from_start_through_terminal_delivery(
    window_session: WindowSession, qapp: QApplication
) -> None:
    prepare(window_session, qapp)
    window = window_session.window
    dialog = open_inspect(window, qapp)
    select_waveform(dialog)
    assert not child(dialog, QLineEdit, "form.input.length").isReadOnly()
    assert isinstance(window.start("inspect-lock"), int)
    assert not button(dialog, "New").isEnabled()
    assert child(dialog, QLineEdit, "form.input.length").isReadOnly()
    assert not child(dialog, QPushButton, "inspect.aliases.add").isEnabled()
    # Worker completion alone must not unlock source edits.
    window_session.harness.background.execute()
    qapp.processEvents()
    assert not button(dialog, "New").isEnabled()
    window_session.harness.background.deliver()
    qapp.processEvents()
    assert button(dialog, "New").isEnabled()
    assert not child(dialog, QLineEdit, "form.input.length").isReadOnly()
    assert child(dialog, QPushButton, "inspect.aliases.add").isEnabled()


def test_open_arb_locks_until_run_terminal_delivery(
    window_session: WindowSession, qapp: QApplication
) -> None:
    prepare(window_session, qapp)
    window = window_session.window
    inspect = open_inspect(window, qapp)
    button(inspect, "Arb Waveforms…").click()
    qapp.processEvents()
    dialog = next(d for d in window.findChildren(ArbWaveformDialog) if d.isVisible())
    key = child(dialog, QLineEdit, "")
    key.setText("pre_run_draft")
    assert not key.isReadOnly()
    assert button(dialog, "Save").isEnabled()

    assert isinstance(window.start("arb-lock"), int)
    assert key.isReadOnly()
    assert not button(dialog, "Save").isEnabled()
    window_session.harness.background.execute()
    qapp.processEvents()
    assert key.isReadOnly()
    assert not button(dialog, "Save").isEnabled()
    window_session.harness.background.deliver()
    qapp.processEvents()
    assert not key.isReadOnly()
    assert button(dialog, "Save").isEnabled()
    assert key.text() == "pre_run_draft"


def test_inspect_opened_while_paused_is_read_only_until_stop(
    window_session: WindowSession, qapp: QApplication
) -> None:
    prepare(window_session, qapp)
    window_session.pause_at = 0
    window = window_session.window
    assert isinstance(window.start("inspect-paused"), int)
    window_session.harness.complete_segment()
    dialog = open_inspect(window, qapp)
    select_waveform(dialog)
    assert child(window, QPushButton, "workflow.source.inspect").isEnabled()
    assert child(dialog, QLineEdit, "form.input.length").isReadOnly()
    button(dialog, "Arb Waveforms…").click()
    qapp.processEvents()
    arb = next(d for d in window.findChildren(ArbWaveformDialog) if d.isVisible())
    assert not button(arb, "Save").isEnabled()
    arb_key = arb.findChild(QLineEdit)
    assert arb_key is not None and arb_key.isReadOnly()
    arb.close()
    qapp.processEvents()
    window.stop("inspect-paused")
    qapp.processEvents()
    assert not child(dialog, QLineEdit, "form.input.length").isReadOnly()
    assert button(dialog, "Arb Waveforms…").isEnabled()


def test_reopening_inspect_reads_the_latest_alias_table(
    window_session: WindowSession, qapp: QApplication
) -> None:
    prepare(window_session, qapp)
    window = window_session.window
    dialog = open_inspect(window, qapp)
    dialog.close()
    qapp.processEvents()
    rows = (AliasRow("new_alias", "Q1"),)
    window_session.harness.services.context.set_alias_rows(rows)
    reopened = open_inspect(window, qapp)
    table = child(reopened, QTableWidget, "inspect.aliases.table")
    assert table.rowCount() == 1
    alias, value = table.item(0, 0), table.item(0, 1)
    assert alias is not None and alias.text() == "new_alias"
    assert value is not None and value.text() == "Q1"


def test_composition_supplies_templates_presets_and_the_shared_arb_dialog(
    tmp_path: Path, qapp: QApplication
) -> None:
    templates = {
        "flat": TemplateDef(
            ConstWaveformCfg,
            "Synthetic flat",
            {"length": 0.3},
            default_name="Q1.waveforms.flat",
        )
    }
    presets = {"default": (), "probe": (AliasRow("probe", "Q1"),)}
    app = WorkflowsGuiBehavior(
        composition=WorkflowComposition(
            registrations=[small_registration()],
            templates=templates,
            alias_presets=presets,
        ),
        project_root=str(tmp_path),
        cache_dir=tmp_path / "cache",
        clean=True,
    )
    app.assemble(None)
    try:
        # Aliases and ARB remain available even before opening a point.
        app.window.show()
        dialog = open_inspect(app.window, qapp)
        menu = child(dialog, QPushButton, "inspect.aliases.load_preset").menu()
        assert menu is not None
        assert [action.text() for action in menu.actions()] == ["default", "probe"]
        menu.actions()[1].trigger()
        child(dialog, QPushButton, "inspect.aliases.apply").click()
        assert app.services.context.get_alias_rows() == presets["probe"]
        button(dialog, "Arb Waveforms…").click()
        qapp.processEvents()
        arb = [d for d in app.window.findChildren(ArbWaveformDialog) if d.isVisible()]
        assert len(arb) == 1
        arb[0].close()
        # A point is needed to materialize a template into an editable entry.
        app.services.settings.new_entry(EntryRequest("sample"))
        app.services.context.new_point(
            PointCreateRequest("source", "", CommandOrigin("gui"))
        )
        child(dialog, QTabWidget, "").setCurrentIndex(1)
        button(dialog, "New").click()
        qapp.processEvents()
        new = dialog.findChild(NewEntryDialog)
        assert new is not None and new.isVisible()
        combo = child(new, QComboBox, "inspect.new_entry.template")
        assert [combo.itemData(i) for i in range(combo.count())] == ["flat"]
        button(new, "Create").click()
        assert app.services.context.read_context().library.waveform_spec(
            "Q1.waveforms.flat"
        ) == ConstWaveformCfg(length=0.3)
    finally:
        app.window.close()
        app.shutdown()
        app.window.deleteLater()
        qapp.processEvents()
