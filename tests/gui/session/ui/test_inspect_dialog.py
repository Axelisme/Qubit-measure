"""The shared Inspect dialog edits library entries through real session owners."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Generator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from qtpy.QtCore import QEvent, Qt
from qtpy.QtGui import QKeyEvent
from qtpy.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QTreeWidget,
    QWidget,
)
from zcu_tools.cfg_model import ConstWaveformCfg, PulseSpec, TemplateDef
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.form import FormErrors
from zcu_tools.gui.form_adapter import module_form
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_control import ContextControlFacet
from zcu_tools.gui.session.context_types import PointUseRequest
from zcu_tools.gui.session.events import ContextSwitchedPayload
from zcu_tools.gui.session.hardware_gate import RunBlocksHardwareGate
from zcu_tools.gui.session.library_entry import open_library_entry
from zcu_tools.gui.session.operation_handles import OperationHandles
from zcu_tools.gui.session.operation_runner import (
    OperationInfrastructure,
    OperationRunner,
)
from zcu_tools.gui.session.services.build import build_session_services
from zcu_tools.gui.session.services.context import ContextService
from zcu_tools.gui.session.services.progress import ProgressService
from zcu_tools.gui.session.state import SessionState
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.gui.session.ui.arb_waveform_dialog import ArbWaveformDialog
from zcu_tools.gui.session.ui.inspect_dialog import InspectDialog
from zcu_tools.resources.entry import ResultEntry
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase

from tests.gui._dialog_fakes import RecordingDialogPresenter
from tests.gui._user_input import ComponentDriver, key_click
from tests.gui.widgets.form._text_input import commit_text_on_focus_loss as commit_text

PULSE = "Q1.pulses.pi"
WAVEFORM = "Q1.waveforms.base"

TEMPLATES: dict[str, TemplateDef] = {
    "probe": TemplateDef(
        PulseSpec,
        "Probe pulse",
        {
            "ch": 1,
            "nqz": 1,
            "freq": 100.0,
            "gain": 0.5,
            "waveform.style": "const",
            "waveform.length": 0.2,
        },
        default_name="Q1.pulses.probe",
    ),
    "drive": TemplateDef(
        PulseSpec,
        "Drive pulse",
        {
            "ch": 2,
            "nqz": 2,
            "freq": 200.0,
            "gain": 0.1,
            "waveform.style": "const",
            "waveform.length": 0.4,
        },
        default_name="Q1.pulses.drive",
    ),
    "flat": TemplateDef(
        ConstWaveformCfg,
        "Flat waveform",
        {"length": 0.3},
        default_name="Q1.waveforms.flat",
    ),
}


class _ArbNames:
    """Arb asset names offered to arb data fields; no asset database is read."""

    def list_data_keys(self) -> list[str]:
        return []


@dataclass
class _Session:
    entry: ResultEntry
    control: ContextControlFacet
    bus: BaseEventBus
    dialogs: list[QDialog] = field(default_factory=list)
    arb_opened: list[bool] = field(default_factory=list)


@pytest.fixture
def session(qapp: QApplication, tmp_path: Path) -> Generator[_Session]:
    entry = ResultEntry.create(
        "entry", result_root=tmp_path / "results", database_root=tmp_path / "Database"
    )
    entry.new_point("point")
    with entry.module_library("point").edit() as draft:
        draft.set_waveform(WAVEFORM, ConstWaveformCfg(length=0.1))
        draft.set_module(
            PULSE,
            PulseSpec(ch=0, nqz=1, freq=100.0, gain=0.25, waveform=WAVEFORM),
        )
    bus = BaseEventBus()
    service = ContextService(
        SessionState(SessionEnv(parameters=None, soc=None, soccfg=None)),
        bus,
        bind_assets=ArbWaveformDatabase.info_at,
    )
    service.install_entry(entry)
    control = ContextControlFacet(context=service)
    control.use_point(PointUseRequest("point"))
    session = _Session(entry, control, bus)
    try:
        yield session
    finally:
        for dialog in session.dialogs:
            # A window-modal New dialog left open would block later owner turns.
            for child in dialog.findChildren(QDialog):
                child.close()
            if isinstance(dialog, InspectDialog):
                dialog.set_read_only(read_only=True)
            dialog.close()
            dialog.deleteLater()
        qapp.processEvents()


def _open(
    session: _Session,
    presenter: RecordingDialogPresenter,
    *,
    templates: Mapping[str, TemplateDef] = TEMPLATES,
) -> InspectDialog:
    """Build the shared Inspect with every required collaborator and show Modules."""
    dialog = InspectDialog(
        session.control,
        session.bus,
        templates=templates,
        arb_waveforms=_ArbNames(),
        open_arb_waveforms=lambda: session.arb_opened.append(True),
        dialog_presenter=presenter,
    )
    session.dialogs.append(dialog)
    tabs = dialog.findChild(QTabWidget)
    assert tabs is not None
    tabs.setCurrentIndex(1)
    dialog.show()
    dialog.activateWindow()
    QApplication.processEvents()
    return dialog


def _child[W: QWidget](parent: QWidget, kind: type[W], name: str) -> W:
    widget = parent.findChild(kind, name)
    assert widget is not None, f"{name!r} is missing"
    return widget


def _type(widget: QWidget, text: str) -> None:
    """Deliver text to widget as user key presses, one character each."""
    for char in text:
        QApplication.sendEvent(
            widget,
            QKeyEvent(
                QEvent.Type.KeyPress,
                Qt.Key.Key_unknown,
                Qt.KeyboardModifier.NoModifier,
                char,
            ),
        )


def _button(parent: QWidget, text: str) -> QPushButton:
    return next(b for b in parent.findChildren(QPushButton) if b.text() == text)


def _select(dialog: InspectDialog, name: str) -> None:
    tree = _child(dialog, QTreeWidget, "inspect.library")
    (item,) = (
        row
        for row in tree.findItems(
            "*", Qt.MatchFlag.MatchWildcard | Qt.MatchFlag.MatchRecursive
        )
        if (data := row.data(0, Qt.ItemDataRole.UserRole)) is not None
        and data[1] == name
    )
    tree.setCurrentItem(item)


def _group_fold(parent: QWidget, name: str) -> QLabel:
    return _child(parent, QLabel, f"form.fold.group.{name}")


def _library(session: _Session):
    return session.entry.module_library("point").snapshot()


def _pulse(session: _Session, name: str = PULSE) -> PulseSpec:
    spec = _library(session).module_spec(name)
    assert isinstance(spec, PulseSpec)
    return spec


@pytest.mark.parametrize("submit", ["enter", "focus-loss"])
def test_module_commits_on_enter_not_while_typing(
    session: _Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, submit: str
) -> None:
    from zcu_tools.gui.widgets.form import value_widgets

    from tests.gui.widgets.form._clock import Clock

    clock = Clock()
    monkeypatch.setattr(value_widgets, "QTimer", clock.timer)
    dialog = _open(session, RecordingDialogPresenter())
    ui = ComponentDriver(dialog)
    ui.resize_window(1100, 760)
    ui.select_tree_row(("modules", "Q1", "pulses", "pi"), object_name="inspect.library")
    evidence = Path(os.environ.get("INSPECT_IMPLICIT_EVIDENCE", str(tmp_path)))
    ui.screenshot(evidence / "modules-01-open.png")
    # Synthetic mouse delivery does not assign Qt focus. Traverse the real
    # focus chain so the focus-loss branch exercises an ordinary FocusOut.
    for _ in range(20):
        if ui.text_input(object_name="form.input.gain").focused:
            break
        ui.press_key(Qt.Key.Key_Tab)
    assert ui.text_input(object_name="form.input.gain").focused
    ui.enter_text(object_name="form.input.gain", value="0.5", commit=False)
    clock.advance(1000)
    ui.screenshot(evidence / "modules-02-typing.png")
    assert _pulse(session).gain == 0.25
    if submit == "enter":
        ui.press_key(Qt.Key.Key_Return, object_name="form.input.gain")
    else:
        ui.press_key(Qt.Key.Key_Tab, object_name="form.input.gain")
    ui.screenshot(evidence / f"modules-03-{submit}.png")
    assert _pulse(session).gain == 0.5
    assert ui.observe(object_name="form.input.gain").text == "0.5"
    assert ui.observe(object_name="inspect.saved").text.startswith("Saved")


def test_module_dropdown_writes_immediately(session: _Session, tmp_path: Path) -> None:
    ui = ComponentDriver(_open(session, RecordingDialogPresenter()))
    ui.resize_window(1100, 760)
    ui.select_tree_row(("modules", "Q1", "pulses", "pi"), object_name="inspect.library")
    ui.click(object_name="form.fold.group.Advanced")
    ui.choose_option(object_name="form.input.nqz", value="2")
    ui.screenshot(
        Path(os.environ.get("INSPECT_IMPLICIT_EVIDENCE", str(tmp_path)))
        / "dropdown-written.png"
    )
    assert _pulse(session).nqz == 2


def test_library_filter_and_hierarchy_keep_leaf_edit_actions(
    session: _Session, tmp_path: Path
) -> None:
    presenter = RecordingDialogPresenter()
    dialog = _open(session, presenter)
    dialog.resize(1100, 760)
    ui = ComponentDriver(dialog)
    evidence = Path(os.environ.get("INSPECT_FILTER_EVIDENCE", str(tmp_path)))

    def rows():
        return {
            row.path: row
            for row in ui.visible().tree_rows
            if row.tree_name == "inspect.library"
        }

    ui.screenshot(evidence / "modules-default.png")
    leaf = ("modules", "Q1", "pulses", "pi")
    assert leaf in rows()
    assert ("waveforms", "Q1", "waveforms", "base") in rows()
    ui.press_key(Qt.Key.Key_Home, object_name="inspect.library")
    ui.press_key(Qt.Key.Key_Left, object_name="inspect.library")
    assert not rows()[("modules",)].expanded
    assert ui.observe(object_name="inspect.library.filter").visible
    ui.enter_text(
        object_name="inspect.library.filter", value="q1.PULSES.pi", commit=False
    )
    ui.screenshot(evidence / "modules-filtered.png")
    assert set(rows()) == {leaf[:depth] for depth in range(1, 5)}
    assert all(rows()[leaf[:depth]].expanded for depth in range(1, 4))
    ui.enter_text(object_name="inspect.library.filter", value="pi", commit=False)
    assert leaf in rows()
    ui.enter_text(object_name="inspect.library.filter", value="", commit=False)
    assert not rows()[("modules",)].expanded
    ui.enter_text(object_name="inspect.library.filter", value="PI", commit=False)
    ui.expand_tree(leaf, object_name="inspect.library")
    assert ui.observe(object_name="inspect.entry.name").text == PULSE
    ui.press_key(Qt.Key.Key_Home, object_name="inspect.library")
    ui.press_key(Qt.Key.Key_Down, object_name="inspect.library")
    assert not ui.observe(object_name="inspect.entry.name").enabled
    ui.expand_tree(leaf, object_name="inspect.library")
    assert ui.observe(object_name="inspect.entry.name").text == PULSE
    ui.enter_text(object_name="inspect.entry.name", value="Q1.pulses.pi_v2")
    assert ui.observe(object_name="inspect.entry.name").text == "Q1.pulses.pi_v2"
    assert ("modules", "Q1", "pulses", "pi_v2") in rows()
    presenter.queue_confirm(True)
    ui.click("Delete")
    ui.enter_text(object_name="inspect.library.filter", value="", commit=False)
    assert ("modules",) not in rows()
    waveform = ("waveforms", "Q1", "waveforms", "base")
    ui.expand_tree(waveform, object_name="inspect.library")
    assert ui.observe(object_name="inspect.entry.name").text == WAVEFORM
    ui.enter_text(object_name="inspect.library.filter", value="no-match", commit=False)
    assert not rows()


def test_reopened_module_preserves_full_frequency_precision_on_commit(
    session: _Session,
) -> None:
    session.control.set_module(
        session.control.read_context().target,
        PULSE,
        PulseSpec(ch=0, nqz=1, freq=5000.1234, gain=0.25, waveform=WAVEFORM),
    )
    ui = ComponentDriver(_open(session, RecordingDialogPresenter()))
    ui.resize_window(1100, 760)
    ui.select_tree_row(("modules", "Q1", "pulses", "pi"), object_name="inspect.library")
    assert ui.observe(object_name="form.input.freq").text == "5000.123"
    ui.enter_text(object_name="form.input.gain", value="0.5")
    written = _library(session).module_spec(PULSE)
    assert isinstance(written, PulseSpec)
    assert written.freq == 5000.1234
    assert written.gain == 0.5


def test_child_waveform_key_stays_a_key_after_apply(session: _Session) -> None:
    # A is a separate public capture. Inspect's capture/state have no public
    # read seam; B below observes only the actual host's Qt controls and Apply.
    capture = session.control.read_context()
    assert WAVEFORM in capture.library.waveform_keys()
    opened = open_library_entry(capture, "module", PULSE, arb_names=())
    try:
        value = opened.state.result(opened.env)
    except ValueError as error:
        pytest.fail(f"A valid captured waveform key must evaluate: {error}")
    assert not isinstance(value, FormErrors)
    initial = opened.spec()
    assert isinstance(initial, PulseSpec)
    assert initial.waveform == WAVEFORM
    current = opened.state.input.data["waveform"]
    assert current == {"kind": "ref", "name": WAVEFORM}
    options = module_form(PulseSpec).model.options(opened.env)[("waveform",)]
    selected = next(option for option in options if option.matches(current))
    assert selected.label == WAVEFORM
    assert selected.value == current

    presenter = RecordingDialogPresenter()
    dialog = _open(session, presenter)
    _select(dialog, PULSE)
    waveform = _child(dialog, QComboBox, "form.input.waveform")
    assert waveform.currentText() == selected.label
    assert waveform.currentData() == selected.value
    assert (selected.label, selected.value) in [
        (waveform.itemText(index), waveform.itemData(index))
        for index in range(waveform.count())
    ]
    assert not any(
        "(missing)" in waveform.itemText(index) for index in range(waveform.count())
    )
    commit_text(_child(dialog, QLineEdit, "form.input.freq"), "150")
    written = _library(session).module_spec(PULSE)
    assert isinstance(written, PulseSpec)
    assert written.waveform == WAVEFORM
    assert written.freq == 150.0
    waveform = _child(dialog, QComboBox, "form.input.waveform")
    assert waveform.currentText() == selected.label
    assert waveform.currentData() == selected.value
    presenter.assert_no_unexpected_messages()


def test_root_family_tag_has_no_row(session: _Session) -> None:
    dialog = _open(session, RecordingDialogPresenter())
    _select(dialog, PULSE)
    assert dialog.findChild(QLineEdit, "form.input.gain") is not None
    assert dialog.findChild(QWidget, "form.input.type") is None


def test_name_enter_renames_and_a_taken_name_keeps_input_inline(
    session: _Session,
    tmp_path: Path,
) -> None:
    session.control.set_module(
        session.control.read_context().target,
        f"{PULSE}_v2",
        PulseSpec(ch=0, nqz=1, freq=1.0, gain=0.1, waveform=WAVEFORM),
    )
    before = _library(session).document()
    ui = ComponentDriver(_open(session, RecordingDialogPresenter()))
    ui.resize_window(1100, 760)
    ui.select_tree_row(("modules", "Q1", "pulses", "pi"), object_name="inspect.library")
    ui.enter_text(object_name="inspect.entry.name", value=f"{PULSE}_v2")
    ui.screenshot(
        Path(os.environ.get("INSPECT_IMPLICIT_EVIDENCE", str(tmp_path)))
        / "name-rejected.png"
    )
    assert ui.observe(object_name="inspect.entry.error").text
    assert ui.observe(object_name="inspect.entry.name").text == f"{PULSE}_v2"
    assert _library(session).document() == before
    ui.enter_text(object_name="inspect.entry.name", value=f"{PULSE}_v3")
    ui.screenshot(
        Path(os.environ.get("INSPECT_IMPLICIT_EVIDENCE", str(tmp_path)))
        / "name-renamed.png"
    )
    assert ui.observe(object_name="inspect.entry.error").text == ""
    assert PULSE not in _library(session).module_keys()
    assert _pulse(session, f"{PULSE}_v3").gain == 0.25


@pytest.mark.parametrize("invalid", ["oops", "2"])
def test_invalid_module_input_is_red_and_discarded_when_switching(
    session: _Session, tmp_path: Path, invalid: str
) -> None:
    presenter = RecordingDialogPresenter()
    ui = ComponentDriver(_open(session, presenter))
    ui.resize_window(1100, 760)
    ui.select_tree_row(("modules", "Q1", "pulses", "pi"), object_name="inspect.library")
    before = _library(session).document()
    help_before = ui.observe(object_name="form.row.gain").tooltip
    ui.enter_text(object_name="form.input.gain", value=invalid)
    ui.screenshot(
        Path(os.environ.get("INSPECT_IMPLICIT_EVIDENCE", str(tmp_path)))
        / "invalid-module.png"
    )
    assert ui.observe(object_name="form.row.gain").tooltip != help_before
    assert ui.observe(object_name="form.input.gain").text == invalid
    assert _library(session).document() == before
    ui.select_tree_row(
        ("waveforms", "Q1", "waveforms", "base"), object_name="inspect.library"
    )
    ui.screenshot(
        Path(os.environ.get("INSPECT_IMPLICIT_EVIDENCE", str(tmp_path)))
        / "invalid-discarded.png"
    )
    ui.select_tree_row(("modules", "Q1", "pulses", "pi"), object_name="inspect.library")
    assert ui.observe(object_name="form.input.gain").text == "0.25"
    ui.click("Close")
    presenter.assert_no_unexpected_messages()


def test_escape_retires_inspect_before_late_context_broadcast(
    session: _Session, qapp: QApplication, tmp_path: Path
) -> None:
    presenter = RecordingDialogPresenter()
    dialog = _open(session, presenter)
    dialog.resize(1100, 760)
    driver = ComponentDriver(dialog)
    driver.expand_tree(("modules", "Q1", "pulses", "pi"), object_name="inspect.library")
    assert driver.observe(object_name="inspect.entry.name").text == PULSE
    driver.enter_text(object_name="form.input.gain", value="0.8", commit=False)
    driver.screenshot(tmp_path / "01-pending-inspect.png")
    # Reject removes the form synchronously. Target a surviving dialog control
    # so the real key release is not sent to the retired input.
    driver.press_key(Qt.Key.Key_Escape, text="Close")
    assert not driver.visible().widgets
    session.bus.emit(ContextSwitchedPayload(binding_id=None, sources=None))
    # Use an actual new view to observe that rejecting the old editor saved nothing.
    replacement = _open(session, presenter)
    replacement.resize(1100, 760)
    other = ComponentDriver(replacement)
    other.expand_tree(("modules", "Q1", "pulses", "pi"), object_name="inspect.library")
    other.screenshot(tmp_path / "02-reopened-inspect.png")
    assert other.observe(object_name="form.input.gain").text == "0.25"
    dialog.deleteLater()
    qapp.processEvents()
    session.bus.emit(ContextSwitchedPayload(binding_id=None, sources=None))
    other.screenshot(tmp_path / "03-after-late-broadcast.png")
    assert other.observe(object_name="inspect.entry.name").text == PULSE


def test_read_only_locks_inputs_but_keeps_scrolling_and_folds(
    session: _Session, qapp: QApplication
) -> None:
    dialog = _open(session, RecordingDialogPresenter())
    dialog.resize(520, 260)
    dialog.show()
    _select(dialog, PULSE)
    dialog.set_read_only(read_only=True)
    qapp.processEvents()

    gain = _child(dialog, QLineEdit, "form.input.gain")
    assert gain.isReadOnly()
    assert _child(dialog, QLineEdit, "inspect.entry.name").isReadOnly()
    for text in ("New", "Delete"):
        assert not _button(dialog, text).isEnabled(), text
    # The host opens ARB read-only by its own rule; the entry stays reachable.
    assert _button(dialog, "Arb Waveforms…").isEnabled()
    _type(gain, "9")
    assert float(gain.text()) == 0.25

    advanced = _group_fold(dialog, "Advanced")
    assert advanced.text().startswith("▸ ")
    key_click(advanced, Qt.Key.Key_Space)
    nqz = _child(dialog, QWidget, "form.input.nqz")
    assert nqz.isVisibleTo(dialog)
    qapp.processEvents()
    bar = _child(dialog, QScrollArea, "inspect.entry.scroll").verticalScrollBar()
    assert bar is not None and bar.maximum() > 0
    bar.setValue(bar.maximum())
    assert bar.value() == bar.maximum()

    dialog.set_read_only(read_only=False)
    assert not _child(dialog, QLineEdit, "form.input.gain").isReadOnly()
    assert _button(dialog, "New").isEnabled()


def test_delete_button_confirms_and_the_tree_delete_key_is_direct(
    session: _Session, qapp: QApplication
) -> None:
    presenter = RecordingDialogPresenter()
    dialog = _open(session, presenter)
    _select(dialog, PULSE)
    before = _library(session).document()

    presenter.queue_confirm(False)
    _button(dialog, "Delete").click()
    assert _library(session).document() == before
    presenter.queue_confirm(True)
    _button(dialog, "Delete").click()
    assert _library(session).module_keys() == ()

    dialog.show()
    _select(dialog, WAVEFORM)
    tree = _child(dialog, QTreeWidget, "inspect.library")
    tree.setFocus()
    qapp.processEvents()
    calls = len(presenter.calls)
    QApplication.sendEvent(
        tree,
        QKeyEvent(
            QEvent.Type.KeyPress, Qt.Key.Key_Delete, Qt.KeyboardModifier.NoModifier
        ),
    )
    assert _library(session).waveform_keys() == ()
    assert len(presenter.calls) == calls, "the Delete key does not confirm"
    presenter.assert_no_unexpected_messages()


def _new_dialog(dialog: InspectDialog) -> QDialog:
    _button(dialog, "New").click()
    created = _child(dialog, QDialog, "inspect.new_entry")
    assert created.isVisible()
    return created


def _choose_template(new: QDialog, label: str) -> None:
    combo = _child(new, QComboBox, "inspect.new_entry.template")
    index = next(i for i in range(combo.count()) if combo.itemText(i).startswith(label))
    combo.setCurrentIndex(index)


def test_new_entry_suggests_names_until_the_user_types_one(
    session: _Session,
) -> None:
    session.control.set_module(
        session.control.read_context().target,
        "Q1.pulses.probe",
        PulseSpec(ch=0, nqz=1, freq=1.0, gain=0.1, waveform=WAVEFORM),
    )
    dialog = _open(session, RecordingDialogPresenter())
    new = _new_dialog(dialog)
    name = _child(new, QLineEdit, "inspect.new_entry.name")
    combo = _child(new, QComboBox, "inspect.new_entry.template")
    assert [combo.itemText(i) for i in range(combo.count())] == [
        "Probe pulse  (module)",
        "Drive pulse  (module)",
        "Flat waveform  (waveform)",
    ]
    assert name.text() == "Q1.pulses.probe_2"
    _choose_template(new, "Drive pulse")
    assert name.text() == "Q1.pulses.drive"
    name.clear()
    _type(name, "mine")
    _choose_template(new, "Flat waveform")
    assert name.text() == "mine"


def test_template_switch_keeps_only_the_name_and_creates_the_new_template(
    session: _Session, qapp: QApplication
) -> None:
    presenter = RecordingDialogPresenter()
    dialog = _open(session, presenter)
    new = _new_dialog(dialog)
    new.resize(420, 240)
    qapp.processEvents()
    name = _child(new, QLineEdit, "inspect.new_entry.name")
    name.clear()
    _type(name, "Q1.pulses.my_pulse")
    commit_text(_child(new, QLineEdit, "form.input.gain"), "0.9")
    key_click(_group_fold(new, "Advanced"), Qt.Key.Key_Space)
    nqz = _child(new, QWidget, "form.input.nqz")
    assert nqz.isVisibleTo(new)
    qapp.processEvents()
    bar = _child(new, QScrollArea, "inspect.new_entry.scroll").verticalScrollBar()
    assert bar is not None and bar.maximum() > 0
    bar.setValue(bar.maximum())

    _choose_template(new, "Drive pulse")
    qapp.processEvents()
    assert name.text() == "Q1.pulses.my_pulse"
    assert float(_child(new, QLineEdit, "form.input.gain").text()) == 0.1
    assert float(_child(new, QLineEdit, "form.input.freq").text()) == 200.0
    assert _group_fold(new, "Advanced").text().startswith("▸ ")
    bar = _child(new, QScrollArea, "inspect.new_entry.scroll").verticalScrollBar()
    assert bar is not None and bar.value() == 0

    _button(new, "Create").click()
    created = _library(session).module_spec("Q1.pulses.my_pulse")
    assert isinstance(created, PulseSpec)
    assert (created.ch, created.nqz, created.freq, created.gain) == (2, 2, 200.0, 0.1)
    assert not new.isVisible()
    assert (
        _child(dialog, QLineEdit, "inspect.entry.name").text() == "Q1.pulses.my_pulse"
    )
    presenter.assert_no_unexpected_messages()


def test_create_rejects_local_parse_errors_and_keeps_the_draft(
    session: _Session,
) -> None:
    presenter = RecordingDialogPresenter()
    dialog = _open(session, presenter)
    new = _new_dialog(dialog)
    name = _child(new, QLineEdit, "inspect.new_entry.name")
    name.clear()
    _type(name, "Q1.pulses.my_pulse")
    commit_text(_child(new, QLineEdit, "form.input.freq"), "321")
    gain = _child(new, QLineEdit, "form.input.gain")
    commit_text(gain, "0.7")
    commit_text(gain, "oops")
    before = _library(session).document()

    _button(new, "Create").click()

    assert _library(session).document() == before
    assert new.isVisible()
    assert _child(new, QLineEdit, "form.input.gain") is gain
    assert gain.text() == "oops"
    assert name.text() == "Q1.pulses.my_pulse"
    assert _child(new, QLineEdit, "form.input.freq").text() == "321"

    commit_text(gain, "0.8")
    _button(new, "Create").click()
    written = _library(session).module_spec("Q1.pulses.my_pulse")
    assert isinstance(written, PulseSpec)
    assert (written.freq, written.gain) == (321.0, 0.8)
    assert not new.isVisible()
    assert _child(dialog, QLineEdit, "inspect.entry.name").text() == name.text()


def test_new_entry_rejects_a_name_taken_in_its_family(session: _Session) -> None:
    presenter = RecordingDialogPresenter()
    dialog = _open(session, presenter)
    new = _new_dialog(dialog)
    _choose_template(new, "Flat waveform")
    name = _child(new, QLineEdit, "inspect.new_entry.name")
    name.setText(WAVEFORM)
    before = _library(session).document()
    _button(new, "Create").click()
    assert presenter.messages("warning")
    assert new.isVisible()
    assert _library(session).document() == before


@pytest.mark.parametrize(
    ("rows", "loaded"),
    [
        pytest.param((AliasRow("resonator", "Q1"),), True, id="bound"),
        pytest.param(
            (AliasRow("unrelated", ""), AliasRow("resonator", "Q1")),
            True,
            id="unrelated-unbound",
        ),
        pytest.param((), False, id="missing"),
        pytest.param((AliasRow("resonator", ""),), False, id="unbound"),
    ],
)
def test_new_entry_uses_point_aliases_and_recovers_after_missing_bindings(
    session: _Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rows: tuple[AliasRow, ...],
    loaded: bool,
) -> None:
    session.control.set_alias_rows(rows)
    templates = {
        "aliased": TemplateDef(
            PulseSpec,
            "Aliased pulse",
            {
                "ch": 0,
                "nqz": 1,
                "freq": 120.0,
                "gain": 0.3,
                "waveform": "{resonator}.waveforms.base",
            },
            default_name="aliased_pulse",
        ),
        "drive": TEMPLATES["drive"],
    }
    unhandled: list[str] = []
    monkeypatch.setattr(
        sys, "excepthook", lambda _kind, error, _tb: unhandled.append(str(error))
    )
    presenter = RecordingDialogPresenter()
    dialog = _open(session, presenter, templates=templates)
    ui = ComponentDriver(dialog)
    ui.screenshot(
        Path(os.environ.get("INSPECT_IMPLICIT_EVIDENCE", str(tmp_path)))
        / "01-inspect.png"
    )
    ui.click("New")
    ui.screenshot(
        Path(os.environ.get("INSPECT_IMPLICIT_EVIDENCE", str(tmp_path)))
        / "02-new-entry.png"
    )
    assert not unhandled, f"New must handle template bindings: {unhandled}"
    assert any(d.title == "Create ModuleLibrary entry" for d in ui.dialogs())
    new = _child(dialog, QDialog, "inspect.new_entry")
    editor = ComponentDriver(new)
    if loaded:
        assert not editor.observe(object_name="inspect.new_entry.error").visible
        assert editor.observe("Create").enabled
        assert editor.observe(object_name="form.input.waveform").text == WAVEFORM
        assert float(editor.observe(object_name="form.input.freq").text) == 120.0
    else:
        error = editor.observe(object_name="inspect.new_entry.error")
        assert error.visible
        assert error.text == (
            "template 需要綁定 {resonator}，請先在 Aliases 設定同名 alias"
        )
        assert not editor.observe("Create").enabled

    editor.choose_option(
        object_name="inspect.new_entry.template", value="Drive pulse  (module)"
    )
    editor.screenshot(
        Path(os.environ.get("INSPECT_IMPLICIT_EVIDENCE", str(tmp_path)))
        / "03-other-template.png"
    )
    assert not unhandled, f"Template switching must not escape Qt: {unhandled}"
    assert editor.observe("Create").enabled
    assert not editor.observe(object_name="inspect.new_entry.error").visible
    assert float(editor.observe(object_name="form.input.freq").text) == 200.0

    editor.choose_option(
        object_name="inspect.new_entry.template", value="Aliased pulse  (module)"
    )
    editor.screenshot(
        Path(os.environ.get("INSPECT_IMPLICIT_EVIDENCE", str(tmp_path)))
        / "04-return-to-aliased.png"
    )
    assert not unhandled
    assert editor.observe("Create").enabled == loaded
    if loaded:
        assert editor.observe(object_name="form.input.waveform").text == WAVEFORM
    else:
        assert editor.observe(object_name="inspect.new_entry.error").visible
    editor.click("Cancel")
    presenter.assert_no_unexpected_messages()


def test_arb_waveforms_button_calls_the_host_entry(session: _Session) -> None:
    dialog = _open(session, RecordingDialogPresenter())
    _button(dialog, "Arb Waveforms…").click()
    assert session.arb_opened == [True]


def test_a_synthetic_host_opens_the_shared_arb_dialog(
    qapp: QApplication, tmp_path: Path
) -> None:
    """A workflows-like host wires Inspect to session services only."""
    state = SessionState(SessionEnv(parameters=None, soc=None, soccfg=None))
    bus = BaseEventBus()
    gate = RunBlocksHardwareGate(run_kind="run", bus=bus)
    handles = OperationHandles()
    background = MagicMock()
    progress = ProgressService(MagicMock())
    services = build_session_services(
        state=state,
        bus=bus,
        operations=OperationInfrastructure(
            gate=gate,
            background=background,
            runner=OperationRunner(gate, handles, progress, background, bus),
            handles=handles,
        ),
        progress=progress,
        bind_assets=ArbWaveformDatabase.info_at,
        project_root=str(tmp_path),
    )
    host = QWidget()
    opened: list[ArbWaveformDialog] = []

    def open_arb_waveforms() -> None:
        arb = ArbWaveformDialog(services.arb_waveform, parent=host)
        opened.append(arb)
        arb.open()

    open_entry: Callable[[], None] = open_arb_waveforms
    dialog = InspectDialog(
        services.context_control,
        bus,
        templates=TEMPLATES,
        arb_waveforms=services.arb_waveform,
        open_arb_waveforms=open_entry,
        dialog_presenter=RecordingDialogPresenter(),
        parent=host,
    )
    try:
        _button(dialog, "Arb Waveforms…").click()
        assert len(opened) == 1 and opened[0].isVisible()
        assert host.findChild(ArbWaveformDialog) is opened[0]
    finally:
        for arb in opened:
            arb.close()
        dialog.close()
        host.deleteLater()
        qapp.processEvents()
