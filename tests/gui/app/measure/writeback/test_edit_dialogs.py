"""Writeback editing feedback through the ledger's public Qt surface."""

import os
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from qtpy.QtCore import QCoreApplication, QEvent, Qt
from qtpy.QtGui import QColor, QPalette
from qtpy.QtWidgets import QDialog, QLabel, QLineEdit, QMessageBox, QPushButton
from zcu_tools.gui.app.measure.writeback import ParameterWriteback, WritebackValues
from zcu_tools.gui.app.measure.writeback.widget import WritebackWidget
from zcu_tools.gui.widgets import error_colour

from tests.gui._user_input import ComponentDriver


def button(parent, text: str) -> QPushButton:
    return next(b for b in parent.findChildren(QPushButton) if b.text() == text)


def label_text(parent) -> str:
    return "\n".join(label.text() for label in parent.findChildren(QLabel))


@pytest.fixture
def value_editor(
    qapp, monkeypatch, request
) -> Iterator[
    tuple[
        WritebackWidget,
        QDialog,
        ParameterWriteback,
        MagicMock,
        list[tuple[object, ...]],
    ]
]:
    item = ParameterWriteback(
        target_name="$resonator.freq",
        description="Resonator frequency",
        proposed_value=getattr(request, "param", 6012.345678),
        stderr=0.0213,
    )
    item.session_id = "param-1"
    ctrl = MagicMock()
    ctrl.writeback_control.get_writeback_values_for_pane.side_effect = lambda *a, **k: {
        item.session_id: WritebackValues(
            actual_path="Q1.freq",
            current=6011.9,
            proposed=item.proposed_value,
            unit="MHz",
        )
    }
    modal_messages = []
    monkeypatch.setattr(
        QMessageBox, "critical", lambda *args: modal_messages.append(args)
    )
    widget = WritebackWidget(ctrl, tab_id="tab-1", pane="analysis")
    widget.populate([item], ctrl.writeback_control.get_writeback_values_for_pane())
    widget.show()
    button(widget, "Edit").click()
    dialog = next(d for d in widget.findChildren(QDialog) if d.isVisible())
    yield widget, dialog, item, ctrl, modal_messages
    dialog.close()
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qapp.processEvents()


def test_value_context_and_live_validation(value_editor, qapp):
    _, dialog, _, ctrl, messages = value_editor
    texts = label_text(dialog)
    assert "Resonator frequency" in texts
    assert "Q1.freq" in texts and "Writes to" in texts
    assert "6011.9 MHz" in texts
    assert "0.0213 MHz" in texts
    edits = dialog.findChildren(QLineEdit)
    name = next(e for e in edits if e.text() == "$resonator.freq")
    value = next(e for e in edits if e.text() == "6012.345678")
    value.setText("not a number")
    assert "Enter a number" in label_text(dialog)
    assert not button(dialog, "Save").isEnabled()
    assert not ctrl.writeback_control.set_writeback_item_for_pane.called
    error = next(
        label
        for label in dialog.findChildren(QLabel)
        if "Enter a number" in label.text()
    )
    palette = dialog.palette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#353535"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#eeeeee"))
    dialog.setPalette(palette)
    qapp.processEvents()
    assert error.palette().color(QPalette.ColorRole.WindowText) == error_colour(dialog)
    value.setText("6013 MHz")
    name.setText(" ")
    assert "Apply-as name must not be empty" in label_text(dialog)
    assert not button(dialog, "Save").isEnabled()
    name.setText("$resonator.freq")
    assert button(dialog, "Save").isEnabled()
    assert messages == []


def test_value_matching_unit_save_and_cancel(value_editor):
    _, dialog, item, ctrl, messages = value_editor
    value = next(e for e in dialog.findChildren(QLineEdit) if e.text() == "6012.345678")
    value.setText("6013.25 MHz")
    button(dialog, "Save").click()
    assert not dialog.isVisible()
    assert item.proposed_value == 6013.25
    ctrl.writeback_control.set_writeback_item_for_pane.assert_called_once_with(
        "tab-1",
        "analysis",
        "param-1",
        target_name="$resonator.freq",
        proposed_value=6013.25,
    )
    assert messages == []


def test_numeric_unit_suffix_requires_digit_or_whitespace_boundary(value_editor):
    _, dialog, item, _, _ = value_editor
    value = dialog.findChild(QLineEdit, "writebackValue")
    value.setText("nanMHz")
    assert not button(dialog, "Save").isEnabled()
    value.setText("6012.3 kMHz")
    assert not button(dialog, "Save").isEnabled()
    for text in ("6012.3 MHz", "6012.3MHz"):
        value.setText(text)
        assert button(dialog, "Save").isEnabled()
    button(dialog, "Save").click()
    assert item.proposed_value == 6012.3


@pytest.mark.parametrize("value_editor", ["abcMHz"], indirect=True)
def test_string_value_keeps_matching_unit_suffix(value_editor):
    _, dialog, item, ctrl, _ = value_editor
    button(dialog, "Save").click()
    assert item.proposed_value == "abcMHz"
    assert (
        ctrl.writeback_control.set_writeback_item_for_pane.call_args.kwargs[
            "proposed_value"
        ]
        == "abcMHz"
    )


def test_value_controller_refusal_stays_inline(value_editor):
    _, dialog, item, ctrl, messages = value_editor
    ctrl.writeback_control.set_writeback_item_for_pane.side_effect = RuntimeError(
        "Tab is busy"
    )
    button(dialog, "Save").click()
    assert "Tab is busy" in label_text(dialog)
    assert dialog.isVisible()
    assert item.proposed_value == 6012.345678
    assert messages == []


@pytest.mark.parametrize(
    ("original", "invalid", "message"),
    [
        (1, "1.2", "Enter an integer"),
        (True, "maybe", "Enter true or false"),
        (1 + 2j, "1+2i", "Enter a complex number"),
    ],
)
def test_value_type_specific_validation(qapp, original, invalid, message):
    item = ParameterWriteback(
        target_name="value", description="Typed value", proposed_value=original
    )
    item.session_id = "param-1"
    widget = WritebackWidget(MagicMock(), tab_id="tab-1")
    widget.populate(
        [item],
        {
            "param-1": WritebackValues(
                actual_path="value", current=None, proposed=original, unit=None
            )
        },
    )
    button(widget, "Edit").click()
    dialog = next(d for d in widget.findChildren(QDialog) if d.isVisible())
    try:
        value = next(
            e for e in dialog.findChildren(QLineEdit) if e.text() == str(original)
        )
        value.setText(invalid)
        assert message in label_text(dialog)
        assert not button(dialog, "Save").isEnabled()
    finally:
        dialog.close()
        widget.deleteLater()
        qapp.processEvents()


def test_module_leaf_changes_include_additions_removals_and_nested_fields():
    from zcu_tools.gui.app.measure.writeback import edit_dialogs

    changes = edit_dialogs.module_changes(
        {"freq": 6000.0, "waveform": {"length": 1.8}, "removed": 2, "same": None},
        {"freq": 6012.0, "waveform": {"length": 2.0}, "added": 3, "same": None},
    )
    assert {c.path: (c.current, c.proposed) for c in changes} == {
        "freq": ("6000.0", "6012.0"),
        "waveform.length": ("1.8", "2.0"),
        "removed": ("2", "—"),
        "added": ("—", "3"),
    }
    assert edit_dialogs.module_changes({"freq": 1}, {"freq": 1}) == []


@pytest.fixture
def module_editor(qapp, monkeypatch):
    from zcu_tools.cfg_model import ConstWaveformCfg, PulseSpec
    from zcu_tools.gui.app.measure.writeback import ModuleWriteback
    from zcu_tools.gui.session.library_entry import ModuleEntryForm

    from tests.gui.app.measure._cfg_fakes import parameter_capture

    spec = PulseSpec(
        ch=0,
        nqz=1,
        freq=6001.0,
        phase=0.0,
        gain=0.2,
        waveform=ConstWaveformCfg(length=2.0),
    )
    form = ModuleEntryForm(spec, parameter_capture(), arb_names=())
    item = ModuleWriteback(
        target_name="pulse", description="Readout pulse", proposal=spec
    )
    item.session_id = "ml-1"
    current = spec.model_dump(mode="python")
    current["freq"] = 6000.0
    ctrl = MagicMock()
    ctrl.writeback_control.get_writeback_item_form_for_pane.return_value = form

    def preview(*args, **kwargs):
        return {
            item.session_id: WritebackValues(
                actual_path=item.target_name,
                current=current,
                proposed=form.spec().model_dump(mode="python"),
                unit=None,
            )
        }

    def set_item(*args, **kwargs):
        item.target_name = kwargs.get("target_name", item.target_name)

    ctrl.writeback_control.set_writeback_item_for_pane.side_effect = set_item
    ctrl.writeback_control.get_writeback_values_for_pane.side_effect = preview
    messages = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a: messages.append(a))
    widget = WritebackWidget(ctrl, tab_id="tab-1", pane="analysis")
    widget.populate([item], preview())
    widget.show()
    button(widget, "Edit").click()
    dialog = next(d for d in widget.findChildren(QDialog) if d.isVisible())
    yield widget, dialog, item, ctrl, form, current, messages
    dialog.close()
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qapp.processEvents()


@pytest.fixture
def module_driver(module_editor, tmp_path):
    widget, dialog, *_ = module_editor
    driver = ComponentDriver(dialog)
    driver.resize_window(700, 700)
    evidence = Path(os.environ.get("WRITEBACK_UI_EVIDENCE", str(tmp_path)))
    return driver, ComponentDriver(widget), evidence


@pytest.mark.parametrize("finish", ["cancel", "escape", "close", "detach"])
def test_module_cancel_discards_content_and_name(module_editor, module_driver, finish):
    widget, _, item, _, form, _, messages = module_editor
    driver, ledger, evidence = module_driver
    prefix = evidence / f"cancel-{finish}"
    driver.screenshot(prefix / "01-open.png")
    driver.enter_text(object_name="form.input.freq", value="6002.0")
    driver.screenshot(prefix / "02-frequency.png")
    driver.enter_text(object_name="writebackApplyAs", value="tuned")
    driver.screenshot(prefix / "03-name.png")
    assert form.spec().freq == 6001.0
    assert item.target_name == "pulse"
    assert "freq  6000.0 → 6002.0" in driver.observe(
        object_name="writebackModuleChanges"
    ).text
    if finish == "cancel":
        driver.click("Cancel")
    elif finish == "escape":
        driver.press_key(Qt.Key.Key_Escape)
    elif finish == "close":
        driver.close_window()
    else:
        widget.detach()
    ledger.screenshot(prefix / "04-finished.png")
    assert not ledger.dialogs()
    assert form.spec().freq == 6001.0
    assert item.target_name == "pulse"
    assert messages == []


def test_module_save_installs_content_then_name(module_editor, module_driver):
    _, _, item, ctrl, form, _, _ = module_editor
    driver, ledger, evidence = module_driver
    driver.enter_text(object_name="form.input.freq", value="6002.0")
    driver.screenshot(evidence / "save/01-frequency.png")
    driver.enter_text(object_name="writebackApplyAs", value="tuned")
    driver.screenshot(evidence / "save/02-name.png")
    assert form.spec().freq == 6001.0
    assert item.target_name == "pulse"

    def retarget(*args, **kwargs):
        assert form.spec().freq == 6002.0
        item.target_name = kwargs["target_name"]

    ctrl.writeback_control.set_writeback_item_for_pane.side_effect = retarget
    driver.click("Save")
    ledger.screenshot(evidence / "save/03-saved.png")
    assert not ledger.dialogs()
    assert form.spec().freq == 6002.0
    assert item.target_name == "tuned"


@pytest.mark.parametrize("invalid", ["abc", "nan"])
def test_module_invalid_copy_disables_save(module_editor, module_driver, invalid):
    _, _, _, _, form, _, _ = module_editor
    driver, _, evidence = module_driver
    driver.enter_text(object_name="form.input.freq", value=invalid)
    driver.screenshot(evidence / f"invalid-{invalid}/01-error.png")
    assert not driver.observe("Save").enabled
    assert "freq:" in driver.observe(object_name="writebackEditError").text
    assert form.spec().freq == 6001.0
    driver.enter_text(object_name="form.input.freq", value="6002.0")
    driver.screenshot(evidence / f"invalid-{invalid}/02-repaired.png")
    assert driver.observe("Save").enabled


@pytest.mark.parametrize("refusal", ["admission", "input"])
def test_module_install_refusal_keeps_original_draft(
    module_editor, module_driver, monkeypatch, refusal
):
    from zcu_tools.gui.expected_error import FailedPreconditionError
    from zcu_tools.gui.form import FormInputRejected

    _, _, item, ctrl, form, _, _ = module_editor
    driver, ledger, evidence = module_driver
    driver.enter_text(object_name="form.input.freq", value="6002.0")
    driver.screenshot(evidence / f"refusal-{refusal}/01-frequency.png")
    driver.enter_text(object_name="writebackApplyAs", value="tuned")
    driver.screenshot(evidence / f"refusal-{refusal}/02-name.png")
    if refusal == "admission":
        ctrl.writeback_control.get_writeback_item_form_for_pane.side_effect = (
            FailedPreconditionError("Tab is busy")
        )
    else:
        def reject_input(*args):
            raise FormInputRejected("Input rejected")

        monkeypatch.setattr(form.state, "install_input", reject_input)
    driver.click("Save")
    driver.screenshot(evidence / f"refusal-{refusal}/03-refused.png")
    assert ledger.dialogs()
    assert driver.observe(object_name="writebackEditError").text
    assert form.spec().freq == 6001.0
    assert item.target_name == "pulse"
    ctrl.writeback_control.set_writeback_item_for_pane.assert_not_called()


def test_missing_library_entry_counts_only_proposed_leaves():
    from zcu_tools.gui.app.measure.writeback import edit_dialogs

    changes = edit_dialogs.module_changes(None, {"freq": 6012.0, "gain": 0.2})
    assert [change.path for change in changes] == ["freq", "gain"]
