"""Writeback editing feedback through the ledger's public Qt surface."""

from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest
from qtpy.QtCore import QCoreApplication, QEvent
from qtpy.QtGui import QColor, QPalette
from qtpy.QtWidgets import QDialog, QLabel, QLineEdit, QMessageBox, QPushButton
from zcu_tools.gui.app.measure.writeback import ParameterWriteback, WritebackValues
from zcu_tools.gui.app.measure.writeback.widget import WritebackWidget
from zcu_tools.gui.widgets import error_colour


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


def test_module_context_changes_follow_shared_state(module_editor):
    _, dialog, _, _, form, current, _ = module_editor
    texts = label_text(dialog)
    assert "Readout pulse" in texts
    assert "Changes go to the writeback draft at once" in texts
    assert "only written by Apply Selected" in texts
    assert "Writes to library" in texts
    assert "Changes from library (1)" in texts
    assert "freq  6000.0 → 6001.0" in texts
    form.state.write({("gain",): 0.3}, form.env)
    assert "Changes from library (2)" in label_text(dialog)
    assert "gain  0.2 → 0.3" in label_text(dialog)
    current.clear()
    # The preview contract uses None, not an empty dict, for missing entries.
    ctrl = module_editor[3]
    ctrl.writeback_control.get_writeback_values_for_pane.side_effect = lambda *a, **k: {
        "ml-1": WritebackValues(
            actual_path="pulse",
            current=None,
            proposed=form.spec().model_dump(mode="python"),
            unit=None,
        )
    }
    form.state.write({("gain",): 0.4}, form.env)
    assert "New library entry" in label_text(dialog)


def test_module_errors_list_path_and_discard_warning(module_editor):
    from tests.gui.widgets.form._text_input import commit_text_on_focus_loss

    _, dialog, _, _, _, _, messages = module_editor
    assert "Changes from library" in label_text(dialog)
    freq = dialog.findChild(QLineEdit, "form.input.freq")
    commit_text_on_focus_loss(freq, "abc")
    texts = label_text(dialog)
    assert "freq:" in texts and "abc" in texts
    assert "Invalid text is discarded when you press Done" in texts
    assert "Unavailable while the configuration is invalid" in texts
    error = dialog.findChild(QLabel, "writebackEditError")
    assert error.isVisible()
    assert error.palette().color(QPalette.ColorRole.WindowText) == error_colour(dialog)
    commit_text_on_focus_loss(freq, "6002.0")
    assert not error.isVisible()
    assert "freq  6000.0 → 6002.0" in label_text(dialog)
    assert messages == []


@pytest.mark.parametrize("finish", ["done", "escape", "close"])
def test_module_finish_commits_name_and_drops_only_local_text(module_editor, finish):
    from qtpy.QtCore import Qt

    from tests.gui._user_input import key_click
    from tests.gui.widgets.form._text_input import commit_text_on_focus_loss

    _, dialog, item, _, form, _, messages = module_editor
    assert "Done" in [b.text() for b in dialog.findChildren(QPushButton)]
    freq = dialog.findChild(QLineEdit, "form.input.freq")
    commit_text_on_focus_loss(freq, "abc")
    name = next(e for e in dialog.findChildren(QLineEdit) if e.text() == "pulse")
    name.setText("tuned")
    if finish == "done":
        button(dialog, "Done").click()
    elif finish == "escape":
        key_click(dialog, Qt.Key.Key_Escape)
    else:
        dialog.close()
    assert not dialog.isVisible()
    assert item.target_name == "tuned"
    assert form.spec().freq == 6001.0
    assert messages == []


def test_module_target_refusal_is_inline(module_editor):
    _, dialog, item, ctrl, _, _, messages = module_editor
    assert "Changes go to the writeback draft at once" in label_text(dialog)
    ctrl.writeback_control.set_writeback_item_for_pane.side_effect = RuntimeError(
        "Tab is busy"
    )
    name = next(e for e in dialog.findChildren(QLineEdit) if e.text() == "pulse")
    name.setText("tuned")
    name.editingFinished.emit()
    assert "Tab is busy" in label_text(dialog)
    assert dialog.isVisible()
    assert item.target_name == "pulse"
    assert messages == []
    ctrl.writeback_control.set_writeback_item_for_pane.side_effect = None


def test_module_ledger_summarizes_changes_and_exposes_full_tooltip(module_editor):
    widget, _, _, _, form, _, _ = module_editor
    current = widget.findChild(QLabel, "writebackCurrent")
    proposed = widget.findChild(QLabel, "writebackProposed")
    assert current.text() == "library entry"
    assert proposed.text() == "1 change"
    assert "freq  6000.0 → 6001.0" in proposed.toolTip()
    form.state.write({("freq",): 6000.0}, form.env)
    assert proposed.text() == "no changes"
    form.state.write({("gain",): 0.4}, form.env)
    assert proposed.text() == "1 change"
    assert "gain  0.2 → 0.4" in proposed.toolTip()


def test_missing_library_entry_counts_only_proposed_leaves():
    from zcu_tools.gui.app.measure.writeback import edit_dialogs

    changes = edit_dialogs.module_changes(None, {"freq": 6012.0, "gain": 0.2})
    assert [change.path for change in changes] == ["freq", "gain"]


def test_invalid_module_ledger_explains_apply_gate_and_recovers(module_editor, qapp):
    from qtpy.QtWidgets import QCheckBox
    from zcu_tools.gui.widgets import warning_colour

    from tests.gui.widgets.form._text_input import commit_text_on_focus_loss

    widget, dialog, item, _, _, _, _ = module_editor
    freq = dialog.findChild(QLineEdit, "form.input.freq")
    commit_text_on_focus_loss(freq, "abc")
    proposed = widget.findChild(QLabel, "writebackProposed")
    assert proposed.text() == "Invalid configuration"
    assert proposed.palette().color(QPalette.ColorRole.WindowText) == error_colour(
        widget
    )
    hint = next(
        label
        for label in widget.findChildren(QLabel)
        if "Fix or uncheck invalid module" in label.text()
    )
    assert item.target_name in hint.text()
    assert hint.palette().color(QPalette.ColorRole.WindowText) == warning_colour(widget)
    assert not button(widget, "Apply Selected").isEnabled()
    palette = widget.palette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#353535"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#eeeeee"))
    widget.setPalette(palette)
    qapp.processEvents()
    assert proposed.palette().color(QPalette.ColorRole.WindowText) == error_colour(
        widget
    )
    checkbox = widget.findChild(QCheckBox)
    checkbox.setChecked(False)
    assert "Select the items to write back" in hint.text()
    checkbox.setChecked(True)
    commit_text_on_focus_loss(freq, "6002.0")
    assert proposed.text() == "1 change"
    assert "Select the items to write back" in hint.text()
    assert button(widget, "Apply Selected").isEnabled()


def test_module_error_list_tracks_edits_while_already_invalid(module_editor, qapp):
    from tests.gui.widgets.form._text_input import commit_text_on_focus_loss

    _, dialog, _, _, _, _, _ = module_editor
    error = dialog.findChild(QLabel, "writebackEditError")
    freq = dialog.findChild(QLineEdit, "form.input.freq")
    gain = dialog.findChild(QLineEdit, "form.input.gain")
    commit_text_on_focus_loss(freq, "bad-frequency")
    qapp.processEvents()
    commit_text_on_focus_loss(gain, "bad-gain")
    qapp.processEvents()
    assert "freq:" in error.text() and "gain:" in error.text()
    commit_text_on_focus_loss(freq, "6002.0")
    qapp.processEvents()
    assert "freq:" not in error.text()
    assert "gain:" in error.text()
