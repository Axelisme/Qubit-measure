"""Observable decisions and labels of the Qt dialog presenter."""

import pytest
from qtpy.QtCore import QTimer, QTranslator
from qtpy.QtWidgets import QMessageBox, QWidget
from zcu_tools.gui.widgets.dialog_presenter import (
    DestructiveConfirmation,
    QtDialogPresenter,
)


class _DiscardTranslation(QTranslator):
    """Mimic a platform whose standard Discard label is longer."""

    def isEmpty(self) -> bool:
        return False

    def translate(
        self,
        context: str | None,
        sourceText: str | None,
        disambiguation: str | None = None,
        n: int = -1,
    ) -> str:
        if context == "QPlatformTheme" and sourceText in ("Discard", "&Discard"):
            return "Close without Saving"
        return ""


@pytest.fixture
def translated_discard(qapp):
    translator = _DiscardTranslation()
    qapp.installTranslator(translator)
    try:
        reference = QMessageBox()
        button = reference.addButton(QMessageBox.StandardButton.Discard)
        assert button is not None
        assert button.text() == "Close without Saving"
        reference.deleteLater()
        yield
    finally:
        qapp.removeTranslator(translator)
        qapp.processEvents()


@pytest.mark.parametrize(
    ("decision", "expected"),
    [
        (QMessageBox.StandardButton.Save, "save"),
        (QMessageBox.StandardButton.Discard, "discard"),
        (QMessageBox.StandardButton.Cancel, "cancel"),
        (None, "cancel"),
    ],
    ids=["save", "discard", "cancel", "dismiss"],
)
def test_unsaved_choice_has_explicit_discard_label_and_preserves_decisions(
    qapp, translated_discard, decision, expected
):
    parent = QWidget()
    observed: list[tuple[str, bool]] = []

    def answer() -> None:
        box = qapp.activeModalWidget()
        try:
            assert isinstance(box, QMessageBox)
            discard = box.button(QMessageBox.StandardButton.Discard)
            assert discard is not None
            observed.append(
                (
                    discard.text(),
                    box.defaultButton()
                    is box.button(QMessageBox.StandardButton.Cancel),
                )
            )
            if decision is None:
                box.reject()
            else:
                button = box.button(decision)
                assert button is not None
                button.click()
        finally:
            # Failed callback assertions must also end the nested exec() loop.
            if box is not None and box.isVisible():
                box.close()

    try:
        QTimer.singleShot(0, answer)
        result = QtDialogPresenter().save_discard_cancel(
            parent, "Unsaved changes", "Save your changes?"
        )
        assert observed == [("Discard", True)]
        assert result == expected
    finally:
        parent.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize("decision", ["action", "retry", "dismiss"])
def test_destructive_choice_only_accepts_explicit_action(qapp, decision):
    parent = QWidget()
    presenter = QtDialogPresenter()
    decisions: list[bool] = []
    try:
        presenter.destructive_confirm(
            parent,
            DestructiveConfirmation(
                title="Background work",
                message="Work is still running.",
                action_text="Force close",
                cancel_text="Wait again",
            ),
            on_decision=decisions.append,
        )
        box = parent.findChild(QMessageBox)
        assert box is not None
        buttons = {button.text(): button for button in box.buttons()}
        assert set(buttons) == {"Force close", "Wait again"}
        assert box.defaultButton() is buttons["Wait again"]
        assert decisions == []
        if decision == "dismiss":
            box.reject()
        else:
            buttons["Force close" if decision == "action" else "Wait again"].click()
        qapp.processEvents()
        assert decisions == [decision == "action"]
    finally:
        parent.deleteLater()
        qapp.processEvents()
