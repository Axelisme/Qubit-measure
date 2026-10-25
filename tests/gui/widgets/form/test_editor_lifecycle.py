"""A host may detach its FormEditor while handling a committed user edit."""

import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import TracebackType

import pytest
from pydantic import JsonValue
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget
from zcu_tools.gui.form import FormEnv, FormModel, FormState, FormWritePath
from zcu_tools.gui.widgets.form import FormEditor

from tests.gui._user_input import ComponentDriver


class AmountForm(FormModel):
    amount: float = 1.0


@pytest.fixture
def detaching_host(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[ComponentDriver, ComponentDriver]]:
    """Show a replacement when the host accepts the old editor's last edit."""
    env = FormEnv()
    state = FormState(AmountForm.default_input(env))
    replacement = QWidget()
    layout = QVBoxLayout(replacement)
    status = QLabel("Waiting for edit", replacement)
    status.setObjectName("lifecycle.status")
    layout.addWidget(status)
    layout.addWidget(FormEditor(FormState(AmountForm.default_input(env)), env))

    def write(_batch: Mapping[FormWritePath, JsonValue]) -> None:
        editor.detach()
        status.setText("Detached without late callbacks")
        replacement.show()
        replacement.activateWindow()

    # Report failures at Qt's exception boundary instead of letting PyQt abort.
    def exception(
        _kind: type[BaseException], error: BaseException, _trace: TracebackType | None
    ) -> None:
        status.setText(f"Callback error: {error}")

    monkeypatch.setattr(sys, "excepthook", exception)
    editor = FormEditor(
        state,
        env,
        write=write,
        select=lambda path, value: state.select(path, value, env),
        run_command=lambda path, name: state.run_command(path, name, env),
    )
    editor.resize(700, 300)
    replacement.resize(700, 300)
    editor.show()
    editor.activateWindow()
    qapp.processEvents()
    yield ComponentDriver(editor), ComponentDriver(replacement)
    editor.close()
    replacement.close()
    editor.deleteLater()
    replacement.deleteLater()
    qapp.processEvents()


@pytest.mark.parametrize("pending", ["7", "invalid"])
def test_focus_loss_from_host_detach_discards_pending_text(
    qapp: QApplication, tmp_path: Path, pending: str
) -> None:
    env = FormEnv()
    state = FormState(AmountForm.default_input(env))
    host = QWidget()
    layout = QVBoxLayout(host)
    status = QLabel("No commits", host)
    status.setObjectName("lifetime.status")

    def write(batch: Mapping[FormWritePath, JsonValue]) -> None:
        state.write(batch, env)
        status.setText("Committed")

    editor = FormEditor(
        state,
        env,
        write=write,
        select=lambda path, value: state.select(path, value, env),
        run_command=lambda path, name: state.run_command(path, name, env),
    )
    detach = QPushButton("Detach", host)
    # The host retires its form before disabling it causes FocusOut.
    detach.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    detach.clicked.connect(editor.detach)
    layout.addWidget(editor)
    layout.addWidget(detach)
    layout.addWidget(status)
    host.resize(700, 300)
    host.show()
    host.activateWindow()
    qapp.processEvents()
    driver = ComponentDriver(host)
    try:
        driver.enter_text(object_name="form.input.amount", value=pending, commit=False)
        driver.screenshot(tmp_path / "01-pending-text.png")
        driver.click("Detach")
        driver.screenshot(tmp_path / "02-after-focus-out.png")
        assert not driver.observe(object_name="form.input.amount").enabled
        assert driver.observe(object_name="lifetime.status").text == "No commits"
    finally:
        editor.detach()
        host.close()
        host.deleteLater()
        qapp.processEvents()


def test_host_detach_during_commit_leaves_the_replacement_usable(
    detaching_host: tuple[ComponentDriver, ComponentDriver], tmp_path: Path
) -> None:
    old, replacement = detaching_host
    old.screenshot(tmp_path / "01-active-editor.png")
    old.enter_text(object_name="form.input.amount", value="7")
    old.screenshot(tmp_path / "02-detached-editor.png")
    replacement.screenshot(tmp_path / "03-replacement.png")
    assert not old.observe(object_name="form.input.amount").enabled
    assert replacement.observe(object_name="lifecycle.status").text == (
        "Detached without late callbacks"
    )
    replacement.enter_text(object_name="form.input.amount", value="9")
    replacement.screenshot(tmp_path / "04-replacement-edited.png")
    assert replacement.observe(object_name="form.input.amount").text == "9"
