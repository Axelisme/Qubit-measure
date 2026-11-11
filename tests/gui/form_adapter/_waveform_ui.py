"""Replay a standalone-envelope editor using visible keyboard and mouse input."""

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from qtpy.QtCore import QCoreApplication, QEvent
from qtpy.QtWidgets import QApplication, QVBoxLayout, QWidget
from zcu_tools.gui.form import FormState
from zcu_tools.gui.widgets.form import FormEditor

from tests.gui._user_input import ComponentDriver

from ._waveform_models import envelope_env, envelope_form


@contextmanager
def envelope_editor(app: QApplication) -> Generator[ComponentDriver]:
    """Own a source-backed synthetic editor and detach it before Qt deletion."""
    form, env = envelope_form(), envelope_env()
    window = QWidget()
    window.setWindowTitle("Independent envelope inputs")
    layout = QVBoxLayout(window)
    editor = FormEditor(FormState(form.default_input(env)), env, window)
    layout.addWidget(editor)
    window.resize(1000, 750)
    window.show()
    app.processEvents()
    try:
        yield ComponentDriver(window)
    finally:
        editor.detach()
        window.close()
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


def run_scenario(app: QApplication, output: Path) -> None:
    """Select a library envelope, edit another shape and retain independent values."""
    output.mkdir(parents=True, exist_ok=True)
    with envelope_editor(app) as driver:
        driver.screenshot(output / "01-initial.png")
        assert driver.observe(object_name="form.input.first").text == "Inline: gauss"
        assert driver.observe(object_name="form.input.second").text == "Inline: gauss"
        driver.choose_option(object_name="form.input.first", value="Q.waveforms.shaped")
        driver.screenshot(output / "02-first-library.png")
        assert (
            driver.observe(object_name="form.input.first").text == "Q.waveforms.shaped"
        )
        driver.choose_option(object_name="form.input.second", value="Inline: cosine")
        driver.screenshot(output / "03-second-cosine.png")
        driver.click(object_name="form.fold.second")
        driver.screenshot(output / "04-second-expanded.png")
        driver.enter_text(object_name="form.input.second.module.length", value="0.24")
        driver.screenshot(output / "05-second-length.png")
        assert (
            float(driver.observe(object_name="form.input.second.module.length").text)
            == 0.24
        )
        assert (
            driver.observe(object_name="form.input.first").text == "Q.waveforms.shaped"
        )
        driver.choose_option(object_name="form.input.first", value="Inline: drag")
        driver.screenshot(output / "06-first-drag.png")
        driver.click(object_name="form.fold.first")
        driver.screenshot(output / "07-first-expanded.png")
        assert (
            float(driver.observe(object_name="form.input.first.module.length").text)
            == 0.48
        )
        assert (
            float(driver.observe(object_name="form.input.first.module.sigma").text)
            == 0.08
        )
        assert (
            float(driver.observe(object_name="form.input.second.module.length").text)
            == 0.24
        )
