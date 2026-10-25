"""Instrument suggestions switch through the visible field's context menu."""

from pathlib import Path
from typing import Annotated

import pytest
from zcu_tools.cfg_model import ConfigBase, DeviceName
from zcu_tools.device import BaseDeviceInfo
from zcu_tools.gui.form import FormEnv, FormState, InputPolicy
from zcu_tools.gui.form_adapter import Form
from zcu_tools.gui.widgets.form import FormEditor

from tests.gui._user_input import ComponentDriver


class InstrumentInput(ConfigBase):
    flux_device: Annotated[DeviceName, InputPolicy(expression=False)] = "offline-coil"


@pytest.fixture
def instrument_editor(qapp):
    env = FormEnv(
        devices={
            "coil": BaseDeviceInfo(address="offline", type="fake"),
            "pump": BaseDeviceInfo(address="offline", type="fake"),
        }
    )
    form = Form(InstrumentInput, {})
    editor = FormEditor(FormState(form.default_input(env)), env)
    editor.resize(620, 180)
    editor.show()
    qapp.processEvents()
    try:
        yield ComponentDriver(editor)
    finally:
        editor.close()
        editor.deleteLater()
        qapp.processEvents()


def test_device_name_list_text_and_unlisted_value(instrument_editor, tmp_path: Path):
    driver = instrument_editor
    control = "form.input.flux_device"
    assert "offline-coil" in driver.observe(object_name=control).text
    driver.screenshot(tmp_path / "01-unlisted-current.png")
    driver.choose_option(object_name=control, value="coil")
    assert driver.observe(object_name=control).text == "coil"
    driver.screenshot(tmp_path / "02-selected.png")
    driver.invoke_context_action(object_name=control, action="Edit as text")
    driver.screenshot(tmp_path / "03-text-mode.png")
    driver.enter_text(object_name=control, value="new instrument", commit=False)
    assert driver.observe(object_name=control).text == "new instrument"
    driver.screenshot(tmp_path / "04-arbitrary-text.png")
    driver.invoke_context_action(object_name=control, action="Choose from list")
    assert "new instrument" in driver.observe(object_name=control).text
    driver.screenshot(tmp_path / "05-back-to-list.png")
    driver.choose_option(object_name=control, value="pump")
    assert driver.observe(object_name=control).text == "pump"
    driver.screenshot(tmp_path / "06-selected-pump.png")
