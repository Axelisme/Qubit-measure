"""User-visible module and waveform switching shares the adapter's raw policy."""

import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

import pytest
from pydantic import create_model
from qtpy.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget
from zcu_tools.cfg_model import ModuleSpec
from zcu_tools.gui.form import (
    FormEnv,
    FormInput,
    FormModel,
    FormPath,
    FormState,
    InputConverter,
    Option,
)
from zcu_tools.gui.form_adapter import ModuleInputField
from zcu_tools.gui.widgets.form import FormEditor

from tests.gui._user_input import ComponentDriver


@pytest.fixture
def inheritance_host(qapp: QApplication):
    field = ModuleInputField(ModuleSpec)

    class Consumer(FormModel):
        @classmethod
        def options(cls, env: FormEnv) -> Mapping[FormPath, tuple[Option, ...]]:
            return {("selection",): field.options(env)}

        @classmethod
        def converters(cls, env: FormEnv) -> Mapping[FormPath, InputConverter]:
            return {("selection",): field.converter(env)}

    model = create_model(
        "Selection", __base__=Consumer, selection=(field.annotation, ...)
    )
    env = FormEnv()
    state = FormState(
        FormInput(
            model,
            {
                "selection": {
                    "kind": "inline",
                    "module": {
                        "type": "pulse",
                        "ch": 1,
                        "nqz": 2,
                        "freq": 103,
                        "gain": 0.3,
                        "waveform": {
                            "kind": "inline",
                            "module": {"style": "gauss", "length": 0.8, "sigma": 0.2},
                        },
                    },
                }
            },
        )
    )
    host = QWidget()
    host.resize(820, 1100)
    layout = QVBoxLayout(host)
    status = QLabel("Editable")
    status.setObjectName("synthetic.status")
    layout.addWidget(status)
    editor = FormEditor(state, env, host)
    layout.addWidget(editor)
    for label in ("Readonly", "Busy", "Editable"):
        button = QPushButton(label)
        button.clicked.connect(
            lambda _checked=False, mode=label: (
                editor.set_editing_enabled(enabled=mode == "Editable"),
                status.setText(mode),
            )
        )
        layout.addWidget(button)
    host.show()
    qapp.processEvents()
    try:
        yield ComponentDriver(host)
    finally:
        host.close()
        host.deleteLater()
        qapp.processEvents()


def snapshot(driver: ComponentDriver, path: Path, name: str) -> None:
    driver.screenshot(path / f"{name}.png")
    (path / f"{name}.json").write_text(
        json.dumps(asdict(driver.visible()), indent=2), encoding="utf-8"
    )


def test_module_and_nested_waveform_switch_show_inherited_values(
    inheritance_host: ComponentDriver, tmp_path: Path
):
    driver = inheritance_host
    waveform = "form.input.selection.module.waveform"
    length = waveform + ".module.length"
    snapshot(driver, tmp_path, "01-pulse-gauss")
    driver.choose_option(object_name=waveform, value="Inline: drag")
    snapshot(driver, tmp_path, "02-drag")
    assert driver.observe(object_name=length).text == "0.8"
    assert driver.observe(object_name=waveform + ".module.sigma").text == "0.2"
    assert driver.observe(object_name=waveform + ".module.alpha").text == ""
    driver.choose_option(object_name=waveform, value="Inline: flat_top")
    snapshot(driver, tmp_path, "03-flat-top")
    assert driver.observe(object_name=length).text == "0.8"
    assert driver.observe(object_name=waveform + ".module.raise_waveform").text == ""
    driver.choose_option(object_name="form.input.selection", value="Inline: readout/pulse")
    snapshot(driver, tmp_path, "04-readout")
    assert driver.observe(object_name="form.input.selection.module.freq").text == "103.0"
    assert driver.observe(object_name=length).text == "0.8"
    assert driver.observe(object_name="form.input.selection.module.gen_ch").text == ""
    driver.enter_text(object_name=length, value="-1")
    driver.choose_option(object_name=waveform, value="Inline: gauss")
    snapshot(driver, tmp_path, "05-invalid-inherited")
    assert driver.observe(object_name=length).text == "-1.0"
    assert driver.observe(object_name=length).tooltip
    driver.choose_option(object_name=waveform, value="Inline: arb")
    snapshot(driver, tmp_path, "06-arb-initial")
    assert driver.observe(object_name=waveform + ".module.data").text == ""
    driver.click("Readonly")
    snapshot(driver, tmp_path, "07-readonly")
    assert driver.observe(object_name="form.input.selection").enabled is False
    assert driver.observe(object_name=waveform).enabled is False
    driver.click("Busy")
    snapshot(driver, tmp_path, "08-busy")
    assert driver.observe(object_name="synthetic.status").text == "Busy"
    assert driver.observe(object_name=waveform).enabled is False
