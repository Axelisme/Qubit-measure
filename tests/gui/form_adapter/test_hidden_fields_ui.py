"""Hidden module literals leave only operator-owned inputs in the editor."""

from pathlib import Path

import pytest
from qtpy.QtWidgets import QPushButton, QVBoxLayout, QWidget
from zcu_tools.cfg_model import (
    ConfigBase,
    Hidden,
    PulseReadoutSpec,
    Template,
    TemplateDef,
)
from zcu_tools.gui.form import FormEnv, FormState
from zcu_tools.gui.form_adapter import Form
from zcu_tools.gui.widgets.form import FormEditor
from zcu_tools.program.v2.modules import PulseReadoutCfg

from tests.gui._user_input import ComponentDriver


class ReadoutPlan(ConfigBase):
    """The channel is operator-owned; Run replaces frequency and duration."""

    readout: PulseReadoutCfg


@pytest.fixture
def hidden_readout_editor(qapp):
    env = FormEnv()
    form = Form(
        ReadoutPlan,
        {
            "readout": Template(
                "readout",
                set={"freq": Hidden(0.0), "waveform.length": Hidden(0.1)},
            )
        },
    )
    form.bind_templates(
        {
            "readout": TemplateDef(
                PulseReadoutSpec,
                "Readout",
                {
                    "nqz": 1,
                    "gain": 0.1,
                    "ro_ch": 0,
                    "ro_length": 0.1,
                    "waveform.style": "const",
                },
            )
        },
        adapter="synthetic",
    )
    window = QWidget()
    window.setWindowTitle("Hidden Run inputs")
    layout = QVBoxLayout(window)
    editor = FormEditor(FormState(form.default_input(env)), env, window)
    layout.addWidget(editor)
    submit = QPushButton("Submit", window)
    submit.setEnabled(editor.errors() is None)
    editor.validity_changed.connect(submit.setEnabled)
    layout.addWidget(submit)
    window.resize(640, 640)
    window.show()
    qapp.processEvents()
    try:
        yield ComponentDriver(window)
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_hidden_module_inputs_do_not_block_operator_completion(
    hidden_readout_editor: ComponentDriver, tmp_path: Path
) -> None:
    driver = hidden_readout_editor
    driver.screenshot(tmp_path / "01-required-channel.png")
    assert not driver.observe("Submit").enabled
    assert (
        "Field required"
        in driver.observe(object_name="form.row.readout.module.gen_ch").tooltip
    )
    assert not any(
        widget.object_name == "form.input.readout.module.freq"
        for widget in driver.visible().widgets
    )

    driver.click(object_name="form.fold.readout")
    driver.screenshot(tmp_path / "02-collapsed-error.png")
    assert not driver.observe("Submit").enabled
    assert "Field required" in driver.observe(object_name="form.row.readout").tooltip

    driver.click(object_name="form.fold.readout")
    driver.screenshot(tmp_path / "03-required-channel.png")
    driver.enter_text(object_name="form.input.readout.module.gen_ch", value="1")
    driver.screenshot(tmp_path / "04-channel-complete.png")
    assert driver.observe("Submit").enabled
    assert (
        "Field required"
        not in driver.observe(object_name="form.row.readout.module.gen_ch").tooltip
    )

    assert not any(
        widget.object_name == "form.input.readout.module.waveform.module.length"
        for widget in driver.visible().widgets
    )
    assert driver.observe("Submit").enabled
