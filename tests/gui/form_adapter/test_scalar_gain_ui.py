"""The editor marks the scalar gain row for both native pulse families."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from qtpy.QtWidgets import QApplication, QPushButton, QVBoxLayout, QWidget
from zcu_tools.cfg_model import ConfigBase
from zcu_tools.gui.form import FormEnv, FormState
from zcu_tools.gui.form_adapter import Form
from zcu_tools.gui.widgets.form import FormEditor
from zcu_tools.program.v2.modules import ConstWaveformCfg, PulseCfg, PulseReadoutCfg

from tests.gui._user_input import ComponentDriver


class ScalarGainPlan(ConfigBase):
    """Synthetic Run inputs with native drive and flat readout modules."""

    drive: PulseCfg = PulseCfg(
        ch=0, nqz=1, freq=100.0, gain=0.2, waveform=ConstWaveformCfg(length=0.1)
    )
    readout: PulseReadoutCfg = PulseReadoutCfg(
        gen_ch=1,
        ro_ch=0,
        nqz=1,
        freq=120.0,
        gain=0.2,
        waveform=ConstWaveformCfg(length=0.2),
        ro_length=0.2,
    )


@pytest.fixture
def gain_editor(qapp: QApplication) -> Iterator[ComponentDriver]:
    env = FormEnv()
    form = Form(ScalarGainPlan, {})
    window = QWidget()
    window.setWindowTitle("Scalar gain inputs")
    layout = QVBoxLayout(window)
    editor = FormEditor(FormState(form.default_input(env)), env, window)
    layout.addWidget(editor)
    submit = QPushButton("Submit", window)
    submit.setEnabled(editor.errors() is None)
    editor.validity_changed.connect(submit.setEnabled)
    layout.addWidget(submit)
    window.resize(800, 680)
    window.show()
    qapp.processEvents()
    try:
        yield ComponentDriver(window)
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize("module", ["drive", "readout"])
def test_scalar_gain_above_full_scale_marks_the_edited_row(
    gain_editor: ComponentDriver, tmp_path: Path, module: str
) -> None:
    path = f"{module}.module.gain"
    gain_editor.click(object_name=f"form.fold.{module}")
    gain_editor.screenshot(tmp_path / "01-valid.png")
    assert gain_editor.observe("Submit").enabled

    gain_editor.enter_text(object_name=f"form.input.{path}", value="1.6")
    gain_editor.screenshot(tmp_path / "02-invalid.png")
    error = gain_editor.observe(object_name=f"form.error.{path}")
    assert error.fully_visible and error.caption_fits
    assert "less than or equal to 1" in error.text
    assert not gain_editor.observe("Submit").enabled

    gain_editor.enter_text(object_name=f"form.input.{path}", value="1")
    gain_editor.screenshot(tmp_path / "03-corrected.png")
    assert not gain_editor.observe(object_name=f"form.error.{path}").visible
    assert gain_editor.observe("Submit").enabled
