"""Replayable synthetic readout and writeback component scenarios."""

import json
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock

from qtpy.QtCore import QCoreApplication, QEvent
from qtpy.QtWidgets import QApplication, QVBoxLayout, QWidget
from zcu_tools.cfg_model import (
    ConfigBase,
    ConstWaveformCfg,
    MuxReadoutCfg,
    MuxReadoutSpec,
    MuxReadoutTone,
    PulseReadoutCfg,
    PulseReadoutSpec,
    Template,
    TemplateDef,
)
from zcu_tools.gui.app.measure.writeback import ModuleWriteback, WritebackValues
from zcu_tools.gui.app.measure.writeback.widget import WritebackWidget
from zcu_tools.gui.form import FormEnv, FormState
from zcu_tools.gui.form_adapter import Form
from zcu_tools.gui.session.library_entry import ModuleEntryForm
from zcu_tools.gui.widgets.form import FormEditor

from tests.gui._user_input import ComponentDriver
from tests.gui.app.measure._cfg_fakes import parameter_capture


class PulsePlan(ConfigBase):
    readout: PulseReadoutCfg


class MuxPlan(ConfigBase):
    readout: MuxReadoutCfg


def pulse_spec() -> PulseReadoutSpec:
    return PulseReadoutSpec(
        gen_ch=0,
        ro_ch=0,
        nqz=1,
        freq=6000.0,
        gain=0.2,
        ro_length=1.0,
        waveform=ConstWaveformCfg(length=1.0),
    )


def mux_spec() -> MuxReadoutSpec:
    return MuxReadoutSpec(
        gen_ch=0,
        nqz=1,
        mixer_freq=6000.0,
        length=1.0,
        ro_length=1.0,
        tones=[
            MuxReadoutTone(freq=6000.0, gain=0.2, ro_ch=0, label="A"),
            MuxReadoutTone(freq=6100.0, gain=0.3, ro_ch=1, label="B"),
        ],
    )


@contextmanager
def plan_editor(app: QApplication, *, mux: bool):
    spec = mux_spec() if mux else pulse_spec()
    form = Form(MuxPlan if mux else PulsePlan, {"readout": Template("readout")})
    form.bind_templates(
        {
            "readout": TemplateDef(
                type(spec),
                "Synthetic readout",
                spec.model_dump(mode="python", exclude={"type", "desc", "classifier"}),
            )
        },
        adapter="synthetic",
    )
    env = FormEnv()
    window = QWidget()
    window.setWindowTitle("Synthetic mux" if mux else "Synthetic pulse")
    layout = QVBoxLayout(window)
    editor = FormEditor(FormState(form.default_input(env)), env, window)
    layout.addWidget(editor)
    window.resize(900, 850)
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


def capture(driver: ComponentDriver, output: Path, name: str) -> None:
    driver.screenshot(output / f"{name}.png")
    (output / f"{name}.json").write_text(json.dumps(asdict(driver.visible()), indent=2))


def exercise_pulse(app: QApplication, output: Path) -> None:
    with plan_editor(app, mux=False) as driver:
        capture(driver, output, "pulse-01-closed")
        driver.click(object_name="form.fold.readout")
        capture(driver, output, "pulse-02-advanced-closed")
        driver.click(object_name="form.fold.group.readout.Advanced")
        capture(driver, output, "pulse-03-none")
        assert (
            driver.observe(object_name="form.input.readout.module.classifier").text
            == "None"
        )
        driver.choose_option(
            object_name="form.input.readout.module.classifier", value="circle"
        )
        capture(driver, output, "pulse-04-circle")
        driver.click(object_name="form.fold.readout.module.classifier")
        capture(driver, output, "pulse-04b-circle-expanded")
        driver.enter_text(
            object_name="form.input.readout.module.classifier.g_center", value="1+2j"
        )
        capture(driver, output, "pulse-05-ground")
        driver.enter_text(
            object_name="form.input.readout.module.classifier.e_center", value="3+4j"
        )
        capture(driver, output, "pulse-06-excited")
        assert (
            driver.observe(
                object_name="form.input.readout.module.classifier.g_center"
            ).text
            == "(1+2j)"
        )
        driver.choose_option(
            object_name="form.input.readout.module.classifier", value="linear"
        )
        capture(driver, output, "pulse-07-linear")
        driver.enter_text(
            object_name="form.input.readout.module.classifier.angle", value="0.5"
        )
        capture(driver, output, "pulse-08-angle")
        driver.enter_text(
            object_name="form.input.readout.module.classifier.threshold", value="2"
        )
        capture(driver, output, "pulse-09-threshold")
        assert (
            float(
                driver.observe(
                    object_name="form.input.readout.module.classifier.threshold"
                ).text
            )
            == 2.0
        )
        driver.choose_option(
            object_name="form.input.readout.module.classifier", value="circle"
        )
        capture(driver, output, "pulse-10-back-to-circle")
        assert driver.observe(
            object_name="form.input.readout.module.classifier.g_center"
        ).visible


def exercise_mux(app: QApplication, output: Path) -> None:
    with plan_editor(app, mux=True) as driver:
        capture(driver, output, "mux-01-closed")
        driver.click(object_name="form.fold.readout")
        capture(driver, output, "mux-02-readout")
        driver.click(object_name="form.fold.readout.module.tones")
        capture(driver, output, "mux-03-tones")
        driver.click(object_name="form.fold.group.readout.module.tones.0.Advanced")
        capture(driver, output, "mux-04-none-a")
        driver.choose_option(
            object_name="form.input.readout.module.tones.0.classifier", value="linear"
        )
        capture(driver, output, "mux-05-linear-a")
        driver.click(object_name="form.fold.readout.module.tones.0.classifier")
        capture(driver, output, "mux-05b-linear-expanded")
        driver.enter_text(
            object_name="form.input.readout.module.tones.0.classifier.angle",
            value="0.75",
        )
        capture(driver, output, "mux-06-angle-a")
        driver.enter_text(
            object_name="form.input.readout.module.tones.0.classifier.threshold",
            value="4",
        )
        capture(driver, output, "mux-07-threshold-a")
        driver.choose_option(
            object_name="form.picker.readout.module.tones", value="1: B"
        )
        capture(driver, output, "mux-08-tone-b")
        driver.click(object_name="form.fold.group.readout.module.tones.1.Advanced")
        capture(driver, output, "mux-09-none-b")
        assert (
            driver.observe(
                object_name="form.input.readout.module.tones.1.classifier"
            ).text
            == "None"
        )
        driver.choose_option(
            object_name="form.input.readout.module.tones.1.classifier", value="circle"
        )
        capture(driver, output, "mux-10-circle-b")
        driver.click(object_name="form.fold.readout.module.tones.1.classifier")
        capture(driver, output, "mux-10b-circle-expanded")
        driver.enter_text(
            object_name="form.input.readout.module.tones.1.classifier.g_center",
            value="5+6j",
        )
        capture(driver, output, "mux-11-ground-b")
        driver.choose_option(
            object_name="form.picker.readout.module.tones", value="0: A"
        )
        capture(driver, output, "mux-12-tone-a-retained")
        assert (
            driver.observe(
                object_name="form.input.readout.module.tones.0.classifier"
            ).text
            == "linear"
        )
        assert (
            float(
                driver.observe(
                    object_name="form.input.readout.module.tones.0.classifier.threshold"
                ).text
            )
            == 4.0
        )


def exercise_writeback(app: QApplication, output: Path) -> None:
    spec = pulse_spec()
    form = ModuleEntryForm(spec, parameter_capture(), arb_names=())
    item = ModuleWriteback(
        target_name="readout", description="Synthetic calibration", proposal=spec
    )
    item.session_id = "ml-1"
    ctrl = MagicMock()
    ctrl.writeback_control.get_writeback_item_form_for_pane.return_value = form

    def preview(*_args, **_kwargs):
        proposed = form.result()
        return {
            "ml-1": WritebackValues(
                actual_path="readout",
                current=spec.model_dump(mode="python"),
                proposed=proposed.model_dump(mode="python")
                if isinstance(proposed, PulseReadoutSpec)
                else {},
                unit=None,
            )
        }

    ctrl.writeback_control.get_writeback_values_for_pane.side_effect = preview
    window = WritebackWidget(ctrl, tab_id="synthetic", pane="analysis")
    window.populate([item], preview())
    window.resize(850, 500)
    window.show()
    app.processEvents()
    driver = ComponentDriver(window)
    try:
        capture(driver, output, "writeback-01-ledger")
        driver.click("Edit")
        capture(driver, output, "writeback-02-dialog")
        driver.click(object_name="form.fold.group.Advanced")
        capture(driver, output, "writeback-03-none")
        assert driver.observe(object_name="form.input.classifier").text == "None"
        driver.choose_option(object_name="form.input.classifier", value="linear")
        capture(driver, output, "writeback-04-linear")
        driver.click(object_name="form.fold.classifier")
        capture(driver, output, "writeback-04b-linear-expanded")
        driver.enter_text(object_name="form.input.classifier.angle", value="0.25")
        capture(driver, output, "writeback-05-angle")
        driver.enter_text(object_name="form.input.classifier.threshold", value="1.5")
        capture(driver, output, "writeback-06-threshold")
        assert (
            driver.observe(object_name="form.input.classifier.threshold").text == "1.5"
        )
        driver.click("Done")
        capture(driver, output, "writeback-07-done")
    finally:
        window.detach()
        window.close()
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


def run_scenarios(app: QApplication, output: Path) -> None:
    """Replay user input and save each visible step, without live I/O."""
    output.mkdir(parents=True, exist_ok=True)
    exercise_pulse(app, output)
    exercise_mux(app, output)
    exercise_writeback(app, output)
