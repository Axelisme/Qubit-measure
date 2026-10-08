"""T1's Qt caller submits the same current input as external JSON edits."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from qtpy.QtWidgets import QApplication, QLineEdit
from zcu_tools.experiment.workflows import (
    Actor,
    DeviceSnapshot,
    Engine,
    EnginePorts,
    RunIdentity,
    RunPaths,
)
from zcu_tools.gui.form import FormEnv, FormErrors, FormState

from tests.experiment.workflows._engine_fakes import (
    ManualClock,
    RecordingBar,
    RecordingDevices,
    RecordingPlots,
)
from tests.zcu_lab.workflows._sim_inputs import Inputs, make_plan, seed_from_inputs
from zcu_lab.workflows.calibration import Calibrated
from zcu_lab.workflows.gui.tunables import T1TunablesForm
from zcu_lab.workflows.gui.tuning_panel import T1TuningPanel
from zcu_lab.workflows.t1 import t1_fluxdep


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        return QApplication([])
    if not isinstance(app, QApplication):
        raise RuntimeError("Tuning tests require a QApplication")
    return app


@pytest.fixture
def tuning(
    qapp: QApplication,
    inputs: Inputs,
    tmp_path: Path,
) -> Iterator[tuple[T1TuningPanel, Engine[Calibrated]]]:
    env = FormEnv()
    accepted = T1TunablesForm.default_input(env)
    result = FormState(accepted).result(env)
    assert isinstance(result, T1TunablesForm)
    engine = Engine(
        EnginePorts(
            RecordingPlots(),
            RecordingBar,
            ManualClock(),
            context=seed_from_inputs(inputs),
            soc=inputs.soc,
            soccfg=inputs.soccfg,
            devices=RecordingDevices(),
        )
    )
    engine.start(
        t1_fluxdep,
        plan=make_plan(inputs),
        tunables=result.to_cfg(),
        paths=RunPaths(tmp_path / "metadata", tmp_path / "data"),
        identity=RunIdentity("tuning", DeviceSnapshot({}), "offline"),
    )
    widget = T1TuningPanel(engine, "tuning", env, accepted)
    yield widget, engine
    widget.close()
    widget.deleteLater()
    qapp.processEvents()


def test_gui_apply_and_json_edits_share_native_t1_revision(
    tuning: tuple[T1TuningPanel, Engine[Calibrated]],
) -> None:
    widget, engine = tuning
    widget.panel.state.write({("guess_us",): {"expr": "10 * 3"}}, FormEnv())
    applied = widget.apply()
    assert not isinstance(applied, FormErrors)
    assert applied.revision == 1
    native_values = engine.tunables("tuning").values
    assert isinstance(native_values, dict)
    assert native_values["guess_us"] == 30.0
    assert widget.host.accepted_input.data["guess_us"] == {"expr": "10 * 3"}
    remote = widget.apply_external({("reps",): 200}, actor=Actor("agent", "rpc"))
    assert not isinstance(remote, FormErrors)
    assert remote.revision == 2
    assert widget.panel.state.input.data["reps"] == 200
    assert widget.host.accepted_input.data["guess_us"] == {"expr": "10 * 3"}


def test_incomplete_gui_text_blocks_apply_and_discard_restores_control(
    tuning: tuple[T1TuningPanel, Engine[Calibrated]],
) -> None:
    widget, engine = tuning
    control = widget.panel.editor.findChild(QLineEdit, "form.input.sweep_factor")
    assert isinstance(control, QLineEdit)
    control.setText("-")
    blocked = widget.apply()
    assert isinstance(blocked, FormErrors)
    assert ("sweep_factor",) in blocked.messages
    assert engine.tunables("tuning").revision == 0
    widget.discard()
    assert widget.panel.editor.local_parse_errors() is None
    restored = widget.panel.editor.findChild(QLineEdit, "form.input.sweep_factor")
    assert isinstance(restored, QLineEdit)
    result = widget.panel.state.result(FormEnv())
    assert isinstance(result, T1TunablesForm)
    assert float(restored.text()) == result.sweep_factor


def test_external_edit_submits_pending_parsed_gui_values_in_the_same_batch(
    tuning: tuple[T1TuningPanel, Engine[Calibrated]],
) -> None:
    widget, engine = tuning
    widget.panel.state.write({("guess_us",): 31.0}, FormEnv())
    applied = widget.apply_external({("reps",): 200}, actor=Actor("agent", "rpc"))
    assert not isinstance(applied, FormErrors)
    values = engine.tunables("tuning").values
    assert isinstance(values, dict)
    assert (values["guess_us"], values["reps"]) == (31.0, 200)
    assert widget.panel.state.input.data == widget.panel.state.base.data
    widget.panel.state.write({("guess_us",): 50.0}, FormEnv())
    widget.discard()
    assert widget.panel.state.input.data["guess_us"] == 31.0
    assert engine.tunables("tuning").revision == 1


def test_successful_external_edit_replaces_unparseable_view_text(
    tuning: tuple[T1TuningPanel, Engine[Calibrated]],
) -> None:
    widget, engine = tuning
    control = widget.panel.editor.findChild(QLineEdit, "form.input.sweep_factor")
    assert isinstance(control, QLineEdit)
    control.setText("-")
    applied = widget.apply_external(
        {("sweep_factor",): 8.0}, actor=Actor("agent", "rpc")
    )
    assert not isinstance(applied, FormErrors)
    assert engine.tunables("tuning").revision == 1
    assert widget.panel.editor.local_parse_errors() is None
    restored = widget.panel.editor.findChild(QLineEdit, "form.input.sweep_factor")
    assert isinstance(restored, QLineEdit)
    assert float(restored.text()) == 8.0


def test_invalid_native_t1_batch_retains_input_without_partial_publication(
    tuning: tuple[T1TuningPanel, Engine[Calibrated]],
) -> None:
    widget, engine = tuning
    before = engine.tunables("tuning")
    blocked = widget.apply_external(
        {("reps",): 200, ("guess_us",): -1.0}, actor=Actor("agent", "rpc")
    )
    assert isinstance(blocked, FormErrors)
    assert ("guess_us",) in blocked.messages
    assert engine.tunables("tuning") == before
    assert widget.panel.state.input.data["reps"] == 200
    assert widget.panel.state.input.data["guess_us"] == -1.0
    assert widget.panel.state.base.data == widget.host.accepted_input.data
    widget.discard()
    assert widget.panel.state.input.data == widget.host.accepted_input.data
