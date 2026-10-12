"""T1's Qt caller submits the same current input as external JSON edits."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from qtpy.QtWidgets import QApplication, QLineEdit
from zcu_tools.datafile import SoftwareProvenance
from zcu_tools.experiment.workflows import (
    Actor,
    Done,
    Engine,
    EnginePorts,
    RunIdentity,
    RunPaths,
    Step,
    WorkflowEnv,
    workflow,
)
from zcu_tools.gui.app.workflows.controller import WorkflowController
from zcu_tools.gui.app.workflows.tunables import TunablesHost
from zcu_tools.gui.form import FormEnv, FormErrors, FormState
from zcu_tools.plotting.plots import NonPresentingHost, Plots

from tests.experiment.workflows._engine_fakes import (
    ManualClock,
    RecordingBar,
    RecordingDevices,
)
from tests.gui.app.workflows._start_fakes import StartHarness
from tests.zcu_lab.workflows._sim_inputs import Inputs, make_plan, seed_from_inputs
from zcu_lab.workflows.calibration import Calibrated
from zcu_lab.workflows.gui.tunables import T1TunablesForm
from zcu_lab.workflows.gui.tuning_panel import T1TuningPanel
from zcu_lab.workflows.t1 import T1Plan, T1Tunables


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
) -> Iterator[tuple[T1TuningPanel[T1TunablesForm, Calibrated], Engine[Calibrated]]]:
    env = FormEnv()
    accepted = T1TunablesForm.default_input(env)
    result = FormState(accepted).result(env)
    assert isinstance(result, T1TunablesForm)
    harness = StartHarness()
    controller: WorkflowController[T1TunablesForm, Calibrated] = WorkflowController(
        harness.runner, "tuning", on_finished=lambda _status, _error: None
    )
    binding: TunablesHost[T1TunablesForm, Calibrated] = TunablesHost(
        FormState(accepted), env, "tuning"
    )

    @workflow("tuning", plan=T1Plan, tunables=T1Tunables, init=lambda _env, _plan: None)
    def step(
        env: WorkflowEnv[Calibrated], plan: T1Plan, tun: T1Tunables, state: None
    ) -> Step[None, int]:
        del plan, tun, state
        controller.pause("tuning")
        yield from env.wait_until(env.started_at)
        return Done()

    controller.start(
        step,
        binding=binding,
        ports=EnginePorts(
            Plots(NonPresentingHost()),
            RecordingBar,
            ManualClock(),
            context=seed_from_inputs(inputs),
            soc=inputs.soc,
            soccfg=inputs.soccfg,
            devices=RecordingDevices(),
        ),
        plan=make_plan(inputs),
        tunables=result.to_cfg(),
        paths=RunPaths(tmp_path / "metadata", tmp_path / "data"),
        identity=RunIdentity(
            "tuning",
            SoftwareProvenance(
                software_versions={},
                git_commit=None,
                git_dirty=None,
                qick_version=None,
                soc_fingerprint=None,
                hostname="offline",
            ),
        ),
    )
    harness.complete_segment()
    engine = controller.engine
    assert engine is not None
    widget = T1TuningPanel(controller)
    assert widget.host is binding
    assert widget.panel.state is binding.state
    yield widget, engine
    controller.stop("tuning")
    widget.close()
    widget.deleteLater()
    qapp.processEvents()


def test_gui_apply_and_json_edits_share_native_t1_revision(
    tuning: tuple[T1TuningPanel[T1TunablesForm, Calibrated], Engine[Calibrated]],
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
    tuning: tuple[T1TuningPanel[T1TunablesForm, Calibrated], Engine[Calibrated]],
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
    tuning: tuple[T1TuningPanel[T1TunablesForm, Calibrated], Engine[Calibrated]],
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
    tuning: tuple[T1TuningPanel[T1TunablesForm, Calibrated], Engine[Calibrated]],
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


def test_journal_failure_propagates_without_accepting_input_or_clearing_local_text(
    tuning: tuple[T1TuningPanel[T1TunablesForm, Calibrated], Engine[Calibrated]],
    tmp_path: Path,
) -> None:
    widget, engine = tuning
    before = engine.tunables("tuning")
    accepted = widget.host.accepted_input.data
    control = widget.panel.editor.findChild(QLineEdit, "form.input.sweep_factor")
    assert isinstance(control, QLineEdit)
    control.setText("-")
    journal = tmp_path / "metadata/journal.jsonl"
    original = journal.read_bytes()
    journal.unlink()
    journal.mkdir()
    try:
        with pytest.raises(IsADirectoryError) as caught:
            widget.apply_external({("guess_us",): 45.0}, actor=Actor("agent", "rpc"))
    finally:
        journal.rmdir()
        journal.write_bytes(original)
    assert caught.value.filename == str(journal)
    assert engine.tunables("tuning") == before
    status = engine.status()
    assert status is not None and status.lifecycle == "failed"
    assert widget.host.accepted_input.data == widget.panel.state.base.data == accepted
    assert widget.panel.state.input.data["guess_us"] == 45.0
    assert (
        widget.panel.editor.local_parse_errors() is not None and control.text() == "-"
    )


def test_invalid_native_t1_batch_retains_input_without_partial_publication(
    tuning: tuple[T1TuningPanel[T1TunablesForm, Calibrated], Engine[Calibrated]],
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
