"""Workflow input mementos through public Window and app lifecycle seams."""

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from pydantic import Field, JsonValue
from qtpy.QtWidgets import QApplication, QLabel
from zcu_tools.cfg_model import PulseSpec
from zcu_tools.datafile import RunSnapshot
from zcu_tools.experiment.workflows import (
    Done,
    InitEnv,
    Step,
    WorkflowEnv,
    workflow,
)
from zcu_tools.gui.app.workflows.app import WorkflowsGuiBehavior
from zcu_tools.gui.app.workflows.attachment import WorkflowAttachment
from zcu_tools.gui.app.workflows.persistence import (
    WorkflowInputCodec,
    WorkflowInputSnapshot,
)
from zcu_tools.gui.app.workflows.window import WorkflowWindow
from zcu_tools.gui.form import (
    FormEnv,
    FormInput,
    FormModel,
    FormPath,
    InputConverter,
    Option,
)
from zcu_tools.gui.form_adapter import (
    LinRange,
    ModuleInline,
    ModuleInputField,
    ModuleRef,
    SweepInput,
)
from zcu_tools.gui.session.persistence import SingleFileCaretaker

from tests.experiment.workflows._engine_fakes import Plan, State, Tunables
from tests.gui.app.workflows._start_fakes import StartHarness
from tests.gui.app.workflows._synthetic import PlanForm, TunablesForm

from ._sample_input import sample_input
from ._synthetic import ALIAS_PRESET, Context, build_registry
from ._window_support import WindowSession, bind_point, install_soc


@pytest.fixture
def window(qapp: QApplication) -> Iterator[WorkflowWindow[Context]]:
    harness = StartHarness()
    view = WorkflowWindow(
        harness.state,
        harness.services,
        harness.owner,
        templates={},
        alias_presets=None,
        registrations=build_registry(),
    )
    try:
        yield view
    finally:
        view.close()
        view.deleteLater()
        qapp.processEvents()


def test_original_expressions_roundtrip_and_restore_base(
    window: WorkflowWindow[Context], tmp_path: Path
) -> None:
    window.plan_panel.state.write(
        {("scale",): {"expr": "0.1 + 0.2"}}, window.plan_panel.env
    )
    window.tunables_panel.state.write(
        {("sample", "value"): {"expr": "7 * 3"}}, window.tunables_panel.env
    )
    saved = window.capture_persisted_state()
    caretaker = SingleFileCaretaker(
        window, codec=WorkflowInputCodec(), filename="input.json", cache_dir=tmp_path
    )
    caretaker.flush()
    raw = json.loads(caretaker.state_path.read_text())
    assert raw["plan"]["scale"] == {"expr": "0.1 + 0.2"}
    assert sample_input(raw["knobs"])["value"] == {"expr": "7 * 3"}
    window.plan_panel.state.reset(PlanForm.default_input(window.plan_panel.env))
    window.tunables_panel.state.reset(
        TunablesForm.default_input(window.tunables_panel.env)
    )
    outcome = caretaker.restore_all()
    assert outcome.load_error is None
    assert window.capture_persisted_state() == saved
    assert window.plan_panel.state.base.data == saved.plan
    assert window.tunables_panel.state.base.data == saved.knobs


class PersistedSweepForm(FormModel):
    sweep: SweepInput = Field(default_factory=LinRange)


@pytest.mark.parametrize(
    "sweep",
    [
        {"kind": "center", "center": {"expr": "3 + 4"}, "span": -6, "n": 4},
        {"kind": "center_step", "center": 0.5, "span": 1, "step": {"expr": "0.1 * 3"}},
    ],
    ids=["center-expts", "center-step"],
)
def test_sweep_modes_and_expressions_roundtrip_through_window_memento(
    sweep_window, sweep: JsonValue
) -> None:
    original = sweep_window(PersistedSweepForm, None)
    original.plan_panel.state.install_input(
        FormInput(PersistedSweepForm, {"sweep": sweep})
    )
    saved = original.capture_persisted_state()
    assert saved.plan["sweep"] == sweep
    codec = WorkflowInputCodec()
    restored = sweep_window(PersistedSweepForm, codec.decode(codec.encode(saved)))
    assert restored.capture_persisted_state() == saved
    assert restored.plan_panel.state.input.data == saved.plan
    assert restored.plan_panel.state.base.data == saved.plan


class PersistedDraftForm(FormModel):
    sweep: SweepInput
    optional: float | None
    required: float


def test_window_memento_restores_separate_selection_and_whole_invalid_draft(sweep_window) -> None:
    original = sweep_window(PersistedDraftForm, None)
    original.plan_panel.state.install_input(
        FormInput(
            PersistedDraftForm,
            {
                "sweep": {"kind": "range", "start": {"expr": "missing + 1"}, "stop": "invalid"},
                "optional": None,
            },
        )
    )
    original.plan_panel.state.write({("sweep",): {"kind": "center"}}, original.plan_panel.env)
    codec = WorkflowInputCodec()
    snapshot = codec.decode(codec.encode(original.capture_persisted_state()))
    restored = sweep_window(PersistedDraftForm, snapshot)
    assert restored.plan_panel.state.input.same_as(original.plan_panel.state.input)
    assert restored.plan_panel.state.base.same_as(original.plan_panel.state.input)
    assert restored.plan_panel.state.preview(restored.plan_panel.env).errors is not None


def test_incompatible_knobs_identity_preserves_both_window_drafts(window) -> None:
    saved = window.capture_persisted_state()
    window.plan_panel.state.write({("scale",): 7}, window.plan_panel.env)
    window.tunables_panel.state.install_input(FormInput(TunablesForm, {}))
    before = window.capture_persisted_state()
    incompatible = FormInput(ValueKnobsForm, {"value": 6})
    # Capture through the same public host seam, then replace just one form.
    other = saved.model_copy(update={"knobs": json.loads(incompatible.to_json())})
    with pytest.raises(ValueError, match="identity|incompatible"):
        window.restore_persisted_state(other)
    assert window.capture_persisted_state() == before


def test_incompatible_fields_drop_without_losing_other_input(
    window: WorkflowWindow[Context],
) -> None:
    defaults = window.capture_persisted_state()
    report = window.restore_persisted_state(
        WorkflowInputSnapshot(
            plan={"scale": ["wrong shape"], "removed": 4},
            knobs={"sample": {"value": {"expr": "8 * 4"}}, "removed_knob": True},
        )
    )
    assert set(report.plan_dropped) == {("scale",), ("removed",)}
    assert report.knobs_dropped == (("removed_knob",),)
    restored = window.capture_persisted_state()
    assert restored.plan == defaults.plan
    assert sample_input(restored.knobs)["value"] == {"expr": "8 * 4"}


def test_open_restore_and_close_flush(qapp: QApplication, tmp_path: Path) -> None:
    path = tmp_path / "t1_workflow_state_v1.json"
    path.write_text(
        json.dumps(
            {
                "version": 2,
                "plan": {"scale": {"expr": "1 / 4"}},
                "knobs": {"sample": {"value": 35.0}},
            }
        )
    )
    app = WorkflowsGuiBehavior(
        templates={},
        registrations=build_registry(),
        alias_presets={"default": ALIAS_PRESET},
        cache_dir=tmp_path,
    )
    app.assemble(None)
    try:
        assert app.window.capture_persisted_state().plan["scale"] == {"expr": "1 / 4"}
        assert sample_input(app.window.capture_persisted_state().knobs)["value"] == 35.0
        app.window.tunables_panel.state.write(
            {("sample", "value"): {"expr": "9 * 5"}}, app.window.tunables_panel.env
        )
        assert app.window.close()
        raw = json.loads(path.read_text())
        assert sample_input(raw["knobs"])["value"] == {"expr": "9 * 5"}
        assert raw["plan"]["scale"] == {"expr": "1 / 4"}
    finally:
        app.shutdown()
        app.window.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize(
    "payload",
    [
        "not JSON",
        '{"version": 1, "plan": {}, "knobs": {}}',
        '{"version": 2, "plan": [], "knobs": {}}',
    ],
)
def test_unreadable_snapshot_reports_and_uses_defaults(
    qapp: QApplication, tmp_path: Path, payload: str
) -> None:
    (tmp_path / "t1_workflow_state_v1.json").write_text(payload)
    app = WorkflowsGuiBehavior(
        templates={},
        registrations=build_registry(),
        alias_presets={"default": ALIAS_PRESET},
        cache_dir=tmp_path,
    )
    app.assemble(None)
    try:
        assert (
            app.window.plan_panel.state.input.data
            == PlanForm.default_input(app.window.plan_panel.env).data
        )
        assert (
            app.window.tunables_panel.state.input.data
            == TunablesForm.default_input(app.window.tunables_panel.env).data
        )
        assert any(
            "Failed to read GUI state" in label.text()
            for label in app.window.findChildren(QLabel)
        )
    finally:
        app.window.close()
        app.shutdown()
        app.window.deleteLater()
        qapp.processEvents()


def test_clean_skips_read_but_close_replaces_snapshot(
    qapp: QApplication, tmp_path: Path
) -> None:
    path = tmp_path / "t1_workflow_state_v1.json"
    path.write_text("deliberately invalid JSON")
    app = WorkflowsGuiBehavior(
        templates={},
        registrations=build_registry(),
        alias_presets={"default": ALIAS_PRESET},
        clean=True,
        cache_dir=tmp_path,
    )
    app.assemble(None)
    try:
        assert path.read_text() == "deliberately invalid JSON"
        assert not any(
            "Failed to read GUI state" in label.text()
            for label in app.window.findChildren(QLabel)
        )
        assert (
            app.window.tunables_panel.state.input.data
            == TunablesForm.default_input(app.window.tunables_panel.env).data
        )
        app.window.plan_panel.state.reset(
            FormInput(PlanForm, {"scale": {"expr": "3 / 8"}})
        )
        assert app.window.close()
        raw = json.loads(path.read_text())
        assert raw["plan"]["scale"] == {"expr": "3 / 8"}
    finally:
        app.shutdown()
        app.window.deleteLater()
        qapp.processEvents()


DRIVE_FIELD = ModuleInputField(PulseSpec)
if TYPE_CHECKING:
    type DriveInput = ModuleRef | ModuleInline
else:
    DriveInput = DRIVE_FIELD.annotation


class DrivePlanForm(FormModel):
    """Synthetic Start plan with one library reference and one ordinary leaf."""

    drive: DriveInput
    count: int = 2

    @classmethod
    def options(cls, env: FormEnv) -> Mapping[FormPath, tuple[Option, ...]]:
        """Supply the generated field's options through the public owner."""
        return {("drive",): DRIVE_FIELD.options(env)}

    @classmethod
    def converters(cls, env: FormEnv) -> Mapping[FormPath, InputConverter]:
        return {("drive",): DRIVE_FIELD.converter(env)}


class ValueKnobsForm(FormModel):
    """Synthetic tunables form matching the native positive value."""

    value: int = 1


def drive_registration() -> WorkflowAttachment[
    Plan, Tunables, State, int, DrivePlanForm, ValueKnobsForm, None
]:
    """Declare a fresh context-free workflow; restoration never runs it."""

    def init(env: InitEnv[None], plan: Plan) -> State:
        del env, plan
        return State()

    def step(
        env: WorkflowEnv[None], plan: Plan, tun: Tunables, state: State
    ) -> Step[State, int]:
        del env, plan, tun, state
        yield from ()
        return Done()

    def plan_to_cfg(form: DrivePlanForm, env: FormEnv, snapshot: RunSnapshot) -> Plan:
        del env, snapshot
        return Plan(count=form.count)

    declared = workflow("synthetic", plan=Plan, tunables=Tunables, init=init)(step)
    return WorkflowAttachment(
        step=declared,
        plan_model=DrivePlanForm,
        knobs_model=ValueKnobsForm,
        plan_to_cfg=plan_to_cfg,
        knobs_to_cfg=lambda form: Tunables(value=form.value),
        seed=lambda plan, form: None,
    )


def test_saved_reference_without_captured_library_is_discarded_on_open(
    qapp: QApplication, tmp_path: Path
) -> None:
    (tmp_path / "t1_workflow_state_v1.json").write_text(
        json.dumps(
            {
                "version": 2,
                "plan": {"drive": {"kind": "ref", "name": "Q1.pulses.pi"}, "count": 5},
                "knobs": {"value": 3},
            }
        )
    )
    app = WorkflowsGuiBehavior(
        templates={}, registrations=[drive_registration()], cache_dir=tmp_path
    )
    app.assemble(None)
    try:
        assert app.window.capture_persisted_state() == WorkflowInputSnapshot(
            plan={"drive": {"kind": "ref"}, "count": 5}, knobs={"value": 3}
        )
        assert any(
            label.text() == "Discarded plan: drive.name"
            for label in app.window.findChildren(QLabel)
        )
    finally:
        app.window.close()
        app.shutdown()
        app.window.deleteLater()
        qapp.processEvents()


def runnable(session: WindowSession) -> WorkflowWindow[None]:
    """Bind the point and SoC so the synthetic_wide default can Start."""
    bind_point(session.harness, session.root)
    install_soc(session.harness)
    return session.window


def lifecycle(window: WorkflowWindow[None]) -> str | None:
    status = window.controller.status()
    return None if status is None else status.lifecycle


def test_running_run_saves_its_live_tunables_draft(
    window_session: WindowSession, qapp: QApplication
) -> None:
    window = runnable(window_session)
    saved: list[WorkflowInputSnapshot] = []

    def edit_live_draft(point: int) -> None:
        if point != 1:
            return
        binding = window.controller.binding
        assert binding is not None and lifecycle(window) == "running"
        binding.state.write({("value",): 9}, binding.env)
        saved.append(window.capture_persisted_state())

    window_session.during_point = edit_live_draft
    assert isinstance(window.start("live"), int)
    window_session.harness.complete_segment()
    qapp.processEvents()

    assert [snapshot.knobs["value"] for snapshot in saved] == [9]
    assert window.tunables_panel.state.input.data["value"] == 1


def test_paused_run_saves_its_live_draft_until_terminal_shows_start_knobs(
    window_session: WindowSession, qapp: QApplication
) -> None:
    window = runnable(window_session)
    window_session.pause_at = 1
    assert isinstance(window.start("paused"), int)
    window_session.harness.complete_segment()
    qapp.processEvents()
    tuning = window.tuning_panel
    assert tuning is not None and lifecycle(window) == "paused"

    tuning.panel.state.write({("value",): 9}, tuning.host.env)
    assert window.capture_persisted_state().knobs["value"] == 9
    window.stop("paused")
    qapp.processEvents()

    assert lifecycle(window) == "stopped"
    assert window.tunables_panel.state.input.data["value"] == 1
    assert window.capture_persisted_state().knobs["value"] == 1


def test_start_tunables_edited_after_terminal_are_saved(
    window_session: WindowSession, qapp: QApplication, tmp_path: Path
) -> None:
    window = runnable(window_session)
    assert isinstance(window.start("finished"), int)
    window_session.harness.complete_segment()
    qapp.processEvents()
    assert lifecycle(window) == "done"

    window.tunables_panel.state.write(
        {("value",): {"expr": "2 + 3"}}, window.tunables_panel.env
    )
    assert window.capture_persisted_state().knobs["value"] == {"expr": "2 + 3"}
    caretaker = SingleFileCaretaker(
        window, codec=WorkflowInputCodec(), filename="input.json", cache_dir=tmp_path
    )
    caretaker.flush()
    raw = json.loads(caretaker.state_path.read_text())
    assert raw["knobs"]["value"] == {"expr": "2 + 3"}
