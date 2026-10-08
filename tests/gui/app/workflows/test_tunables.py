"""Native Engine submission through the workflow form host's public seam."""

from pathlib import Path

import pytest
from pydantic import Field
from zcu_tools.experiment.workflows import (
    Actor,
    Done,
    InvalidRunState,
    RevisionConflict,
    Step,
    TunableChange,
    WorkflowEnv,
)
from zcu_tools.gui.app.workflows.tunables import TunablesHost
from zcu_tools.gui.form import FormEnv, FormErrors, FormModel, FormState

from tests.experiment.workflows._engine_fakes import Plan, Rig, State, Tunables


class ValueForm(FormModel):
    """Input counterpart of the native example's single positive tunable."""

    value: int = Field(default=1, ge=1)


def finish(
    _env: WorkflowEnv[int], _plan: Plan, _tun: Tunables, _state: State
) -> Step[State, int]:
    """An effect-free declared step, used only when settling this run."""
    yield from ()
    return Done()


@pytest.fixture
def host(tmp_path: Path) -> tuple[TunablesHost[ValueForm, int], Rig]:
    rig = Rig(tmp_path)
    rig.start(finish)
    env = FormEnv()
    state = FormState(ValueForm.default_input(env))
    return TunablesHost(state, env, rig.engine, "run"), rig


def test_apply_publishes_numeric_value_but_retains_expression_input(
    host: tuple[TunablesHost[ValueForm, int], Rig],
) -> None:
    editor, rig = host
    editor.state.write({("value",): {"expr": "2 + 3"}}, FormEnv())
    snapshot = editor.apply(expected_revision=0, actor=Actor("user", "gui"))
    assert not isinstance(snapshot, FormErrors)
    assert snapshot.revision == editor.revision == 1
    assert snapshot.values == {"value": 5}
    assert editor.accepted_input.data == {"value": {"expr": "2 + 3"}}
    assert editor.state.base.data == editor.accepted_input.data
    assert rig.engine.tunables("run").values == {"value": 5}


def test_invalid_draft_and_local_text_do_not_publish(
    host: tuple[TunablesHost[ValueForm, int], Rig],
) -> None:
    editor, rig = host
    editor.state.write({("value",): -1}, FormEnv())
    errors = editor.apply(expected_revision=0, actor=Actor("user", "gui"))
    assert isinstance(errors, FormErrors)
    assert ("value",) in errors.messages
    assert editor.state.input.data == {"value": -1}
    editor.discard()
    editor.state.write({("value",): 2}, FormEnv())
    local = FormErrors({("value",): "Incomplete number"})
    blocked = editor.apply(
        expected_revision=0, actor=Actor("user", "gui"), local_errors=local
    )
    assert isinstance(blocked, FormErrors)
    assert blocked.messages == local.messages
    assert rig.engine.tunables("run").revision == 0
    assert editor.accepted_input.data == {"value": 1}
    assert editor.state.input.data == {"value": 2}


def test_stale_apply_retains_draft_and_reports_exact_versions(
    host: tuple[TunablesHost[ValueForm, int], Rig],
) -> None:
    editor, rig = host
    editor.state.write({("value",): 2}, FormEnv())
    rig.engine.update_tunables(
        "run",
        (TunableChange("value", 4),),
        expected_revision=0,
        actor=Actor("agent", "other"),
    )
    with pytest.raises(RevisionConflict) as failed:
        editor.apply(expected_revision=0, actor=Actor("user", "gui"))
    assert (failed.value.expected, failed.value.actual) == (0, 1)
    assert editor.has_conflict
    assert editor.revision == 0
    assert editor.state.input.data == {"value": 2}
    assert editor.state.base.data == {"value": 1}
    assert rig.engine.tunables("run").values == {"value": 4}


def test_external_apply_preserves_gui_draft_then_discard_follows_accepted_input(
    host: tuple[TunablesHost[ValueForm, int], Rig],
) -> None:
    editor, rig = host
    editor.state.write({("value",): 2}, FormEnv())
    applied = editor.apply_external(
        {("value",): {"expr": "3 + 4"}},
        expected_revision=0,
        actor=Actor("agent", "rpc"),
    )
    assert not isinstance(applied, FormErrors)
    assert applied.values == {"value": 7}
    assert editor.has_conflict
    assert editor.state.input.data == {"value": 2}
    assert editor.state.base.data == {"value": 1}
    assert editor.revision == 0
    with pytest.raises(RevisionConflict):
        editor.apply(expected_revision=0, actor=Actor("user", "gui"))
    editor.discard()
    assert not editor.has_conflict
    assert editor.revision == 1
    assert editor.state.input.data == {"value": {"expr": "3 + 4"}}
    assert rig.engine.tunables("run").revision == 1


def test_external_apply_updates_clean_gui_and_invalid_batch_is_atomic(
    host: tuple[TunablesHost[ValueForm, int], Rig],
) -> None:
    editor, rig = host
    errors = editor.apply_external(
        {("value",): 0}, expected_revision=0, actor=Actor("agent", "rpc")
    )
    assert isinstance(errors, FormErrors)
    assert editor.state.input.data == {"value": 1}
    assert rig.engine.tunables("run").revision == 0
    accepted = editor.apply_external(
        {("value",): 8}, expected_revision=0, actor=Actor("agent", "rpc")
    )
    assert not isinstance(accepted, FormErrors)
    assert editor.state.input.data == editor.state.base.data == {"value": 8}
    assert editor.revision == 1
    assert not editor.has_conflict


def test_native_lifecycle_and_empty_changes_fail_without_reset(
    host: tuple[TunablesHost[ValueForm, int], Rig],
) -> None:
    editor, rig = host
    with pytest.raises(ValueError, match="nonempty"):
        editor.apply(expected_revision=0, actor=Actor("user", "gui"))
    rig.engine.stop("run")
    assert rig.engine.execute("run").lifecycle == "stopped"
    editor.state.write({("value",): 2}, FormEnv())
    with pytest.raises(InvalidRunState):
        editor.apply(expected_revision=0, actor=Actor("user", "gui"))
    assert editor.state.input.data == {"value": 2}
    assert editor.state.base.data == {"value": 1}


def test_external_stale_batch_preserves_previously_accepted_input(
    host: tuple[TunablesHost[ValueForm, int], Rig],
) -> None:
    editor, rig = host
    accepted = editor.apply_external(
        {("value",): 8}, expected_revision=0, actor=Actor("agent", "rpc")
    )
    assert not isinstance(accepted, FormErrors)
    with pytest.raises(RevisionConflict) as failed:
        editor.apply_external(
            {("value",): 9}, expected_revision=0, actor=Actor("agent", "rpc")
        )
    assert (failed.value.expected, failed.value.actual) == (0, 1)
    assert editor.accepted_input.data == editor.state.input.data == {"value": 8}
    assert rig.engine.tunables("run").values == {"value": 8}
