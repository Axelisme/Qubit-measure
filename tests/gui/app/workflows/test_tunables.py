"""Current form input is submitted through the native Engine as one batch."""

from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict, Field
from zcu_tools.datafile import SoftwareProvenance
from zcu_tools.experiment.workflows import (
    Actor,
    Done,
    InvalidRunState,
    RunIdentity,
    RunMismatch,
    Step,
    TunableChange,
    WorkflowEnv,
    workflow,
)
from zcu_tools.gui.app.workflows.tunables import TunablesHost
from zcu_tools.gui.form import FormEnv, FormErrors, FormInput, FormModel, FormState

from tests.experiment.workflows._engine_fakes import Plan, Rig, State, Tunables


class ValueForm(FormModel):
    """Input counterpart of the native example's single positive tunable."""

    value: int = Field(default=1, ge=1)


def finish(
    _env: WorkflowEnv[int], _plan: Plan, _tun: Tunables, _state: State
) -> Step[State, int]:
    """An effect-free declared step for settling this run."""
    yield from ()
    return Done()


@pytest.fixture
def host(tmp_path: Path) -> tuple[TunablesHost[ValueForm, int], Rig]:
    rig = Rig(tmp_path)
    rig.start(finish)
    env = FormEnv()
    state = FormState(ValueForm.default_input(env))
    binding: TunablesHost[ValueForm, int] = TunablesHost(state, env, "run")
    binding.accept(rig.engine)
    return binding, rig


def test_indexed_apply_returns_directed_errors_without_changing_native_or_draft(
    tmp_path: Path,
) -> None:
    from tests.gui.form_adapter._fixed_models import named_form, named_input

    rig = Rig(tmp_path)
    model = named_form()

    @workflow("indexed", plan=Plan, tunables=model, init=rig.init)
    def indexed(
        env: WorkflowEnv[int], plan: Plan, tun: FormModel, state: State
    ) -> Step[State, int]:
        del env, plan, tun, state
        yield from ()
        return Done()

    rig.engine.start(
        indexed,
        plan=Plan(),
        tunables=model.model_validate(named_input(model).data),
        paths=rig.paths,
        identity=RunIdentity(
            "indexed",
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
    env = FormEnv()
    state = FormState(named_input(model))
    binding: TunablesHost[FormModel, int] = TunablesHost(state, env, "indexed")
    binding.accept(rig.engine)
    state.write({("tones", 1, "gain"): {"expr": "3 * 2"}}, env)
    before_native = rig.engine.tunables("indexed")
    before_draft, before_base, before_accepted = (
        state.input,
        state.base,
        binding.accepted_input,
    )
    notifications: list[str] = []
    state.subscribe(lambda: notifications.append("changed"))
    try:
        result = binding.apply(actor=Actor("user", "gui"))
    except TypeError as exc:
        # Observe the old boundary failure as an outcome, not a setup error.
        result = exc
    assert isinstance(result, FormErrors), (
        "Indexed apply must return path-directed FormErrors"
    )
    path = ("tones", 1, "gain")
    assert path in result.messages
    assert "native" in result.messages[path].lower()
    assert "list" in result.messages[path].lower()
    assert rig.engine.tunables("indexed") == before_native
    assert state.input.same_as(before_draft)
    assert state.base.same_as(before_base)
    assert binding.accepted_input.same_as(before_accepted)
    assert notifications == []


def test_apply_publishes_numeric_value_but_retains_expression_input(
    host: tuple[TunablesHost[ValueForm, int], Rig],
) -> None:
    editor, rig = host
    editor.state.write({("value",): {"expr": "2 + 3"}}, FormEnv())
    snapshot = editor.apply(actor=Actor("user", "gui"))
    assert not isinstance(snapshot, FormErrors)
    assert snapshot.revision == 1
    assert snapshot.values == {"value": 5}
    assert editor.accepted_input.data == {"value": {"expr": "2 + 3"}}
    assert editor.state.base.data == editor.accepted_input.data
    assert rig.engine.tunables("run").values == {"value": 5}


def test_invalid_draft_and_local_text_do_not_publish(
    host: tuple[TunablesHost[ValueForm, int], Rig],
) -> None:
    editor, rig = host
    editor.state.write({("value",): -1}, FormEnv())
    errors = editor.apply(actor=Actor("user", "gui"))
    assert isinstance(errors, FormErrors)
    assert ("value",) in errors.messages
    assert editor.state.input.data == {"value": -1}
    editor.discard()
    editor.state.write({("value",): 2}, FormEnv())
    local = FormErrors({("value",): "Incomplete number"})
    blocked = editor.apply(actor=Actor("user", "gui"), local_errors=local)
    assert isinstance(blocked, FormErrors)
    assert blocked.messages == local.messages
    assert rig.engine.tunables("run").revision == 0
    assert editor.accepted_input.data == {"value": 1}
    assert editor.state.input.data == {"value": 2}


def test_external_edit_submits_current_shared_input_and_discard_uses_accepted_base(
    host: tuple[TunablesHost[ValueForm, int], Rig],
) -> None:
    editor, rig = host
    accepted = editor.apply_external(
        {("value",): {"expr": "3 + 4"}}, actor=Actor("agent", "rpc")
    )
    assert not isinstance(accepted, FormErrors)
    assert accepted.values == {"value": 7}
    assert (
        editor.state.input.data
        == editor.state.base.data
        == {"value": {"expr": "3 + 4"}}
    )
    editor.state.write({("value",): 9}, FormEnv())
    editor.discard()
    assert editor.state.input.data == {"value": {"expr": "3 + 4"}}
    assert rig.engine.tunables("run").revision == 1


def test_invalid_external_input_stays_in_draft_until_discard(
    host: tuple[TunablesHost[ValueForm, int], Rig],
) -> None:
    editor, rig = host
    errors = editor.apply_external({("value",): 0}, actor=Actor("agent", "rpc"))
    assert isinstance(errors, FormErrors)
    assert editor.state.input.data == {"value": 0}
    assert editor.accepted_input.data == editor.state.base.data == {"value": 1}
    assert rig.engine.tunables("run").revision == 0
    editor.discard()
    assert editor.state.input.data == {"value": 1}


def test_apply_submits_complete_defaults_even_when_input_has_no_changed_leaves(
    host: tuple[TunablesHost[ValueForm, int], Rig],
) -> None:
    editor, rig = host
    rig.engine.update_tunables(
        "run",
        (TunableChange("value", 4),),
        expected_revision=0,
        actor=Actor("agent", "native"),
    )
    editor.state.reset(FormInput(ValueForm, {}))
    applied = editor.apply(actor=Actor("user", "gui"))
    assert not isinstance(applied, FormErrors)
    assert applied.revision == 2
    assert applied.values == {"value": 1}
    repeated = editor.apply(actor=Actor("user", "gui"))
    assert not isinstance(repeated, FormErrors)
    assert repeated.revision == 3
    assert repeated.values == {"value": 1}


def test_native_lifecycle_failure_retains_draft_and_accepted_input(
    host: tuple[TunablesHost[ValueForm, int], Rig],
) -> None:
    editor, rig = host
    rig.engine.stop("run")
    assert rig.engine.execute("run").lifecycle == "stopped"
    editor.state.write({("value",): 2}, FormEnv())
    with pytest.raises(InvalidRunState):
        editor.apply(actor=Actor("user", "gui"))
    assert editor.state.input.data == {"value": 2}
    assert editor.state.base.data == editor.accepted_input.data == {"value": 1}


@pytest.mark.parametrize("external", [False, True])
def test_pending_binding_rejects_apply_before_draft_writes(external: bool) -> None:
    env = FormEnv()
    state = FormState(ValueForm.default_input(env))
    binding: TunablesHost[ValueForm, int] = TunablesHost(state, env, "pending")
    before = state.input
    base = state.base
    if external:
        with pytest.raises(RuntimeError, match="requires native acceptance"):
            binding.apply_external({("value",): 7}, actor=Actor("agent", "rpc"))
    else:
        with pytest.raises(RuntimeError, match="requires native acceptance"):
            binding.apply(actor=Actor("user", "gui"))
    assert binding.engine is None
    assert binding.run_id == "pending"
    assert binding.env is env
    assert state.input.data == before.data
    assert state.base.data == base.data
    assert binding.accepted_input.data == before.data


def test_native_acceptance_preserves_unique_draft_and_submitted_base(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path)
    env = FormEnv()
    state = FormState(ValueForm.default_input(env))
    binding: TunablesHost[ValueForm, int] = TunablesHost(state, env, "run")
    submitted = binding.accepted_input
    state.write({("value",): {"expr": "2 + 3"}}, env)
    rig.start(finish)
    binding.accept(rig.engine)
    assert binding.engine is rig.engine
    assert binding.state is state
    assert state.input.data == {"value": {"expr": "2 + 3"}}
    assert state.base.data == binding.accepted_input.data == submitted.data
    result = binding.apply(actor=Actor("user", "gui"))
    assert not isinstance(result, FormErrors)
    assert result.values == {"value": 5}
    assert binding.accepted_input.data == {"value": {"expr": "2 + 3"}}


def test_wrong_or_repeated_native_acceptance_does_not_replace_binding(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path)
    rig.start(finish)
    env = FormEnv()
    state = FormState(ValueForm.default_input(env))
    wrong: TunablesHost[ValueForm, int] = TunablesHost(state, env, "wrong")
    with pytest.raises(RunMismatch) as caught:
        wrong.accept(rig.engine)
    assert caught.value.expected == "wrong"
    assert caught.value.actual == "run"
    assert wrong.engine is None
    binding: TunablesHost[ValueForm, int] = TunablesHost(state, env, "run")
    binding.accept(rig.engine)
    with pytest.raises(RuntimeError, match="already has native acceptance"):
        binding.accept(rig.engine)
    assert binding.engine is rig.engine
    assert binding.state is state


class ArrayTunables(BaseModel):
    """Native whole-array leaf accepted by Engine."""

    model_config = ConfigDict(extra="forbid")
    values: tuple[int, ...] = (1, 2)


class ArrayForm(FormModel):
    """Tuple input counterpart, submitted as JSON array values."""

    values: tuple[int, ...] = (1, 2)


def test_whole_tuple_leaf_is_submitted_as_json_array(tmp_path: Path) -> None:
    rig = Rig(tmp_path)

    @workflow(
        "arrays",
        plan=Plan,
        tunables=ArrayTunables,
        init=rig.init,
    )
    def arrays(
        env: WorkflowEnv[int],
        plan: Plan,
        tun: ArrayTunables,
        state: State,
    ) -> Step[State, int]:
        del env, plan, tun, state
        yield from ()
        return Done()

    rig.engine.start(
        arrays,
        plan=Plan(),
        tunables=ArrayTunables(),
        paths=rig.paths,
        identity=RunIdentity(
            "arrays",
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
    env = FormEnv()
    state = FormState(ArrayForm.default_input(env))
    editor: TunablesHost[ArrayForm, int] = TunablesHost(state, env, "arrays")
    editor.accept(rig.engine)
    state.write({("values",): [3, 4, 5]}, env)
    applied = editor.apply(actor=Actor("user", "gui"))
    assert not isinstance(applied, FormErrors)
    assert applied.values == {"values": [3, 4, 5]}
    assert state.base.data == {"values": [3, 4, 5]}
