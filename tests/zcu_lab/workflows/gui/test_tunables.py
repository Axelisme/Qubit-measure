"""The shipped T1 form consumes the generic public input/state contract."""

import pytest
from jsonschema import Draft202012Validator
from pydantic import JsonValue
from zcu_tools.gui.form import FormEnv, FormErrors, FormInput, FormState
from zcu_tools.gui.session.context_types import EntryDraft
from zcu_tools.gui.session.parameters import ParameterContext
from zcu_tools.resources.context.library import LibrarySnapshot, ModuleLibraryDocument
from zcu_tools.resources.entry.schema import PointDocument, PointGeneral

from zcu_lab.workflows.gui.tunables import T1TunablesForm
from zcu_lab.workflows.t1 import T1Tunables


def environment(
    guess_us: float | str | None = 25.0, *, present: bool = True
) -> FormEnv:
    """Capture the lifetime guess in a real point's extension namespace."""
    return FormEnv(
        parameters=ParameterContext(
            point=PointDocument(
                format="zcu.parameter-container",
                format_version="1.0",
                general=PointGeneral(
                    created_at="2026-10-08T00:00:00Z",
                    ext={"guess_us": guess_us} if present else {},
                ),
            ),
            library=LibrarySnapshot(
                ModuleLibraryDocument(
                    format="zcu.module-library",
                    format_version="1.0",
                )
            ),
            target=EntryDraft(entry_id="entry"),
            roles=None,
        )
    )


@pytest.mark.parametrize("previous", [{}, {"guess_us": 7.0}, {"reps": 200}])
def test_defaults_use_expression_only_for_initial_input(
    previous: dict[str, JsonValue],
) -> None:
    env = environment()
    before = dict(previous)
    data = T1TunablesForm.defaults(env, previous)
    assert data["guess_us"] == (
        {"expr": "general.ext.guess_us"} if not previous else 20.0
    )
    assert previous == before
    assert FormInput(T1TunablesForm, data).data == data


def test_defaults_expression_state_changes_and_native_conversion() -> None:
    env = environment()
    base = T1TunablesForm.default_input(env)
    assert base.data["guess_us"] == {"expr": "general.ext.guess_us"}
    state = FormState(base)
    defaults = state.result(env)
    assert isinstance(defaults, T1TunablesForm)
    expected_defaults = T1Tunables(guess_us=25.0)
    assert defaults.to_cfg() == expected_defaults
    state.write(
        {
            ("guess_us",): {"expr": "general.ext.guess_us * 1.2"},
            ("cal", "span_mhz"): {"expr": "general.ext.guess_us * 2"},
            ("cal", "qf_widen"): [1.0, 3.0],
            ("reps",): 200,
        },
        env,
    )
    form = state.result(env)
    assert isinstance(form, T1TunablesForm)
    native = form.to_cfg()
    assert native.guess_us == 30.0
    assert native.cal.span_mhz == 50.0
    assert native.cal.qf_widen == (1.0, 3.0)
    assert native.reps == 200
    assert native.rounds == expected_defaults.rounds
    changes = state.changes(env)
    assert not isinstance(changes, FormErrors)
    by_path = {change.path: change for change in changes}
    assert set(by_path) == {
        ("guess_us",),
        ("cal", "span_mhz"),
        ("cal", "qf_widen"),
        ("reps",),
    }
    assert by_path[("guess_us",)].value == native.guess_us
    assert by_path[("guess_us",)].input == {"expr": "general.ext.guess_us * 1.2"}
    assert by_path[("cal", "span_mhz")].input == {"expr": "general.ext.guess_us * 2"}
    assert by_path[("cal", "qf_widen")].input == [1.0, 3.0]
    assert base.data["guess_us"] == {"expr": "general.ext.guess_us"}

    native.cal.span_mhz = 70.0
    assert form.cal.span_mhz == 50.0


@pytest.mark.parametrize("value", [-5.0, None, "not a lifetime"])
def test_context_fallback_and_invalid_referenced_value_are_distinct(
    value: float | str | None,
) -> None:
    absent = environment(present=False)
    fallback = FormState(T1TunablesForm.default_input(absent)).result(absent)
    assert isinstance(fallback, T1TunablesForm)
    assert fallback.to_cfg() == T1Tunables()

    no_capture = FormEnv()
    ordinary = FormState(T1TunablesForm.default_input(no_capture)).result(no_capture)
    assert isinstance(ordinary, T1TunablesForm)
    assert ordinary.to_cfg() == T1Tunables()

    invalid = environment(value)
    base = T1TunablesForm.default_input(invalid)
    assert base.data["guess_us"] == {"expr": "general.ext.guess_us"}
    result = FormState(base).result(invalid)
    assert isinstance(result, FormErrors)
    assert ("guess_us",) in result.messages


def test_integer_expression_validation_and_direct_same_value_replaces_expression() -> (
    None
):
    env = environment()
    state = FormState(T1TunablesForm.default_input(env))
    state.write({("reps",): {"expr": "general.ext.guess_us / 10"}}, env)
    invalid = state.result(env)
    assert isinstance(invalid, FormErrors)
    assert ("reps",) in invalid.messages
    state.write(
        {("reps",): {"expr": "general.ext.guess_us / 5"}, ("guess_us",): 25.0}, env
    )
    result = state.result(env)
    assert isinstance(result, T1TunablesForm)
    assert result.to_cfg().reps == 5
    changes = state.changes(env)
    assert not isinstance(changes, FormErrors)
    by_path = {change.path: change for change in changes}
    assert by_path[("reps",)].input == {"expr": "general.ext.guess_us / 5"}
    assert by_path[("guess_us",)].value == 25.0
    assert by_path[("guess_us",)].input == 25.0


def test_nested_rabi_error_is_directed_and_invalid_draft_can_be_repaired() -> None:
    env = environment()
    state = FormState(T1TunablesForm.default_input(env))
    state.write(
        {("cal", "rabi_start_factor"): 4.0, ("cal", "rabi_stop_factor"): 3.0}, env
    )
    result = state.result(env)
    changes = state.changes(env)
    assert isinstance(result, FormErrors)
    assert isinstance(changes, FormErrors)
    assert set(result.messages) == {("cal", "rabi_stop_factor")}
    assert changes.messages == result.messages
    state.write({("cal", "rabi_stop_factor"): 5.0}, env)
    repaired = state.result(env)
    assert isinstance(repaired, T1TunablesForm)
    assert repaired.to_cfg().cal.rabi_stop_factor == 5.0


def test_save_restore_preserves_expressions_and_reports_incompatible_leaves() -> None:
    env = environment()
    state = FormState(T1TunablesForm.default_input(env))
    state.write(
        {("guess_us",): {"expr": "general.ext.guess_us * 2"}, ("cal", "ro_every"): 7},
        env,
    )
    saved = FormInput.from_json(T1TunablesForm, state.input.to_json())
    restored = FormState(T1TunablesForm.default_input(env))
    assert restored.write_compatible(saved.data, env) == ()
    result = restored.result(env)
    assert isinstance(result, T1TunablesForm)
    assert result.to_cfg().guess_us == 50.0
    assert result.to_cfg().cal.ro_every == 7
    assert restored.input.data["guess_us"] == {"expr": "general.ext.guess_us * 2"}

    dropped = restored.write_compatible(
        {"reps": "wrong type", "removed": 1, "cal": {"rabi_start_factor": 8.0}}, env
    )
    assert set(dropped) == {("reps",), ("removed",)}
    result = restored.result(env)
    assert isinstance(result, FormErrors)
    assert ("cal", "rabi_stop_factor") in result.messages


def test_t1_schema_accepts_partial_agent_input_and_rejects_bad_scalar_shape() -> None:
    env = environment()
    validator = Draft202012Validator(T1TunablesForm.input_schema(env))
    assert validator.is_valid({"guess_us": {"expr": "general.ext.guess_us * 1.5"}})
    assert validator.is_valid({"cal": {"qf_widen": [1.0, 3.0]}, "reps": 200})
    assert not validator.is_valid({"reps": 0})
    assert not validator.is_valid({"guess_us": {"expr": False}})
