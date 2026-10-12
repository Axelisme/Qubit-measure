"""The shipped T1 form consumes the generic public input/state contract."""

import pytest
from jsonschema import Draft202012Validator
from pydantic import JsonValue
from zcu_tools.gui.form import FormEnv, FormErrors, FormInput, FormState
from zcu_tools.gui.session.context_types import SavedPoint
from zcu_tools.gui.session.parameters import ParameterContext
from zcu_tools.resources.context.library import LibrarySnapshot, ModuleLibraryDocument
from zcu_tools.resources.entry.schema import PointDocument, PointGeneral

from zcu_lab.workflows.coherence import CoherenceTunables
from zcu_lab.workflows.gui.coherence import CoherenceTunablesForm

from ._coherence_input import t1_input


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
            target=SavedPoint(entry_id="entry", point="synthetic"),
            roles=None,
            storage=None,
        )
    )


@pytest.mark.parametrize(
    "previous", [{}, {"t1": {"guess_us": 7.0}}, {"t1": {"t1": {"reps": 200}}}]
)
def test_defaults_use_expression_only_for_initial_input(
    previous: dict[str, JsonValue],
) -> None:
    env = environment()
    before = dict(previous)
    data = CoherenceTunablesForm.defaults(env, previous)
    assert t1_input(data)["guess_us"] == (
        {"expr": "general.ext.guess_us"} if not previous else 20.0
    )
    assert previous == before
    assert FormInput(CoherenceTunablesForm, data).data == data


def test_defaults_expression_state_changes_and_native_conversion() -> None:
    env = environment()
    base = CoherenceTunablesForm.default_input(env)
    assert t1_input(base.data)["guess_us"] == {"expr": "general.ext.guess_us"}
    state = FormState(base)
    defaults = state.result(env)
    assert isinstance(defaults, CoherenceTunablesForm)
    expected_defaults = CoherenceTunables.model_validate({"t1": {"guess_us": 25.0}})
    assert defaults.to_cfg() == expected_defaults
    state.write(
        {
            ("t1", "guess_us"): {"expr": "general.ext.guess_us * 1.2"},
            ("cal", "span_mhz"): {"expr": "general.ext.guess_us * 2"},
            ("cal", "qf_widen"): [1.0, 3.0],
            ("t1", "reps"): 200,
        },
        env,
    )
    form = state.result(env)
    assert isinstance(form, CoherenceTunablesForm)
    native = form.to_cfg()
    assert native.t1.guess_us == 30.0
    assert native.cal.span_mhz == 50.0
    assert native.cal.qf_widen == (1.0, 3.0)
    assert native.t1.reps == 200
    assert native.t1.rounds == expected_defaults.t1.rounds
    changes = state.changes(env)
    assert not isinstance(changes, FormErrors)
    by_path = {change.path: change for change in changes}
    assert set(by_path) == {
        ("t1", "guess_us"),
        ("cal", "span_mhz"),
        ("cal", "qf_widen"),
        ("t1", "reps"),
    }
    assert by_path[("t1", "guess_us")].value == native.t1.guess_us
    assert by_path[("t1", "guess_us")].input == {"expr": "general.ext.guess_us * 1.2"}
    assert by_path[("cal", "span_mhz")].input == {"expr": "general.ext.guess_us * 2"}
    assert by_path[("cal", "qf_widen")].input == [1.0, 3.0]
    assert t1_input(base.data)["guess_us"] == {"expr": "general.ext.guess_us"}

    native.cal.span_mhz = 70.0
    assert form.cal.span_mhz == 50.0


@pytest.mark.parametrize("value", [-5.0, None, "not a lifetime"])
def test_context_fallback_and_invalid_referenced_value_are_distinct(
    value: float | str | None,
) -> None:
    absent = environment(present=False)
    fallback = FormState(CoherenceTunablesForm.default_input(absent)).result(absent)
    assert isinstance(fallback, CoherenceTunablesForm)
    assert fallback.to_cfg() == CoherenceTunables()

    no_capture = FormEnv()
    ordinary = FormState(CoherenceTunablesForm.default_input(no_capture)).result(
        no_capture
    )
    assert isinstance(ordinary, CoherenceTunablesForm)
    assert ordinary.to_cfg() == CoherenceTunables()

    invalid = environment(value)
    base = CoherenceTunablesForm.default_input(invalid)
    assert t1_input(base.data)["guess_us"] == {"expr": "general.ext.guess_us"}
    result = FormState(base).result(invalid)
    assert isinstance(result, FormErrors)
    assert ("t1", "guess_us") in result.messages


def test_integer_expression_validation_and_direct_same_value_replaces_expression() -> (
    None
):
    env = environment()
    state = FormState(CoherenceTunablesForm.default_input(env))
    state.write({("t1", "reps"): {"expr": "general.ext.guess_us / 10"}}, env)
    invalid = state.result(env)
    assert isinstance(invalid, FormErrors)
    assert ("t1", "reps") in invalid.messages
    state.write(
        {
            ("t1", "reps"): {"expr": "general.ext.guess_us / 5"},
            ("t1", "guess_us"): 25.0,
        },
        env,
    )
    result = state.result(env)
    assert isinstance(result, CoherenceTunablesForm)
    assert result.to_cfg().t1.reps == 5
    changes = state.changes(env)
    assert not isinstance(changes, FormErrors)
    by_path = {change.path: change for change in changes}
    assert by_path[("t1", "reps")].input == {"expr": "general.ext.guess_us / 5"}
    assert by_path[("t1", "guess_us")].value == 25.0
    assert by_path[("t1", "guess_us")].input == 25.0


def test_nested_rabi_error_is_directed_and_invalid_draft_can_be_repaired() -> None:
    env = environment()
    state = FormState(CoherenceTunablesForm.default_input(env))
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
    assert isinstance(repaired, CoherenceTunablesForm)
    assert repaired.to_cfg().cal.rabi_stop_factor == 5.0


def test_save_restore_preserves_expressions_and_reports_incompatible_leaves() -> None:
    env = environment()
    state = FormState(CoherenceTunablesForm.default_input(env))
    state.write(
        {
            ("t1", "guess_us"): {"expr": "general.ext.guess_us * 2"},
            ("cal", "ro_every"): 7,
        },
        env,
    )
    saved = FormInput.from_json(CoherenceTunablesForm, state.input.to_json())
    restored = FormState(CoherenceTunablesForm.default_input(env))
    assert restored.write_compatible(saved.data, env) == ()
    result = restored.result(env)
    assert isinstance(result, CoherenceTunablesForm)
    assert result.to_cfg().t1.guess_us == 50.0
    assert result.to_cfg().cal.ro_every == 7
    assert t1_input(restored.input.data)["guess_us"] == {
        "expr": "general.ext.guess_us * 2"
    }

    dropped = restored.write_compatible(
        {"t1": {"reps": "wrong type"}, "removed": 1, "cal": {"rabi_start_factor": 8.0}},
        env,
    )
    assert set(dropped) == {("t1", "reps"), ("removed",)}
    result = restored.result(env)
    assert isinstance(result, FormErrors)
    assert ("cal", "rabi_stop_factor") in result.messages


def test_t1_schema_accepts_partial_agent_input_and_rejects_bad_scalar_shape() -> None:
    env = environment()
    validator = Draft202012Validator(CoherenceTunablesForm.input_schema(env))
    assert validator.is_valid(
        {"t1": {"guess_us": {"expr": "general.ext.guess_us * 1.5"}}}
    )
    assert validator.is_valid({"cal": {"qf_widen": [1.0, 3.0]}, "t1": {"reps": 200}})
    assert not validator.is_valid({"t1": {"reps": 0}})
    assert not validator.is_valid({"t1": {"guess_us": {"expr": False}}})
