"""Input schema is a dynamic public contract, not documentation text."""

from collections.abc import Mapping
from typing import Annotated

from jsonschema import Draft202012Validator
from pydantic import Field, JsonValue
from zcu_tools.gui.form import Display, FormEnv, FormModel, FormPath, FormState
from zcu_tools.resources.context import MetaDict


class SchemaForm(FormModel):
    """Project numeric/complex envelopes, nullable values and dynamic enums."""

    amount: Annotated[float, Display(unit="us")] = Field(default=2.0, gt=0)
    amplitude: complex = 1 + 0j
    mode: str = "alpha"
    optional: float | None = None

    @classmethod
    def choices(cls, env: FormEnv) -> Mapping[FormPath, tuple[JsonValue, ...]]:
        """The same captured choices drive schema and model validation."""
        return {("mode",): ("beta",) if env.md.get("beta_only") else ("alpha",)}


def test_schema_without_state_describes_partial_expression_complex_and_choices() -> (
    None
):
    env = FormEnv(md=MetaDict())
    schema = SchemaForm.input_schema(env)
    validator = Draft202012Validator(schema)
    assert validator.is_valid({})
    assert validator.is_valid(
        {"amount": {"expr": "seed"}, "amplitude": {"__complex__": [1.0, 2.0]}}
    )
    assert validator.is_valid({"amplitude": {"expr": "1 + 2j"}, "optional": None})
    assert validator.is_valid({"amount": 3.0, "mode": "alpha"})
    assert not validator.is_valid({"amount": -1.0})
    assert not validator.is_valid({"amount": {"expr": 3}})
    assert not validator.is_valid({"amplitude": {"__complex__": [1.0]}})
    assert not validator.is_valid({"mode": "beta"})
    state = FormState(SchemaForm.default_input(env))
    assert state.input_schema(env) == schema


def test_schema_recomputes_dynamic_choices_for_each_request() -> None:
    alpha_env = FormEnv(md=MetaDict())
    beta_md = MetaDict()
    beta_md.update({"beta_only": True})
    beta_env = FormEnv(md=beta_md.snapshot())
    alpha = Draft202012Validator(SchemaForm.input_schema(alpha_env))
    beta = Draft202012Validator(SchemaForm.input_schema(beta_env))
    assert alpha.is_valid({"mode": "alpha"})
    assert not alpha.is_valid({"mode": "beta"})
    assert beta.is_valid({"mode": "beta"})
    assert not beta.is_valid({"mode": "alpha"})
