"""Selection-aware, lossy declaration migration through the public FormState."""

from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import BaseModel, Field
from zcu_tools.gui.form import (
    FormEnv,
    FormInput,
    FormModel,
    FormPath,
    FormState,
    InputConverter,
    Option,
)


class BranchA(BaseModel):
    kind: Literal["a"] = "a"
    amount: float = 1


class BranchB(BaseModel):
    kind: Literal["b"] = "b"
    label: str = "B"


class MigrationForm(FormModel):
    branch: Annotated[BranchA | BranchB, Field(discriminator="kind")]
    added: int = 7

    @classmethod
    def options(cls, env: FormEnv) -> Mapping[FormPath, tuple[Option, ...]]:
        raise AssertionError("Migration must not consult options")

    @classmethod
    def converters(cls, env: FormEnv) -> Mapping[FormPath, InputConverter]:
        raise AssertionError("Migration must not consult converters")


def test_migration_keeps_separate_override_and_raw_while_filling_new_defaults() -> None:
    env = FormEnv()
    saved = FormInput(
        MigrationForm,
        {"branch": {"kind": "a", "amount": {"expr": "missing + 1"}}},
        selections={("branch",): {"kind": "b"}},
    )
    base = FormInput(MigrationForm, {"branch": {"kind": "a", "amount": 1}, "added": 7})
    state = FormState(base)
    notifications: list[bool] = []
    state.subscribe(lambda: notifications.append(True))

    dropped = state.migrate_input(saved, env)

    assert dropped == ()
    assert state.input.data == {
        "branch": {"kind": "a", "amount": {"expr": "missing + 1"}},
        "added": 7,
    }
    assert state.input.selections == {("branch",): {"kind": "b"}}
    assert state.base.same_as(base)
    assert saved.data == {"branch": {"kind": "a", "amount": {"expr": "missing + 1"}}}
    assert notifications == [True]
