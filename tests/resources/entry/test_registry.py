"""Notebook component declarations through the public registry interface."""

from typing import cast

import pytest
from pydantic import BaseModel
from zcu_tools.resources.entry import ComponentRegistry, ComponentSchema


class PairSchema(ComponentSchema):
    target: str


class UnrelatedSchema(BaseModel):
    target: str


def test_registration_rejects_non_component_models_without_reserving_the_kind() -> None:
    registry = ComponentRegistry()
    invalid_model = cast(type[ComponentSchema], UnrelatedSchema)

    with pytest.raises(TypeError, match="ComponentSchema"):
        registry.register("notebook/pair", invalid_model)

    registry.register("notebook/pair", PairSchema, references=("target",))
    assert registry.get("notebook/pair") is PairSchema
