"""Notebook component declarations through the public registry interface."""

from typing import cast

import pytest
from pydantic import BaseModel, ConfigDict, Field
from zcu_tools.resources.entry import (
    ComponentRegistry,
    ComponentSchema,
    UnknownKindError,
)


class PairSchema(ComponentSchema):
    target: str


class UnrelatedSchema(BaseModel):
    target: str


class PairLinks(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target: str
    targets: list[str]


class NestedPairSchema(ComponentSchema):
    links: PairLinks


def test_registration_rejects_non_component_models_without_reserving_the_kind() -> None:
    registry = ComponentRegistry()
    invalid_model = cast(type[ComponentSchema], UnrelatedSchema)

    with pytest.raises(TypeError, match="ComponentSchema"):
        registry.register("notebook/pair", invalid_model)

    registry.register("notebook/pair", PairSchema)
    assert registry.get("notebook/pair") is PairSchema


def test_registry_lifecycle_rejects_duplicates_and_allows_explicit_replacement() -> (
    None
):
    registry = ComponentRegistry()
    registry.register("notebook/pair", PairSchema)
    with pytest.raises(ValueError, match="already registered"):
        registry.register("notebook/pair", NestedPairSchema)
    assert registry.get("notebook/pair") is PairSchema

    registry.unregister("notebook/pair")
    with pytest.raises(UnknownKindError):
        registry.get("notebook/pair")
    registry.register("notebook/pair", NestedPairSchema)
    assert registry.get("notebook/pair") is NestedPairSchema


def test_registration_preserves_required_wiring_indices_without_physical_units() -> (
    None
):
    class RequiredWiring(BaseModel):
        ch: int = Field(ge=0, strict=True)

    class RequiredChannelModel(ComponentSchema):
        wiring: RequiredWiring

    registry = ComponentRegistry()
    registry.register("notebook/required-channel", RequiredChannelModel)
    model = RequiredChannelModel.model_validate(
        {"kind": "notebook/required-channel", "wiring": {"ch": 3}}
    )
    assert model.wiring.ch == 3
