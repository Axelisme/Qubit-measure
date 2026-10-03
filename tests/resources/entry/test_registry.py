"""Notebook component declarations through the public registry interface."""

from typing import Annotated, cast

import pytest
from pydantic import BaseModel, ConfigDict
from zcu_tools.resources.document_store import UnitSpec
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

    registry.register("notebook/pair", PairSchema, references=("target",))
    assert registry.get("notebook/pair") is PairSchema


def test_registry_lifecycle_rejects_duplicates_and_allows_explicit_replacement() -> (
    None
):
    registry = ComponentRegistry()
    registry.register("notebook/pair", PairSchema, references=("target",))
    with pytest.raises(ValueError, match="already registered"):
        registry.register("notebook/pair", NestedPairSchema)
    assert registry.get("notebook/pair") is PairSchema

    registry.unregister("notebook/pair")
    with pytest.raises(UnknownKindError):
        registry.get("notebook/pair")
    registry.register("notebook/pair", NestedPairSchema, references=("links.target",))
    assert registry.get("notebook/pair") is NestedPairSchema


@pytest.mark.parametrize("spec", [UnitSpec("Hz", "us"), UnitSpec("Hz", "unknown")])
def test_registration_rejects_invalid_units_without_reserving_the_kind(
    spec: UnitSpec,
) -> None:
    class UnitModel(ComponentSchema):
        freq: Annotated[float, spec]

    registry = ComponentRegistry()
    with pytest.raises(ValueError, match="units"):
        registry.register("notebook/physical", UnitModel)

    registry.register("notebook/physical", PairSchema, references=("target",))
    assert registry.get("notebook/physical") is PairSchema


@pytest.mark.parametrize(
    "reference",
    [
        "links.missing",
        "links",
        "links.targets",
        "ext.target",
        "wiring.ch",
        "links..target",
        "",
    ],
)
def test_registration_rejects_invalid_reference_paths_without_reserving_the_kind(
    reference: str,
) -> None:
    registry = ComponentRegistry()
    with pytest.raises(ValueError, match="reference"):
        registry.register("notebook/pair", NestedPairSchema, references=(reference,))

    registry.register("notebook/pair", NestedPairSchema, references=("links.target",))
    assert registry.get("notebook/pair") is NestedPairSchema
