"""Notebook component declarations through the public registry interface."""

from typing import Annotated, Literal, Self, cast

import pytest
from pydantic import (
    BaseModel,
    ConfigDict,
    ModelWrapValidatorHandler,
    create_model,
    model_validator,
)
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


class BeforeModelSchema(ComponentSchema):
    @model_validator(mode="before")
    @classmethod
    def prepare_input(cls, value: object) -> object:
        return value


class WrapModelSchema(ComponentSchema):
    @model_validator(mode="wrap")
    @classmethod
    def prepare_input(
        cls, value: object, handler: ModelWrapValidatorHandler[Self]
    ) -> Self:
        return handler(value)


class PostInitSchema(ComponentSchema):
    def model_post_init(self, context: object) -> None:
        pass


@pytest.mark.parametrize(
    ("model", "reason"),
    [
        (BeforeModelSchema, "before"),
        (WrapModelSchema, "wrap"),
        (PostInitSchema, "model_post_init"),
    ],
)
@pytest.mark.parametrize("placement", ["direct", "inherited", "nested"])
def test_registration_rejects_full_model_transformations_without_reserving_kind(
    model: type[ComponentSchema],
    reason: str,
    placement: Literal["direct", "inherited", "nested"],
) -> None:
    if placement == "inherited":
        model = create_model("InheritedModel", __base__=model)
    elif placement == "nested":
        model = create_model(
            "NestedModel", __base__=ComponentSchema, child=(model, ...)
        )
    registry = ComponentRegistry()
    with pytest.raises(ValueError, match=rf"{reason}.*partial"):
        registry.register("notebook/transform", model)
    registry.register("notebook/transform", PairSchema)
    assert registry.get("notebook/transform") is PairSchema


@pytest.mark.parametrize("extra", ["allow", "ignore"])
def test_registration_rejects_models_that_would_silently_accept_unknown_fields(
    extra: Literal["allow", "ignore"],
) -> None:
    class PermissiveModel(ComponentSchema):
        model_config = ConfigDict(extra=extra)

    registry = ComponentRegistry()
    with pytest.raises(TypeError, match="extra=forbid"):
        registry.register("notebook/permissive", PermissiveModel)
    registry.register("notebook/permissive", PairSchema)
    assert registry.get("notebook/permissive") is PairSchema


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


def test_registration_rejects_unit_metadata_on_non_numeric_fields() -> None:
    class TextUnitModel(ComponentSchema):
        freq: Annotated[str, UnitSpec("Hz", "MHz")]

    registry = ComponentRegistry()
    with pytest.raises(TypeError, match="numeric"):
        registry.register("notebook/physical", TextUnitModel)
    registry.register("notebook/physical", PairSchema)
    assert registry.get("notebook/physical") is PairSchema


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
