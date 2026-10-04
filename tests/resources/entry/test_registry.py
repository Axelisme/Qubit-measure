"""Notebook component declarations through the public registry interface."""

from collections.abc import Mapping
from typing import Annotated, Literal, Self, cast

import pytest
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ModelWrapValidatorHandler,
    create_model,
    model_validator,
)
from zcu_tools.resources.entry import (
    ComponentRegistry,
    ComponentSchema,
    ModuleSlot,
    Ref,
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


class ScalarSlotSchema(ComponentSchema):
    slots: Annotated[str, ModuleSlot()]


class NumericSlotSchema(ComponentSchema):
    slots: Annotated[dict[str, int], ModuleSlot()]


class NumericKeySlotSchema(ComponentSchema):
    slots: Annotated[dict[int, str], ModuleSlot()]


class NullableSlotSchema(ComponentSchema):
    slots: Annotated[dict[str, str] | None, ModuleSlot()] = None


class MappingSlotSchema(ComponentSchema):
    slots: Annotated[Mapping[str, str], ModuleSlot()]


class NumericRefSchema(ComponentSchema):
    target: Annotated[int, Ref()]


@pytest.mark.parametrize(
    "model",
    [ScalarSlotSchema, NumericSlotSchema, NumericKeySlotSchema, NullableSlotSchema],
)
@pytest.mark.parametrize("placement", ["direct", "nullable"])
def test_malformed_module_slot_rejects_registration_without_reserving_kind(
    model: type[ComponentSchema], placement: str
) -> None:
    if placement == "nullable":
        model = create_model(
            "NestedSlot", __base__=ComponentSchema, child=(model | None, None)
        )
    registry = ComponentRegistry()
    with pytest.raises(TypeError, match="ModuleSlot"):
        registry.register("notebook/slots", model)
    registry.register("notebook/slots", MappingSlotSchema)
    assert registry.get("notebook/slots") is MappingSlotSchema


def test_malformed_reference_marker_does_not_reserve_kind() -> None:
    registry = ComponentRegistry()
    with pytest.raises(ValueError, match="reference"):
        registry.register("notebook/link", NumericRefSchema)
    registry.register("notebook/link", PairSchema, references=("target",))
    assert registry.get("notebook/link") is PairSchema


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


class AfterModelSchema(ComponentSchema):
    @model_validator(mode="after")
    def check_model(self) -> Self:
        return self


class PostInitSchema(ComponentSchema):
    def model_post_init(self, context: object) -> None:
        pass


@pytest.mark.parametrize(
    ("model", "reason"),
    [
        (BeforeModelSchema, "before"),
        (WrapModelSchema, "wrap"),
        (AfterModelSchema, "after"),
        (PostInitSchema, "model_post_init"),
    ],
)
@pytest.mark.parametrize("placement", ["direct", "inherited", "nested", "nullable"])
def test_registration_rejects_full_model_transformations_without_reserving_kind(
    model: type[ComponentSchema],
    reason: str,
    placement: Literal["direct", "inherited", "nested", "nullable"],
) -> None:
    rejected_model = model.__name__
    if placement == "inherited":
        model = create_model("InheritedModel", __base__=model)
        rejected_model = model.__name__
    elif placement in ("nested", "nullable"):
        model = create_model(
            "NestedModel",
            __base__=ComponentSchema,
            child=(model | None if placement == "nullable" else model, ...),
        )
    registry = ComponentRegistry()
    with pytest.raises(ValueError, match=reason) as error:
        registry.register("notebook/transform", model)
    assert rejected_model in str(error.value)
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
