"""Notebook component declarations through the public registry interface."""

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
from zcu_tools.resources.document_store import UnitSpec
from zcu_tools.resources.entry import (
    ComponentRegistry,
    ComponentSchema,
    UnknownKindError,
)
from zcu_tools.resources.entry.schema import WiringSchema


class PairSchema(ComponentSchema):
    target: str


class ScalarWithoutUnits(ComponentSchema):
    freq: float


class IntegerWithoutUnits(ComponentSchema):
    gain: int


class NullableScalarWithoutUnits(ComponentSchema):
    freq: float | None


class NestedPhysicalWithoutUnits(BaseModel):
    model_config = ConfigDict(extra="forbid")
    freq: float


class NestedWithoutUnits(ComponentSchema):
    physical: NestedPhysicalWithoutUnits


class NullableNestedWithoutUnits(ComponentSchema):
    physical: NestedPhysicalWithoutUnits | None


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


@pytest.mark.parametrize("units", [UnitSpec("Hz", "us"), UnitSpec("bogus", "MHz")])
def test_nullable_branch_invalid_units_do_not_reserve_kind(units: UnitSpec) -> None:
    class InvalidFrequency(ComponentSchema):
        freq: Annotated[float, units] | None = None

    class ValidFrequency(ComponentSchema):
        freq: Annotated[float, UnitSpec("Hz", "MHz")] | None = None

    registry = ComponentRegistry()
    kind = "notebook/nullable-unit"
    with pytest.raises(ValueError, match="Incompatible or unsupported units"):
        registry.register(kind, InvalidFrequency)
    with pytest.raises(UnknownKindError):
        registry.get(kind)
    registry.register(kind, ValidFrequency)
    assert registry.get(kind) is ValidFrequency


def test_conflicting_nullable_branch_units_do_not_reserve_kind() -> None:
    class AmbiguousFrequency(ComponentSchema):
        freq: (
            Annotated[float, UnitSpec("Hz", "MHz")]
            | Annotated[int, UnitSpec("Hz", "GHz")]
            | None
        ) = None

    registry = ComponentRegistry()
    kind = "notebook/ambiguous-unit"
    with pytest.raises(ValueError, match="Conflicting UnitSpec"):
        registry.register(kind, AmbiguousFrequency)
    with pytest.raises(UnknownKindError):
        registry.get(kind)


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


@pytest.mark.parametrize(
    "model",
    (
        ScalarWithoutUnits,
        IntegerWithoutUnits,
        NullableScalarWithoutUnits,
        NestedWithoutUnits,
        NullableNestedWithoutUnits,
    ),
)
def test_registration_requires_units_for_physical_numbers_without_reserving_kind(
    model: type[ComponentSchema],
) -> None:
    registry = ComponentRegistry()
    with pytest.raises(ValueError, match="UnitSpec.*(freq|gain)"):
        registry.register("notebook/physical", model)

    registry.register("notebook/physical", PairSchema)
    assert registry.get("notebook/physical") is PairSchema


def test_registration_accepts_dimensionless_units_and_unconverted_channel_indices() -> (
    None
):
    class PhysicalModel(ComponentSchema):
        freq: Annotated[float, UnitSpec("Hz", "MHz")]
        gain: Annotated[int, UnitSpec("1", "1")]

    registry = ComponentRegistry()
    registry.register("notebook/physical", PhysicalModel)
    assert registry.get("notebook/physical") is PhysicalModel
    assert registry.units("notebook/physical") == {
        ("freq",): UnitSpec("Hz", "MHz"),
        ("gain",): UnitSpec("1", "1"),
        ("wiring", "time_of_flight"): UnitSpec("s", "us"),
    }
    model = PhysicalModel.model_validate(
        {"kind": "notebook/physical", "freq": 5.0, "gain": 2, "wiring": {"ch": 3}}
    )
    assert model.wiring.ch == 3
    assert model.gain == 2


def test_registration_preserves_required_wiring_indices_without_physical_units() -> (
    None
):
    RequiredWiring = create_model(
        "RequiredWiring", __base__=WiringSchema, ch=(int, Field(ge=0, strict=True))
    )
    RequiredChannelModel = create_model(
        "RequiredChannelModel",
        __base__=ComponentSchema,
        wiring=(RequiredWiring, ...),
    )

    registry = ComponentRegistry()
    registry.register("notebook/required-channel", RequiredChannelModel)
    assert registry.units("notebook/required-channel") == {
        ("wiring", "time_of_flight"): UnitSpec("s", "us")
    }
    model = RequiredChannelModel.model_validate(
        {"kind": "notebook/required-channel", "wiring": {"ch": 3}}
    )
    assert model.wiring.ch == 3


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
