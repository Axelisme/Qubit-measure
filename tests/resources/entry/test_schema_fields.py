"""Declared component fields project into editor rows without touching values."""

from __future__ import annotations

from typing import Literal

import pytest
from pydantic import BaseModel, ConfigDict, Field, field_validator
from zcu_tools.resources import entry
from zcu_tools.resources.entry import (
    ComponentSchema,
    ResultEntry,
    UnknownKindError,
    role_candidates,
    snapshot,
)

from .fakes import Profiled
from .view_support import registered_model


def _by_segments(kind: str) -> dict[tuple[str, ...], entry.FieldSpec]:
    return {spec.segments: spec for spec in entry.component_field_specs(kind)}


def test_nested_models_expand_depth_first_after_their_group_row() -> None:
    with registered_model("test/profiled", Profiled) as kind:
        specs = entry.component_field_specs(kind)
    order = [spec.segments for spec in specs]

    assert ("kind",) not in order
    group = order.index(("routing",))
    main = order.index(("routing", "main"))
    main_ch = order.index(("routing", "main", "ch"))
    bias = order.index(("routing", "bias"))
    assert group < main < main_ch < bias
    # Declaration order is kept around the nested block.
    assert order.index(("label",)) < group < order.index(("rate",))
    kinds = {spec.segments: spec.value_kind for spec in specs}
    assert kinds[("routing",)] == "group"
    assert kinds[("routing", "main")] == "group"
    assert kinds[("routing", "main", "ch")] == "integer"


def test_required_unit_and_description_follow_the_declarations() -> None:
    with registered_model("test/profiled", Profiled) as kind:
        specs = _by_segments(kind)

    assert specs[("label",)].required
    assert not specs[("rate",)].required
    # Required is relative to the enclosing group once it is supplied.
    assert specs[("routing", "main", "ch")].required
    assert not specs[("routing", "main")].required
    assert specs[("rate",)].unit == "MHz"
    assert specs[("label",)].unit is None
    assert specs[("rate",)].description == "Sample rate."
    assert specs[("routing", "bias")].description == "Bias line."
    assert specs[("routing", "main")].description is None


def test_value_kinds_and_choices_strip_optional_none() -> None:
    with registered_model("test/profiled", Profiled) as kind:
        specs = _by_segments(kind)

    assert specs[("label",)].value_kind == "text"
    assert specs[("rate",)].value_kind == "number"
    # A type alias wrapping Annotated metadata still projects its base type.
    assert specs[("width",)].value_kind == "number"
    assert specs[("count",)].value_kind == "integer"
    assert specs[("enabled",)].value_kind == "boolean"
    assert specs[("mode",)].value_kind == "choice"
    assert specs[("mode",)].choices == ("A", "V")
    assert specs[("tags",)].value_kind == "mapping"
    assert specs[("ext",)].value_kind == "mapping"
    assert specs[("points",)].value_kind == "yaml"
    assert all(
        spec.choices == () for spec in specs.values() if spec.value_kind != "choice"
    )


class MixedChoice(ComponentSchema):
    mixed: Literal["a", 1] | None = None
    either: int | str | None = None


def test_non_string_literals_and_multi_type_unions_fall_back_to_yaml() -> None:
    with registered_model("test/mixed", MixedChoice) as kind:
        specs = _by_segments(kind)

    assert specs[("mixed",)].value_kind == "yaml"
    assert specs[("mixed",)].choices == ()
    assert specs[("either",)].value_kind == "yaml"


class Aliased(ComponentSchema):
    model_config = ConfigDict(populate_by_name=True)
    internal_name: float | None = Field(default=None, alias="public")


def test_segments_use_the_declared_alias() -> None:
    with registered_model("test/aliased", Aliased) as kind:
        specs = _by_segments(kind)

    assert ("public",) in specs
    assert ("internal_name",) not in specs


def _exploding_factory() -> dict[str, str]:
    raise AssertionError("projection must not build defaults")


class Guarded(ComponentSchema):
    value: float | None = None
    generated: dict[str, str] = Field(default_factory=_exploding_factory)

    @field_validator("value")
    @classmethod
    def refuse(cls, value: object) -> object:
        raise AssertionError(f"projection must not validate {value!r}")


def test_projection_reads_declarations_without_running_factories_or_validators() -> (
    None
):
    with registered_model("test/guarded", Guarded) as kind:
        specs = _by_segments(kind)

    assert specs[("value",)].value_kind == "number"
    assert specs[("generated",)].value_kind == "mapping"
    assert not specs[("generated",)].required


class TreeNode(BaseModel):
    label: str | None = None
    child: TreeNode | None = None


class Recursive(ComponentSchema):
    root: TreeNode | None = None


def test_recursive_models_stop_expanding_at_the_repeated_model() -> None:
    with registered_model("test/recursive", Recursive) as kind:
        specs = _by_segments(kind)

    assert specs[("root",)].value_kind == "group"
    assert specs[("root", "label")].value_kind == "text"
    assert specs[("root", "child")].value_kind == "yaml"
    assert not any(len(segments) > 2 for segments in specs)


class BadUnit(ComponentSchema):
    rate: float | None = Field(default=None, json_schema_extra={"unit": ""})


def test_malformed_unit_metadata_is_rejected() -> None:
    with (
        registered_model("test/bad-unit", BadUnit) as kind,
        pytest.raises(ValueError, match="unit"),
    ):
        entry.component_field_specs(kind)


def test_unknown_kind_raises_the_registry_error() -> None:
    with pytest.raises(UnknownKindError):
        entry.component_field_specs("test/never-registered")


def test_reference_role_marks_top_level_text_fields_named_after_a_role() -> None:
    specs = _by_segments("fake/drive/a")

    # "sense" is a registered role without via; "source" names no role.
    assert specs[("sense",)].reference_role == "sense"
    assert specs[("source",)].reference_role is None
    assert specs[("rate",)].reference_role is None
    assert all(
        spec.reference_role is None for key, spec in specs.items() if len(key) > 1
    )


def test_role_candidates_list_matching_components_in_document_order(
    entry: ResultEntry,
) -> None:
    entry.setup.add_component("D2", kind="fake/drive/b")
    entry.setup.add_component("S1", kind="fake/sensor")
    entry.setup.add_component("D1", kind="fake/drive/a")
    document = snapshot(entry.setup)

    assert role_candidates(document, "driver") == ("D2", "D1")
    assert role_candidates(document, "sense") == ("S1",)
    assert role_candidates(document, "coupler") == ()
    with pytest.raises(ValueError, match="nope"):
        role_candidates(document, "nope")
