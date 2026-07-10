"""Contracts for the domain-free paired Spec/Value assembler."""

from __future__ import annotations

import pytest
from zcu_tools.gui.cfg import (
    CfgSchemaAssembler,
    CfgSectionSpec,
    CfgSectionValue,
    ChoiceSectionSpec,
    DirectValue,
    LiteralSpec,
    ReferenceSpec,
    ScalarSpec,
    SweepSpec,
    SweepValue,
)


def test_declare_builds_spec_and_value_in_lockstep_with_consumer_labels() -> None:
    assembler = CfgSchemaAssembler(
        label="Root",
        section_labeler=lambda key: {"group": "Domain Group"}.get(key, key.title()),
    )
    assembler.declare("group.count", ScalarSpec(label="Count", type=int), 3)
    assembler.declare(
        "group.axis",
        SweepSpec(label="Axis"),
        SweepValue(0.0, 1.0, expts=11),
    )

    schema = assembler.build()

    group_spec = schema.spec.fields["group"]
    group_value = schema.value.fields["group"]
    assert isinstance(group_spec, CfgSectionSpec)
    assert group_spec.label == "Domain Group"
    assert list(group_spec.fields) == ["count", "axis"]
    assert isinstance(group_value, CfgSectionValue)
    assert group_value.fields["count"] == DirectValue(3)


def test_declaration_conflicts_fail_before_mutating_an_existing_leaf() -> None:
    assembler = CfgSchemaAssembler()
    assembler.declare("value", ScalarSpec(label="Value", type=int), 1)

    with pytest.raises(ValueError, match="already exists"):
        assembler.declare("value", ScalarSpec(label="Again", type=int), 2)
    with pytest.raises(TypeError, match="cannot descend"):
        assembler.declare("value.child", ScalarSpec(label="Child", type=int), 2)

    schema = assembler.build()
    assert list(schema.spec.fields) == ["value"]
    assert schema.value.fields["value"] == DirectValue(1)


def test_declare_owns_spec_default_and_property_snapshots() -> None:
    child_spec = CfgSectionSpec(
        label="Child",
        fields={"value": ScalarSpec(label="Value", type=int)},
    )
    child_value = CfgSectionValue(fields={"value": DirectValue(3)})
    sweep = SweepValue(0.0, 1.0, expts=11)
    assembler = CfgSchemaAssembler()
    assembler.declare("child", child_spec, child_value)
    assembler.declare("axis", SweepSpec(label="Axis"), sweep)

    child_spec.fields.clear()
    child_value.fields.clear()
    sweep.stop = 99.0
    sweep.expts = 2
    exposed = assembler.spec
    exposed.fields.clear()

    schema = assembler.build()
    built_child_spec = schema.spec.fields["child"]
    built_child_value = schema.value.fields["child"]
    assert isinstance(built_child_spec, CfgSectionSpec)
    assert list(built_child_spec.fields) == ["value"]
    assert isinstance(built_child_value, CfgSectionValue)
    assert built_child_value.fields == {"value": DirectValue(3)}
    built_sweep = schema.value.fields["axis"]
    assert isinstance(built_sweep, SweepValue)
    assert built_sweep.stop == 1.0
    assert built_sweep.expts == 11


def test_batch_preflight_detects_internal_duplicate_and_ancestor_conflict() -> None:
    assembler = CfgSchemaAssembler()

    with pytest.raises(ValueError, match="already exists"):
        assembler.validate_declarations(("a", "a"))
    with pytest.raises(TypeError, match="cannot descend"):
        assembler.validate_declarations(("a", "a.b"))


def test_optional_reference_none_and_literal_alignment_are_structural() -> None:
    shape = CfgSectionSpec(label="Shape", fields={})
    assembler = CfgSchemaAssembler()
    assembler.declare(
        "optional",
        ReferenceSpec(kind="module", allowed=[shape], optional=True),
        None,
    )
    assembler.declare("locked", LiteralSpec(value=7), 99)

    schema = assembler.build()

    assert "optional" in schema.value.fields
    assert schema.value.fields["optional"] is None
    assert schema.value.fields["locked"] == DirectValue(7)


def test_choice_binding_is_applied_only_to_built_snapshot() -> None:
    assembler = CfgSchemaAssembler()
    assembler.ensure_section("options", label="Options")
    assembler.declare(
        "options.mode",
        ScalarSpec(label="Mode", type=str, choices=["a", "b"]),
        "a",
    )
    assembler.declare("options.a_value", ScalarSpec(label="A", type=int), 1)
    assembler.add_choice_binding(
        "options",
        "mode",
        {"a": ("a_value",), "b": ()},
        section_label=None,
    )

    schema = assembler.build()

    assert isinstance(schema.spec.fields["options"], ChoiceSectionSpec)
    assert isinstance(assembler.spec.fields["options"], CfgSectionSpec)
    assert not isinstance(assembler.spec.fields["options"], ChoiceSectionSpec)
    with pytest.raises(RuntimeError, match="already built"):
        assembler.build()
