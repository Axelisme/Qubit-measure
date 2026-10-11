"""Complete reference shape replacement and unresolved linked-shape diagnostics."""

import pytest
from zcu_tools.gui.cfg.binding.ports import ResolvedReference
from zcu_tools.gui.cfg.model import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    EvalValue,
    LiteralSpec,
    ReferenceSpec,
    ReferenceValue,
    ScalarSpec,
)
from zcu_tools.gui.cfg.resource import (
    CfgEdit,
    CfgInput,
    CfgInputError,
    CfgPreconditionError,
    CfgResource,
    CfgStatus,
)

from ._reference_support import Catalog


def complete_second_input() -> dict[str, CfgInput]:
    return {
        "type": "second",
        "number": 3.0,
        "optional": None,
        "child": {"x": 4.0},
        "range": {"step": 2.0, "start": 6.0, "stop": 14.0},
    }


def test_complete_shape_switch_uses_new_inputs_and_existing_range_owner(
    switchable_resource: CfgResource,
) -> None:
    resource = switchable_resource
    changed = resource.edit(
        (CfgEdit(("ref",), complete_second_input()),)
    )
    assert changed.status is CfgStatus.VALID
    assert resource.accept().values == {
        "ref": {
            "type": "second",
            "fixed": "locked",
            "number": 3.0,
            "child": {"x": 4.0},
            "range": {"start": 6.0, "stop": 14.0, "expts": 5, "step": 2.0},
        }
    }


@pytest.mark.parametrize("missing", ["number", "optional", "child", "range"])
def test_shape_switch_does_not_fill_missing_input_defaults(
    switchable_resource: CfgResource, missing: str
) -> None:
    resource = switchable_resource
    before = resource.observe()
    payload = complete_second_input()
    del payload[missing]
    with pytest.raises(CfgInputError):
        resource.edit((CfgEdit(("ref",), payload),))
    assert resource.observe() == before


@pytest.mark.parametrize(
    "replacement", [{}, {"start": 6.0, "stop": 14.0}, {"start": 6.0, "step": 2.0}]
)
def test_shape_switch_requires_complete_nested_range(
    switchable_resource: CfgResource, replacement
) -> None:
    resource = switchable_resource
    before = resource.observe()
    payload = complete_second_input()
    payload["range"] = replacement
    with pytest.raises(CfgInputError):
        resource.edit((CfgEdit(("ref",), payload),))
    assert resource.observe() == before


@pytest.mark.parametrize(
    "payload",
    [
        {"type": "unknown"},
        {"type": "second", "child": {}},
        {"type": "second", "child": {"x": 4.0}, "unknown": 1.0},
    ],
)
def test_invalid_shape_payload_preserves_publication(
    switchable_resource: CfgResource, payload
) -> None:
    resource = switchable_resource
    before = resource.observe()
    full = complete_second_input()
    full.update(payload)
    with pytest.raises(CfgInputError):
        resource.edit((CfgEdit(("ref",), full),))
    assert resource.observe() == before


def test_shape_switch_preserves_expression_and_nested_link_input(
    switchable_resource: CfgResource,
) -> None:
    resource = switchable_resource
    payload = complete_second_input()
    payload["number"] = EvalValue("dynamic")
    payload["child"] = {"__ref": "child"}
    changed = resource.edit(
        (CfgEdit(("ref",), payload),)
    )
    ref = changed.tree.children["ref"]
    number = ref.children["number"].value
    child = ref.children["child"].value
    assert isinstance(number, EvalValue)
    assert number.expr == "dynamic"
    assert isinstance(child, ReferenceValue)
    assert child.chosen_key == "child"


@pytest.mark.parametrize("unavailable", ["disabled", "missing"])
def test_complete_single_shape_input_rebuilds_unavailable_reference(
    reference_resource, unavailable: str
) -> None:
    resource, catalog = reference_resource
    resource.observe()
    if unavailable == "disabled":
        resource.edit(
            (CfgEdit(("ref", "__ref"), None),)
        )
    else:
        del catalog.entries["first"]
        resource.refresh()
    resource.edit((CfgEdit(("ref",), {"x": 7.0}),))
    assert resource.accept().values == {"ref": {"x": 7.0}}
    refreshed = resource.refresh()
    assert refreshed.status is CfgStatus.VALID


def test_shape_switch_accepts_complete_but_unfinished_text(
    switchable_resource: CfgResource,
) -> None:
    resource = switchable_resource
    payload = complete_second_input()
    payload["number"] = DirectValue(None, raw="-")
    changed = resource.edit(
        (CfgEdit(("ref",), payload),)
    )
    assert changed.status is CfgStatus.INVALID
    number = changed.tree.children["ref"].children["number"].value
    assert isinstance(number, DirectValue)
    assert number.raw == "-"


def _two_shape_schema(reference: ReferenceValue) -> CfgSchema:
    """Two discriminator-selected shapes for unresolved-reference contracts."""
    first = CfgSectionSpec(
        fields={"type": LiteralSpec("first"), "x": ScalarSpec("X", float)},
        label="First",
    )
    second = CfgSectionSpec(
        fields={"type": LiteralSpec("second"), "y": ScalarSpec("Y", float)},
        label="Second",
    )
    return CfgSchema(
        CfgSectionSpec(
            fields={"ref": ReferenceSpec("test", [first, second], discriminator="type")}
        ),
        CfgSectionValue({"ref": reference}),
    )


@pytest.mark.parametrize("second", [False, True], ids=["first", "second"])
def test_empty_linked_reference_waits_for_catalog_shape(second: bool) -> None:
    key = "alias.module.slot"
    schema = _two_shape_schema(ReferenceValue(key, CfgSectionValue()))
    catalog = Catalog()
    resource = CfgResource(lambda: schema, resolution=catalog.snapshot)
    initial = resource.observe()
    assert initial.status is CfgStatus.INVALID
    node = initial.tree.children["ref"]
    assert node.children == {}
    assert isinstance(node.value, ReferenceValue)
    assert node.value.value == CfgSectionValue()
    assert any(
        diagnostic.path == ("ref",) and key in diagnostic.message
        for diagnostic in initial.diagnostics
    )
    with pytest.raises(CfgPreconditionError):
        resource.accept()
    field_name = "y" if second else "x"
    type_name = "second" if second else "first"
    catalog.entries[key] = ResolvedReference(
        "Second" if second else "First",
        CfgSectionValue({"type": DirectValue(type_name), field_name: DirectValue(2.0)}),
    )
    refreshed = resource.refresh()
    assert refreshed.status is CfgStatus.VALID
    assert resource.accept().values == {
        "ref": {"type": type_name, field_name: 2.0}
    }


def test_missing_reference_diagnostic_survives_valid_cached_children(
    reference_resource,
) -> None:
    resource, catalog = reference_resource
    del catalog.entries["first"]
    missing = resource.refresh()
    assert missing.status is CfgStatus.INVALID
    node = missing.tree.children["ref"]
    assert isinstance(node.value, ReferenceValue)
    assert node.children["x"].valid
    assert any(
        diagnostic.path == ("ref",)
        and diagnostic.message == node.value.error
        and "first" in diagnostic.message
        for diagnostic in missing.diagnostics
    )


@pytest.mark.parametrize(
    ("reference", "error"),
    [
        (
            ReferenceValue("linked", CfgSectionValue({"type": DirectValue("unknown")})),
            RuntimeError,
        ),
        (
            ReferenceValue(
                "linked",
                CfgSectionValue(
                    {
                        "type": DirectValue("first"),
                        "x": DirectValue(1.0),
                        "unknown": DirectValue(2.0),
                    }
                ),
            ),
            KeyError,
        ),
    ],
    ids=["linked-bad-type", "linked-extra-field"],
)
def test_only_unresolved_linked_empty_payload_defers_shape_selection(
    reference: ReferenceValue, error: type[Exception]
) -> None:
    schema = _two_shape_schema(reference)
    with pytest.raises(error):
        CfgResource(lambda: schema, resolution=Catalog().snapshot)


def test_custom_empty_reference_keeps_explicit_shape_and_defaults() -> None:
    schema = _two_shape_schema(ReferenceValue("<Custom:First>", CfgSectionValue()))
    resource = CfgResource(lambda: schema, resolution=Catalog().snapshot)
    initial = resource.observe()
    assert initial.status is CfgStatus.VALID
    node = initial.tree.children["ref"]
    assert set(node.children) == {"type", "x"}
    assert isinstance(node.value, ReferenceValue)
    assert node.value.error is None
    selected = resource.accept().values["ref"]
    assert isinstance(selected, dict)
    assert selected["type"] == "first"
    assert "x" in selected


def test_missing_reference_does_not_offer_stale_children_for_partial_edit(
    reference_resource,
) -> None:
    resource, catalog = reference_resource
    del catalog.entries["first"]
    before = resource.refresh()
    assert before.status is CfgStatus.INVALID
    with pytest.raises(CfgPreconditionError):
        resource.accept()
    with pytest.raises(CfgInputError):
        resource.edit((CfgEdit(("ref", "x"), 7.0),))
    assert resource.observe() == before


def test_discriminator_is_only_writable_as_aggregate_selection(
    switchable_resource: CfgResource,
) -> None:
    resource = switchable_resource
    before = resource.observe()
    with pytest.raises(CfgInputError):
        resource.edit((CfgEdit(("ref", "type"), "second"),))
    payload = complete_second_input()
    payload["fixed"] = "locked"
    with pytest.raises(CfgInputError):
        resource.edit((CfgEdit(("ref",), payload),))
    assert resource.observe() == before


@pytest.mark.parametrize("payload", [{}, {"number": 3.0}])
def test_complete_input_never_fabricates_readonly_values(payload) -> None:
    shape = CfgSectionSpec(
        fields={"number": ScalarSpec("Number", float, editable=False)}, label="Readonly"
    )
    schema = CfgSchema(
        CfgSectionSpec(fields={"ref": ReferenceSpec("test", [shape], optional=True)}),
        CfgSectionValue({"ref": None}),
    )
    resource = CfgResource(
        lambda: schema,
        resolution=Catalog().snapshot,
    )
    before = resource.observe()
    with pytest.raises(CfgInputError):
        resource.edit((CfgEdit(("ref",), payload),))
    assert resource.observe() == before


def test_inline_shape_uses_declared_discriminator_not_other_literals() -> None:
    first = CfgSectionSpec(
        fields={
            "decoration": LiteralSpec("red"),
            "variant": LiteralSpec("one"),
            "x": ScalarSpec("X", float),
        },
        label="First",
    )
    second = CfgSectionSpec(
        fields={
            "decoration": LiteralSpec("blue"),
            "variant": LiteralSpec("two"),
            "y": ScalarSpec("Y", float),
        },
        label="Second",
    )
    schema = CfgSchema(
        CfgSectionSpec(
            fields={
                "ref": ReferenceSpec("test", [first, second], discriminator="variant")
            }
        ),
        CfgSectionValue(
            {
                "ref": ReferenceValue(
                    "<Custom:Second>",
                    CfgSectionValue(
                        {"variant": DirectValue("two"), "y": DirectValue(4.0)}
                    ),
                )
            }
        ),
    )
    resource = CfgResource(
        lambda: schema,
        resolution=Catalog().snapshot,
    )
    observation = resource.observe()
    assert observation.status is CfgStatus.VALID
    assert resource.accept().values == {
        "ref": {"decoration": "blue", "variant": "two", "y": 4.0}
    }
