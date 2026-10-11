"""Reference editing and source dependency behavior through CfgResource."""

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
    CfgInputError,
    CfgInputReason,
    CfgObservation,
    CfgResource,
    CfgStatus,
)

from ._reference_support import Catalog, entry, nested_waveform_resource


def test_ref_children_are_readonly(reference_resource) -> None:
    resource, _ = reference_resource
    before = resource.observe()
    with pytest.raises(CfgInputError) as rejected:
        resource.edit(before.ref.revision, (CfgEdit(("ref", "x"), 7.0),))
    assert rejected.value.reason is CfgInputReason.READONLY
    assert resource.observe() == before
    node = before.tree.children["ref"]
    assert node.mode == "ref"
    assert node.children["x"].readonly


def test_convert_reference_to_inline_copies_published_value(reference_resource) -> None:
    resource, catalog = reference_resource
    accepted = resource.accept(resource.observe().ref.revision)
    catalog.entries["first"] = entry(99.0)
    converted = resource.convert_reference_to_inline(
        resource.observe().ref.revision, ("ref",)
    )
    assert converted.tree.children["ref"].mode == "inline"
    assert not converted.tree.children["ref"].children["x"].readonly
    assert resource.accept(converted.ref.revision).values == {"ref": {"x": 1.0}}
    edited = resource.edit(converted.ref.revision, (CfgEdit(("ref", "x"), 7.0),))
    refreshed = resource.refresh(edited.ref.revision)
    assert resource.accept(refreshed.ref.revision).values == {"ref": {"x": 7.0}}
    assert accepted.values == {"ref": {"x": 1.0}}


@pytest.mark.parametrize("failure", ["missing", "unavailable", "invalid"])
def test_conversion_rejects_invalid_publication_without_cached_copy(
    reference_resource, failure
) -> None:
    resource, catalog = reference_resource
    if failure == "missing":
        del catalog.entries["first"]
    elif failure == "unavailable":
        catalog.failed_keys = frozenset({"first"})
    else:
        catalog.entries["first"] = ResolvedReference(
            "Shape",
            CfgSectionValue({"x": DirectValue(None, raw="1e", error="unfinished")}),
        )
    before = resource.refresh(resource.observe().ref.revision)
    assert before.status is not CfgStatus.VALID
    with pytest.raises(CfgInputError) as rejected:
        resource.convert_reference_to_inline(before.ref.revision, ("ref",))
    assert rejected.value.reason is CfgInputReason.INVALID_VALUE
    assert resource.observe() == before


def test_conversion_freezes_nested_waveform_and_expression() -> None:
    resource, catalog = nested_waveform_resource()
    catalog.entries["rise"] = ResolvedReference(
        "Gauss",
        CfgSectionValue(
            {
                "style": DirectValue("gauss"),
                "length": EvalValue("length"),
                "sigma": DirectValue(0.1),
            }
        ),
    )
    before = resource.refresh(resource.observe().ref.revision)
    converted = resource.convert_reference_to_inline(before.ref.revision, ("ref",))
    nested = converted.tree.children["ref"].children["raise_waveform"]
    assert nested.mode == "inline"
    assert nested.children["length"].value == DirectValue(0.0)
    del catalog.entries["rise"]
    del catalog.entries["parent"]
    refreshed = resource.refresh(converted.ref.revision)
    assert refreshed.status is CfgStatus.VALID
    assert resource.accept(refreshed.ref.revision).values["ref"] == {
        "style": "flat_top",
        "length": 1.0,
        "raise_waveform": {"style": "gauss", "length": 0.0, "sigma": 0.1},
    }


def test_one_off_acceptance_resolves_linked_reference_without_rebinding(
    reference_resource: tuple[CfgResource, Catalog],
) -> None:
    resource, catalog = reference_resource
    before = resource.observe()
    inputs = resource.snapshot_inputs()
    seen: list[CfgObservation] = []
    resource.watch(seen.append)
    temporary = Catalog({"first": entry(8.0)}).snapshot()
    catalog.lookups.clear()

    accepted = resource.accept(before.ref.revision, resolution=temporary)

    assert accepted.values["ref"] == {"x": 8.0}
    assert resource.accept(before.ref.revision).values["ref"] == {"x": 1.0}
    assert resource.observe() == before
    assert resource.snapshot_inputs() == inputs
    assert seen == [before]
    assert catalog.lookups == []


@pytest.mark.parametrize("reversed_keys", [False, True])
def test_ref_and_children_payload_is_rejected_atomically(
    reference_resource, reversed_keys: bool
) -> None:
    resource, _ = reference_resource
    pairs = [("__ref", "second"), ("x", 7.0)]
    payload = dict(reversed(pairs) if reversed_keys else pairs)
    before = resource.observe()
    with pytest.raises(CfgInputError) as rejected:
        resource.edit(
            before.ref.revision,
            (CfgEdit(("ref", "__ref"), "second"), CfgEdit(("ref",), payload)),
        )
    assert rejected.value.reason is CfgInputReason.INVALID_VALUE
    assert resource.observe() == before


@pytest.mark.parametrize(
    "payload", [{"__ref": None, "x": 7.0}, {"__ref": "second", "missing": 7.0}]
)
def test_failed_reference_write_leaves_no_relinked_prefix(
    reference_resource, payload
) -> None:
    resource, _ = reference_resource
    before = resource.observe()
    with pytest.raises(CfgInputError):
        resource.edit(before.ref.revision, (CfgEdit(("ref",), payload),))
    assert resource.observe() == before


def test_disabled_reference_can_be_relinked(reference_resource) -> None:
    resource, _ = reference_resource
    disabled = resource.edit(
        resource.observe().ref.revision, (CfgEdit(("ref", "__ref"), None),)
    )
    assert disabled.tree.children["ref"].value is None
    assert resource.accept(disabled.ref.revision).values == {}
    enabled = resource.edit(
        disabled.ref.revision, (CfgEdit(("ref", "__ref"), "second"),)
    )
    assert resource.accept(enabled.ref.revision).values["ref"] == {"x": 2.0}


def test_complete_inline_replacement_recovers_without_resolving_failed_old_key(
    reference_resource,
) -> None:
    resource, catalog = reference_resource
    catalog.failed_keys = frozenset({"first"})
    unavailable = resource.refresh(resource.observe().ref.revision)
    assert unavailable.status is CfgStatus.UNAVAILABLE
    catalog.lookups.clear()
    changed = resource.edit(unavailable.ref.revision, (CfgEdit(("ref",), {"x": 7.0}),))
    assert changed.status is CfgStatus.VALID
    assert resource.accept(changed.ref.revision).values == {"ref": {"x": 7.0}}
    assert "first" not in catalog.lookups


def test_ref_partial_edit_cannot_recover_by_using_failed_key_cache(
    reference_resource,
) -> None:
    resource, catalog = reference_resource
    catalog.failed_keys = frozenset({"first"})
    unavailable = resource.refresh(resource.observe().ref.revision)
    with pytest.raises(CfgInputError) as rejected:
        resource.edit(unavailable.ref.revision, (CfgEdit(("ref", "x"), 7.0),))
    assert rejected.value.reason is CfgInputReason.READONLY
    assert resource.observe() == unavailable


def test_linked_refresh_uses_new_catalog_value(reference_resource) -> None:
    resource, catalog = reference_resource
    before = resource.observe()
    catalog.entries["first"] = entry(10.0)
    changed = resource.refresh(before.ref.revision)
    assert changed.status is CfgStatus.VALID
    assert resource.accept(changed.ref.revision).values["ref"] == {"x": 10.0}
    assert resource.accept(changed.ref.revision).ref == changed.ref


@pytest.mark.parametrize(
    "edit",
    [CfgEdit(("ref", "x"), 1.0), CfgEdit(("ref",), {})],
)
def test_ref_content_write_rejects_leaf_and_empty_aggregate(
    reference_resource, edit: CfgEdit
) -> None:
    resource, _ = reference_resource
    before = resource.observe()
    with pytest.raises(CfgInputError) as rejected:
        resource.edit(before.ref.revision, (edit,))
    assert rejected.value.reason is CfgInputReason.READONLY
    assert resource.observe() == before


def test_complete_same_value_payload_selects_inline(reference_resource) -> None:
    resource, catalog = reference_resource
    changed = resource.edit(
        resource.observe().ref.revision, (CfgEdit(("ref",), {"x": 1.0}),)
    )
    assert changed.tree.children["ref"].mode == "inline"
    catalog.entries["first"] = entry(20.0)
    catalog.lookups.clear()
    refreshed = resource.refresh(changed.ref.revision)
    assert resource.accept(refreshed.ref.revision).values["ref"] == {"x": 1.0}
    assert "first" not in catalog.lookups


def test_inline_conversion_no_longer_resolves_original_key(reference_resource) -> None:
    resource, catalog = reference_resource
    converted = resource.convert_reference_to_inline(
        resource.observe().ref.revision, ("ref",)
    )
    changed = resource.edit(converted.ref.revision, (CfgEdit(("ref", "x"), 7.0),))
    del catalog.entries["first"]
    catalog.lookups.clear()
    refreshed = resource.refresh(changed.ref.revision)
    assert refreshed.status is CfgStatus.VALID
    assert resource.accept(refreshed.ref.revision).values["ref"] == {"x": 7.0}
    assert "first" not in catalog.lookups
    catalog.entries["first"] = entry(20.0)
    restored = resource.refresh(refreshed.ref.revision)
    assert resource.accept(restored.ref.revision).values["ref"] == {"x": 7.0}
    relinked = resource.edit(
        restored.ref.revision, (CfgEdit(("ref", "__ref"), "first"),)
    )
    assert resource.accept(relinked.ref.revision).values["ref"] == {"x": 20.0}


@pytest.mark.parametrize("written", [1.0, 7.0])
def test_nested_ref_ancestor_rejects_child_edit(
    written: float,
) -> None:
    leaf = CfgSectionSpec(fields={"x": ScalarSpec("X", float)}, label="Shape")
    parent = CfgSectionSpec(
        fields={name: ReferenceSpec("test", [leaf]) for name in ("left", "right")},
        label="Parent",
    )
    children = CfgSectionValue(
        {
            name: ReferenceValue(name, CfgSectionValue({"x": DirectValue(1.0)}))
            for name in ("left", "right")
        }
    )
    catalog = Catalog(
        {
            "parent": ResolvedReference("Parent", children),
            "left": entry(1.0),
            "right": entry(2.0),
        }
    )
    schema = CfgSchema(
        CfgSectionSpec(fields={"parent": ReferenceSpec("test", [parent])}),
        CfgSectionValue({"parent": ReferenceValue("parent", children)}),
    )
    resource = CfgResource(
        lambda: schema,
        resolution=catalog.snapshot,
    )
    before = resource.observe()
    with pytest.raises(CfgInputError) as rejected:
        resource.edit(before.ref.revision, (CfgEdit(("parent", "left", "x"), written),))
    assert rejected.value.reason is CfgInputReason.READONLY
    assert resource.observe() == before


def test_later_edit_uses_shape_selected_by_earlier_inline_replacement() -> None:
    first = CfgSectionSpec(
        fields={"type": LiteralSpec("first"), "x": ScalarSpec("X", float)},
        label="First",
    )
    second = CfgSectionSpec(
        fields={"type": LiteralSpec("second"), "y": ScalarSpec("Y", float)},
        label="Second",
    )
    first_value = CfgSectionValue({"type": DirectValue("first"), "x": DirectValue(1.0)})
    second_value = CfgSectionValue(
        {"type": DirectValue("second"), "y": DirectValue(2.0)}
    )
    catalog = Catalog(
        {
            "first": ResolvedReference("First", first_value),
            "second": ResolvedReference("Second", second_value),
        }
    )
    schema = CfgSchema(
        CfgSectionSpec(
            fields={
                "ref": ReferenceSpec("module", [first, second], discriminator="type")
            }
        ),
        CfgSectionValue({"ref": ReferenceValue("first", first_value)}),
    )
    resource = CfgResource(
        lambda: schema,
        resolution=catalog.snapshot,
    )
    changed = resource.edit(
        resource.observe().ref.revision,
        (CfgEdit(("ref",), {"type": "second", "y": 2.0}), CfgEdit(("ref", "y"), 8.0)),
    )
    assert resource.accept(changed.ref.revision).values == {
        "ref": {"type": "second", "y": 8.0}
    }
    with pytest.raises(CfgInputError):
        resource.edit(
            changed.ref.revision,
            (CfgEdit(("ref",), {"__ref": "second", "type": "first", "x": 9.0}),),
        )
    assert resource.observe() == changed
