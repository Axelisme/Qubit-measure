"""Reference editing and source dependency behavior through CfgResource."""

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field

import pytest
from zcu_tools.gui.cfg.binding.ports import ResolvedReference
from zcu_tools.gui.cfg.model import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    LiteralSpec,
    ReferenceSpec,
    ReferenceValue,
    ScalarSpec,
)
from zcu_tools.gui.cfg.resource import (
    CfgEdit,
    CfgInputError,
    CfgResolution,
    CfgResource,
    CfgStatus,
)
from zcu_tools.gui.session.expression import validate_scalar_expr


@dataclass
class Catalog:
    entries: dict[str, ResolvedReference] = field(default_factory=dict)
    lookups: list[str] = field(default_factory=list)

    def keys(self, kind: str, allowed_labels: frozenset[str]) -> Sequence[str]:
        return tuple(
            key for key, value in self.entries.items() if value.label in allowed_labels
        )

    def resolve(self, kind: str, key: str) -> ResolvedReference | None:
        self.lookups.append(key)
        return self.entries.get(key)

    def snapshot(self) -> CfgResolution:
        frozen = Catalog(deepcopy(self.entries), self.lookups)
        return CfgResolution(
            (),
            lambda expression: 0.0,
            lambda source_id: (),
            frozen,
            lambda name: 0.0,
            validate_scalar_expr,
        )


def entry(value: float) -> ResolvedReference:
    return ResolvedReference("Shape", CfgSectionValue({"x": DirectValue(value)}))


@pytest.fixture
def reference_resource() -> tuple[CfgResource, Catalog]:
    shape = CfgSectionSpec(fields={"x": ScalarSpec("X", float)}, label="Shape")
    schema = CfgSchema(
        CfgSectionSpec(fields={"ref": ReferenceSpec("test", [shape], optional=True)}),
        CfgSectionValue(
            {"ref": ReferenceValue("first", CfgSectionValue({"x": DirectValue(1.0)}))}
        ),
    )
    catalog = Catalog({"first": entry(1.0), "second": entry(2.0)})
    resource = CfgResource(
        lambda: schema,
        resolution=catalog.snapshot,
        make_range=lambda start, stop, *, expts: (start, stop, expts),
    )
    return resource, catalog


@pytest.mark.parametrize("reversed_keys", [False, True])
def test_relink_then_children_is_independent_of_key_order(
    reference_resource, reversed_keys: bool
) -> None:
    resource, _ = reference_resource
    pairs = [("__ref", "second"), ("x", 7.0)]
    payload = dict(reversed(pairs) if reversed_keys else pairs)
    changed = resource.edit(
        resource.observe().ref.revision, (CfgEdit(("ref",), payload),)
    )
    value = changed.tree.children["ref"].value
    assert isinstance(value, ReferenceValue)
    assert value.chosen_key == "second"
    assert value.is_overridden
    assert resource.accept(changed.ref.revision).values["ref"] == {"x": 7.0}
    relinked = resource.edit(
        changed.ref.revision, (CfgEdit(("ref", "__ref"), "second"),)
    )
    assert resource.accept(relinked.ref.revision).values["ref"] == {"x": 2.0}
    relinked_value = relinked.tree.children["ref"].value
    assert isinstance(relinked_value, ReferenceValue)
    assert not relinked_value.is_overridden


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


def test_linked_refresh_uses_new_catalog_value(reference_resource) -> None:
    resource, catalog = reference_resource
    before = resource.observe()
    catalog.entries["first"] = entry(10.0)
    changed = resource.refresh(before.ref.revision)
    assert changed.status is CfgStatus.VALID
    assert resource.accept(changed.ref.revision).values["ref"] == {"x": 10.0}
    assert resource.accept(changed.ref.revision).ref == changed.ref


def test_override_no_longer_resolves_original_key(reference_resource) -> None:
    resource, catalog = reference_resource
    changed = resource.edit(
        resource.observe().ref.revision, (CfgEdit(("ref", "x"), 7.0),)
    )
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


def test_nested_edit_detaches_only_ancestors_and_keeps_sibling_link() -> None:
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
        make_range=lambda start, stop, *, expts: (start, stop, expts),
    )
    changed = resource.edit(
        resource.observe().ref.revision, (CfgEdit(("parent", "left", "x"), 7.0),)
    )
    del catalog.entries["parent"]
    del catalog.entries["left"]
    catalog.entries["right"] = entry(20.0)
    catalog.lookups.clear()
    refreshed = resource.refresh(changed.ref.revision)
    assert refreshed.status is CfgStatus.VALID
    assert resource.accept(refreshed.ref.revision).values == {
        "parent": {"left": {"x": 7.0}, "right": {"x": 20.0}}
    }
    assert set(catalog.lookups) == {"right"}


def test_later_edit_uses_shape_selected_by_earlier_relink() -> None:
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
        CfgSectionSpec(fields={"ref": ReferenceSpec("module", [first, second])}),
        CfgSectionValue({"ref": ReferenceValue("first", first_value)}),
    )
    resource = CfgResource(
        lambda: schema,
        resolution=catalog.snapshot,
        make_range=lambda start, stop, *, expts: (start, stop, expts),
    )
    changed = resource.edit(
        resource.observe().ref.revision,
        (CfgEdit(("ref", "__ref"), "second"), CfgEdit(("ref", "y"), 8.0)),
    )
    assert resource.accept(changed.ref.revision).values == {
        "ref": {"type": "second", "y": 8.0}
    }
