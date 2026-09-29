"""Reference editing and source dependency behavior through CfgResource."""

from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass, field

import pytest
from zcu_tools.experiment.cfg_editing.catalog import PROGRAM_SHAPES, ProgramSpecPolicy
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
    SweepSpec,
)
from zcu_tools.gui.cfg.resource import (
    CfgEdit,
    CfgInput,
    CfgInputError,
    CfgPreconditionError,
    CfgResolution,
    CfgResource,
    CfgRevision,
    CfgStatus,
)
from zcu_tools.gui.session.expression import validate_scalar_expr


@dataclass
class Catalog:
    entries: dict[str, ResolvedReference] = field(default_factory=dict)
    lookups: list[str] = field(default_factory=list)
    failed_keys: frozenset[str] = frozenset()

    def keys(self, kind: str, allowed_labels: frozenset[str]) -> Sequence[str]:
        return tuple(
            key for key, value in self.entries.items() if value.label in allowed_labels
        )

    def resolve(self, kind: str, key: str) -> ResolvedReference | None:
        self.lookups.append(key)
        if key in self.failed_keys:
            raise RuntimeError("reference resolver defect")
        return self.entries.get(key)

    def snapshot(self) -> CfgResolution:
        frozen = Catalog(deepcopy(self.entries), self.lookups, self.failed_keys)
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


def test_complete_override_recovers_without_resolving_failed_old_key(
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


def test_partial_override_cannot_recover_by_using_failed_key_cache(
    reference_resource,
) -> None:
    resource, catalog = reference_resource
    catalog.failed_keys = frozenset({"first"})
    unavailable = resource.refresh(resource.observe().ref.revision)
    with pytest.raises(RuntimeError, match="reference resolver defect"):
        resource.edit(unavailable.ref.revision, (CfgEdit(("ref", "x"), 7.0),))
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
    [CfgEdit(("ref", "x"), 1.0), CfgEdit(("ref",), {"x": 1.0}), CfgEdit(("ref",), {})],
)
def test_same_value_content_write_detaches_from_source(
    reference_resource, edit: CfgEdit
) -> None:
    resource, catalog = reference_resource
    changed = resource.edit(resource.observe().ref.revision, (edit,))
    catalog.entries["first"] = entry(20.0)
    catalog.lookups.clear()
    refreshed = resource.refresh(changed.ref.revision)
    assert resource.accept(refreshed.ref.revision).values["ref"] == {"x": 1.0}
    assert "first" not in catalog.lookups


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


@pytest.mark.parametrize("written", [1.0, 7.0])
def test_nested_edit_detaches_only_ancestors_and_keeps_sibling_link(
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
        make_range=lambda start, stop, *, expts: (start, stop, expts),
    )
    changed = resource.edit(
        resource.observe().ref.revision, (CfgEdit(("parent", "left", "x"), written),)
    )
    del catalog.entries["parent"]
    del catalog.entries["left"]
    catalog.entries["right"] = entry(20.0)
    catalog.lookups.clear()
    refreshed = resource.refresh(changed.ref.revision)
    assert refreshed.status is CfgStatus.VALID
    assert resource.accept(refreshed.ref.revision).values == {
        "parent": {"left": {"x": written}, "right": {"x": 20.0}}
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
        make_range=lambda start, stop, *, expts: (start, stop, expts),
    )
    changed = resource.edit(
        resource.observe().ref.revision,
        (CfgEdit(("ref", "__ref"), "second"), CfgEdit(("ref", "y"), 8.0)),
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


@pytest.fixture
def switchable_resource() -> CfgResource:
    first = CfgSectionSpec(
        fields={"type": LiteralSpec("first"), "old": ScalarSpec("Old", float)},
        label="First",
    )
    child = CfgSectionSpec(fields={"x": ScalarSpec("X", float)}, label="Child")
    second = CfgSectionSpec(
        fields={
            "type": LiteralSpec("second"),
            "fixed": LiteralSpec("locked"),
            "number": ScalarSpec("Number", float),
            "optional": ScalarSpec("Optional", float, optional=True),
            "child": ReferenceSpec("test", [child]),
            "range": SweepSpec(),
        },
        label="Second",
    )
    schema = CfgSchema(
        CfgSectionSpec(
            fields={"ref": ReferenceSpec("test", [first, second], discriminator="type")}
        ),
        CfgSectionValue(
            {
                "ref": ReferenceValue(
                    "<Custom:First>",
                    CfgSectionValue(
                        {"type": DirectValue("first"), "old": DirectValue(99.0)}
                    ),
                )
            }
        ),
    )
    return CfgResource(
        lambda: schema,
        resolution=Catalog(
            {
                "child": ResolvedReference(
                    "Child", CfgSectionValue({"x": DirectValue(2.0)})
                )
            }
        ).snapshot,
        make_range=lambda start, stop, *, expts: (start, stop, expts),
    )


def complete_second_input() -> dict[str, CfgInput]:
    return {
        "type": "second",
        "number": 3.0,
        "optional": None,
        "child": {"x": 4.0},
        "range": {"step": 2.0, "start": 6.0, "stop": 14.0},
    }


def _waveform_resource(
    *,
    initial_style: str = "gauss",
    catalog: Catalog | None = None,
    chosen_key: str | None = None,
    value: CfgSectionValue | None = None,
    resolution: Callable[[], CfgResolution] | None = None,
) -> CfgResource:
    policy = ProgramSpecPolicy()
    allowed = [shape.make_spec(policy) for shape in PROGRAM_SHAPES.waveforms()]
    old = PROGRAM_SHAPES.waveform(initial_style).make_spec(policy)
    initial = (
        value
        if value is not None
        else CfgSectionValue(
            {
                "style": DirectValue(initial_style),
                "length": DirectValue(0.8),
                "sigma": DirectValue(0.2),
            }
        )
    )
    schema = CfgSchema(
        CfgSectionSpec(
            fields={"ref": ReferenceSpec("waveform", allowed, discriminator="style")}
        ),
        CfgSectionValue(
            {"ref": ReferenceValue(chosen_key or f"<Custom:{old.label}>", initial)}
        ),
    )
    return CfgResource(
        lambda: schema,
        resolution=resolution or (catalog or Catalog()).snapshot,
        make_range=lambda start, stop, *, expts: (start, stop, expts),
    )


def test_select_custom_waveform_inherits_compatible_values_only() -> None:
    resource = _waveform_resource()
    before = resource.observe()

    changed = resource.select_custom_reference(before.ref.revision, ("ref",), "DRAG")

    selected = changed.tree.children["ref"].value
    assert isinstance(selected, ReferenceValue)
    assert selected.chosen_key == "<Custom:DRAG>"
    assert selected.value.fields == {
        "style": DirectValue("drag"),
        "length": DirectValue(0.8),
        "sigma": DirectValue(0.2),
        "delta": DirectValue(0.0),
        "alpha": DirectValue(0.0),
    }
    assert changed.ref.revision == before.ref.revision + 1
    assert resource.accept(changed.ref.revision).values["ref"] == {
        "style": "drag",
        "length": 0.8,
        "sigma": 0.2,
        "delta": 0.0,
        "alpha": 0.0,
    }
    arb = resource.select_custom_reference(changed.ref.revision, ("ref",), "Arb")
    restored = resource.select_custom_reference(arb.ref.revision, ("ref",), "Gauss")
    gauss = restored.tree.children["ref"].value
    assert isinstance(gauss, ReferenceValue)
    assert gauss.value.fields["length"] == DirectValue(0.0)


def test_select_custom_re_resolves_inherited_expression() -> None:
    sources = {"duration": 0.8}
    catalog = Catalog()

    def resolution() -> CfgResolution:
        snapshot = dict(sources)
        return CfgResolution(
            (),
            lambda expression: snapshot[expression],
            lambda source_id: (),
            catalog.snapshot().references,
            lambda name: 0.0,
            validate_scalar_expr,
        )

    resource = _waveform_resource(
        value=CfgSectionValue(
            {
                "style": DirectValue("gauss"),
                "length": EvalValue("duration", resolved=99.0),
                "sigma": DirectValue(0.2),
            }
        ),
        resolution=resolution,
    )
    before = resource.observe()
    length = before.tree.children["ref"].children["length"].value
    assert isinstance(length, EvalValue) and length.resolved == 0.8

    sources["duration"] = 1.4
    changed = resource.select_custom_reference(before.ref.revision, ("ref",), "DRAG")

    length = changed.tree.children["ref"].children["length"].value
    assert isinstance(length, EvalValue)
    assert length.expr == "duration"
    assert length.resolved == 1.4
    assert resource.accept(changed.ref.revision).values["ref"] == {
        "style": "drag",
        "length": 1.4,
        "sigma": 0.2,
        "delta": 0.0,
        "alpha": 0.0,
    }


def test_select_custom_preserves_invalid_inherited_input() -> None:
    resource = _waveform_resource()
    invalid = resource.edit(
        resource.observe().ref.revision,
        (CfgEdit(("ref", "length"), DirectValue(None, raw="-")),),
    )
    assert invalid.status is CfgStatus.INVALID

    switched = resource.select_custom_reference(invalid.ref.revision, ("ref",), "DRAG")

    assert switched.status is CfgStatus.INVALID
    length = switched.tree.children["ref"].children["length"].value
    assert isinstance(length, DirectValue)
    assert length.raw == "-" and length.error is not None
    with pytest.raises(CfgPreconditionError):
        resource.accept(switched.ref.revision)


def _nested_waveform_resource() -> tuple[CfgResource, Catalog]:
    rise = CfgSectionValue(
        {
            "style": DirectValue("gauss"),
            "length": DirectValue(0.4),
            "sigma": DirectValue(0.1),
        }
    )
    parent = CfgSectionValue(
        {
            "style": DirectValue("flat_top"),
            "length": DirectValue(1.0),
            "raise_waveform": ReferenceValue("rise", rise),
        }
    )
    catalog = Catalog(
        {
            "parent": ResolvedReference("FlatTop", parent),
            "rise": ResolvedReference("Gauss", rise),
        }
    )
    return _waveform_resource(
        initial_style="flat_top", catalog=catalog, chosen_key="parent", value=parent
    ), catalog


def test_select_custom_detaches_parent_but_keeps_nested_link() -> None:
    resource, catalog = _nested_waveform_resource()
    parent = deepcopy(catalog.entries["parent"].value)
    assert isinstance(parent, CfgSectionValue)
    parent.fields["length"] = DirectValue(1.6)
    catalog.entries["parent"] = ResolvedReference("FlatTop", parent)
    current = resource.refresh(resource.observe().ref.revision)
    published = resource.accept(current.ref.revision).values["ref"]
    assert isinstance(published, dict) and published["length"] == 1.6
    catalog.failed_keys = frozenset({"parent"})
    catalog.lookups.clear()

    switched = resource.select_custom_reference(
        current.ref.revision, ("ref",), "FlatTop"
    )

    selected = switched.tree.children["ref"].value
    assert isinstance(selected, ReferenceValue)
    assert selected.chosen_key == "<Custom:FlatTop>"
    published = resource.accept(switched.ref.revision).values["ref"]
    assert isinstance(published, dict) and published["length"] == 1.6
    nested = selected.value.fields["raise_waveform"]
    assert isinstance(nested, ReferenceValue)
    assert nested.chosen_key == "rise" and not nested.is_overridden
    assert "parent" not in catalog.lookups
    rise = deepcopy(catalog.entries["rise"].value)
    assert isinstance(rise, CfgSectionValue)
    rise.fields["length"] = DirectValue(0.7)
    catalog.entries["rise"] = ResolvedReference("Gauss", rise)
    refreshed = resource.refresh(switched.ref.revision)
    values = resource.accept(refreshed.ref.revision).values["ref"]
    assert isinstance(values, dict)
    assert values["length"] == 1.6
    nested_values = values["raise_waveform"]
    assert isinstance(nested_values, dict) and nested_values["length"] == 0.7
    assert "parent" not in catalog.lookups


def test_select_custom_nested_waveform_detaches_only_edited_layers() -> None:
    resource, catalog = _nested_waveform_resource()
    current = resource.observe()

    changed = resource.select_custom_reference(
        current.ref.revision, ("ref", "raise_waveform"), "DRAG"
    )

    parent = changed.tree.children["ref"].value
    assert isinstance(parent, ReferenceValue) and parent.is_overridden
    nested = parent.value.fields["raise_waveform"]
    assert isinstance(nested, ReferenceValue)
    assert nested.chosen_key == "<Custom:DRAG>"
    assert nested.value.fields["length"] == DirectValue(0.4)
    assert nested.value.fields["sigma"] == DirectValue(0.1)
    catalog.failed_keys = frozenset({"parent", "rise"})
    catalog.lookups.clear()
    refreshed = resource.refresh(changed.ref.revision)
    assert refreshed.status is CfgStatus.VALID
    assert catalog.lookups == []


def test_select_custom_preserves_nested_override_without_old_source() -> None:
    resource, catalog = _nested_waveform_resource()
    edited = resource.edit(
        resource.observe().ref.revision,
        (CfgEdit(("ref", "raise_waveform", "length"), 0.6),),
    )
    catalog.failed_keys = frozenset({"parent", "rise"})
    catalog.lookups.clear()

    switched = resource.select_custom_reference(
        edited.ref.revision, ("ref",), "FlatTop"
    )

    selected = switched.tree.children["ref"].value
    assert isinstance(selected, ReferenceValue)
    nested = selected.value.fields["raise_waveform"]
    assert isinstance(nested, ReferenceValue)
    assert nested.chosen_key == "rise" and nested.is_overridden
    assert catalog.lookups == []
    values = resource.accept(switched.ref.revision).values["ref"]
    assert isinstance(values, dict)
    nested_values = values["raise_waveform"]
    assert isinstance(nested_values, dict) and nested_values["length"] == 0.6


def test_select_custom_preparation_failure_preserves_publication() -> None:
    resource, catalog = _nested_waveform_resource()
    before = resource.observe()
    catalog.failed_keys = frozenset({"rise"})

    with pytest.raises(RuntimeError, match="reference resolver defect"):
        resource.select_custom_reference(before.ref.revision, ("ref",), "FlatTop")

    assert resource.observe() == before


def test_select_custom_notifies_once_and_rejects_reentrant_selection() -> None:
    resource = _waveform_resource()
    seen: list[int] = []

    def subscriber(observation) -> None:
        seen.append(observation.ref.revision)
        with pytest.raises(CfgPreconditionError):
            resource.select_custom_reference(observation.ref.revision, ("ref",), "DRAG")

    unsubscribe = resource.watch(subscriber)
    changed = resource.select_custom_reference(
        resource.observe().ref.revision, ("ref",), "DRAG"
    )
    assert seen == [0, 1]
    assert resource.observe() == changed
    unsubscribe()


def test_select_custom_uses_default_label_for_unnamed_shape() -> None:
    shape = CfgSectionSpec(fields={"x": ScalarSpec("X", float)})
    schema = CfgSchema(
        CfgSectionSpec(fields={"ref": ReferenceSpec("test", [shape])}),
        CfgSectionValue(
            {
                "ref": ReferenceValue(
                    "<Custom:Custom>", CfgSectionValue({"x": DirectValue(7.0)})
                )
            }
        ),
    )
    resource = CfgResource(
        lambda: schema,
        resolution=Catalog().snapshot,
        make_range=lambda start, stop, *, expts: (start, stop, expts),
    )

    changed = resource.select_custom_reference(
        resource.observe().ref.revision, ("ref",), "Custom"
    )

    assert resource.accept(changed.ref.revision).values == {"ref": {"x": 7.0}}


def test_select_custom_rejects_invalid_choice_without_publishing(
    switchable_resource: CfgResource,
) -> None:
    resource = switchable_resource
    before = resource.observe()
    for path, label in (((), "Second"), (("missing",), "Second"), (("ref",), "Other")):
        with pytest.raises(CfgInputError):
            resource.select_custom_reference(before.ref.revision, path, label)
        assert resource.observe() == before
    with pytest.raises(CfgPreconditionError):
        resource.select_custom_reference(
            CfgRevision(before.ref.revision + 1), ("ref",), "Second"
        )
    assert resource.observe() == before


def test_complete_shape_switch_uses_new_inputs_and_existing_range_owner(
    switchable_resource: CfgResource,
) -> None:
    resource = switchable_resource
    changed = resource.edit(
        resource.observe().ref.revision, (CfgEdit(("ref",), complete_second_input()),)
    )
    assert changed.status is CfgStatus.VALID
    assert resource.accept(changed.ref.revision).values == {
        "ref": {
            "type": "second",
            "fixed": "locked",
            "number": 3.0,
            "child": {"x": 4.0},
            "range": (6.0, 14.0, 5),
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
        resource.edit(before.ref.revision, (CfgEdit(("ref",), payload),))
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
        resource.edit(before.ref.revision, (CfgEdit(("ref",), payload),))
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
        resource.edit(before.ref.revision, (CfgEdit(("ref",), full),))
    assert resource.observe() == before


def test_shape_switch_preserves_expression_and_nested_link_input(
    switchable_resource: CfgResource,
) -> None:
    resource = switchable_resource
    payload = complete_second_input()
    payload["number"] = EvalValue("dynamic")
    payload["child"] = {"__ref": "child"}
    changed = resource.edit(
        resource.observe().ref.revision, (CfgEdit(("ref",), payload),)
    )
    ref = changed.tree.children["ref"]
    number = ref.children["number"].value
    child = ref.children["child"].value
    assert isinstance(number, EvalValue)
    assert number.expr == "dynamic"
    assert isinstance(child, ReferenceValue)
    assert child.chosen_key == "child"
    assert not child.is_overridden


@pytest.mark.parametrize("unavailable", ["disabled", "missing"])
def test_complete_single_shape_input_rebuilds_unavailable_reference(
    reference_resource, unavailable: str
) -> None:
    resource, catalog = reference_resource
    current = resource.observe()
    if unavailable == "disabled":
        current = resource.edit(
            current.ref.revision, (CfgEdit(("ref", "__ref"), None),)
        )
    else:
        del catalog.entries["first"]
        current = resource.refresh(current.ref.revision)
    changed = resource.edit(current.ref.revision, (CfgEdit(("ref",), {"x": 7.0}),))
    assert resource.accept(changed.ref.revision).values == {"ref": {"x": 7.0}}
    refreshed = resource.refresh(changed.ref.revision)
    assert refreshed.status is CfgStatus.VALID


def test_shape_switch_accepts_complete_but_unfinished_text(
    switchable_resource: CfgResource,
) -> None:
    resource = switchable_resource
    payload = complete_second_input()
    payload["number"] = DirectValue(None, raw="-")
    changed = resource.edit(
        resource.observe().ref.revision, (CfgEdit(("ref",), payload),)
    )
    assert changed.status is CfgStatus.INVALID
    number = changed.tree.children["ref"].children["number"].value
    assert isinstance(number, DirectValue)
    assert number.raw == "-"


def test_missing_reference_does_not_offer_stale_children_for_partial_edit(
    reference_resource,
) -> None:
    resource, catalog = reference_resource
    del catalog.entries["first"]
    before = resource.refresh(resource.observe().ref.revision)
    with pytest.raises(CfgInputError):
        resource.edit(before.ref.revision, (CfgEdit(("ref", "x"), 7.0),))
    assert resource.observe() == before


def test_discriminator_is_only_writable_as_aggregate_selection(
    switchable_resource: CfgResource,
) -> None:
    resource = switchable_resource
    before = resource.observe()
    with pytest.raises(CfgInputError):
        resource.edit(before.ref.revision, (CfgEdit(("ref", "type"), "second"),))
    payload = complete_second_input()
    payload["fixed"] = "locked"
    with pytest.raises(CfgInputError):
        resource.edit(before.ref.revision, (CfgEdit(("ref",), payload),))
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
        make_range=lambda start, stop, *, expts: (start, stop, expts),
    )
    before = resource.observe()
    with pytest.raises(CfgInputError):
        resource.edit(before.ref.revision, (CfgEdit(("ref",), payload),))
    assert resource.observe() == before


def test_override_shape_uses_declared_discriminator_not_other_literals() -> None:
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
                    "unavailable",
                    CfgSectionValue(
                        {"variant": DirectValue("two"), "y": DirectValue(4.0)}
                    ),
                    is_overridden=True,
                )
            }
        ),
    )
    resource = CfgResource(
        lambda: schema,
        resolution=Catalog().snapshot,
        make_range=lambda start, stop, *, expts: (start, stop, expts),
    )
    observation = resource.observe()
    assert observation.status is CfgStatus.VALID
    assert resource.accept(observation.ref.revision).values == {
        "ref": {"decoration": "blue", "variant": "two", "y": 4.0}
    }
