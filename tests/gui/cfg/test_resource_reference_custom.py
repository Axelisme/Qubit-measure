"""Custom reference shape selection and compatible-input inheritance."""

from copy import deepcopy

import pytest
from zcu_tools.gui.cfg.binding.ports import ResolvedReference
from zcu_tools.gui.cfg.model import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    EvalValue,
    ReferenceSpec,
    ReferenceValue,
    ScalarSpec,
)
from zcu_tools.gui.cfg.resource import (
    CfgEdit,
    CfgInputError,
    CfgInputReason,
    CfgPreconditionError,
    CfgResolution,
    CfgResource,
    CfgRevision,
    CfgStatus,
)
from zcu_tools.gui.session.expression import validate_scalar_expr

from ._reference_support import Catalog, nested_waveform_resource, waveform_resource


def test_select_custom_waveform_inherits_compatible_values_only() -> None:
    resource = waveform_resource()
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

    resource = waveform_resource(
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
    resource = waveform_resource()
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


def test_select_custom_detaches_parent_but_keeps_nested_link() -> None:
    resource, catalog = nested_waveform_resource()
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
    assert nested.chosen_key == "rise"
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


def test_select_custom_nested_waveform_rejects_ref_ancestor() -> None:
    resource, _ = nested_waveform_resource()
    current = resource.observe()
    with pytest.raises(CfgInputError) as rejected:
        resource.select_custom_reference(
            current.ref.revision, ("ref", "raise_waveform"), "DRAG"
        )
    assert rejected.value.reason is CfgInputReason.READONLY
    assert resource.observe() == current


def test_select_custom_preserves_nested_inline_without_old_source() -> None:
    resource, catalog = nested_waveform_resource()
    converted = resource.convert_reference_to_inline(
        resource.observe().ref.revision, ("ref",)
    )
    edited = resource.edit(
        converted.ref.revision, (CfgEdit(("ref", "raise_waveform", "length"), 0.6),)
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
    assert nested.chosen_key == "<Custom:Gauss>"
    assert catalog.lookups == []
    values = resource.accept(switched.ref.revision).values["ref"]
    assert isinstance(values, dict)
    nested_values = values["raise_waveform"]
    assert isinstance(nested_values, dict) and nested_values["length"] == 0.6


def test_select_custom_preparation_failure_preserves_publication() -> None:
    resource, catalog = nested_waveform_resource()
    before = resource.observe()
    catalog.failed_keys = frozenset({"rise"})

    with pytest.raises(RuntimeError, match="reference resolver defect"):
        resource.select_custom_reference(before.ref.revision, ("ref",), "FlatTop")

    assert resource.observe() == before


def test_select_custom_notifies_once_and_rejects_reentrant_selection() -> None:
    resource = waveform_resource()
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
