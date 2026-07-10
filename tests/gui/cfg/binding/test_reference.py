from __future__ import annotations

from collections.abc import Sequence
from typing import cast
from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.cfg import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    LiteralSpec,
    ReferenceSpec,
    ReferenceValue,
    ScalarSpec,
)
from zcu_tools.gui.cfg.binding import (
    CfgDraft,
    LibraryBindingState,
    ReferenceField,
    ResolvedReference,
    ScalarField,
)

from ._fakes import BindingPorts


def _shape() -> CfgSectionSpec:
    return CfgSectionSpec(
        label="Pulse",
        fields={
            "type": LiteralSpec("pulse"),
            "gain": ScalarSpec("Gain", float),
        },
    )


def _value(gain: float) -> CfgSectionValue:
    return CfgSectionValue({"type": DirectValue("pulse"), "gain": DirectValue(gain)})


def _failing_reference_spec(*, optional: bool = False) -> ReferenceSpec:
    return ReferenceSpec(
        "module",
        [
            CfgSectionSpec(
                label="Good",
                fields={
                    "type": LiteralSpec("good"),
                    "value": ScalarSpec("Value", int),
                },
            ),
            CfgSectionSpec(
                label="Bad",
                fields={
                    "type": LiteralSpec("bad"),
                    "mode": ScalarSpec(
                        "Mode",
                        str,
                        choices_source="missing",
                    ),
                },
            ),
        ],
        label="Failure probe",
        optional=optional,
    )


def _good_reference_value(key: str = "<Custom:Good>") -> ReferenceValue:
    return ReferenceValue(
        key,
        CfgSectionValue(
            {
                "type": DirectValue("good"),
                "value": DirectValue(1),
            }
        ),
    )


def _draft(
    ports: BindingPorts,
    value: ReferenceValue | None,
    *,
    reference_spec: ReferenceSpec | None = None,
) -> CfgDraft:
    drive_spec = reference_spec or ReferenceSpec("module", [_shape()], label="Drive")
    spec = CfgSectionSpec(fields={"drive": drive_spec})
    return CfgDraft(
        CfgSchema(spec, CfgSectionValue({"drive": value})),
        evaluate_expression=ports.evaluate,
        provide_options=ports.provide,
        references=ports,
    )


def _field(draft: CfgDraft) -> ReferenceField:
    return cast(ReferenceField, draft.root.fields["drive"])


def test_reference_catalog_supplies_compatible_keys_and_linked_snapshot() -> None:
    ports = BindingPorts()
    ports.references[("module", "drive_lib")] = ResolvedReference(
        _shape().label, _value(0.25)
    )
    draft = _draft(ports, ReferenceValue("drive_lib", _value(0.25)))
    field = _field(draft)

    assert field.available_keys() == ("drive_lib",)
    assert field._binding_state is LibraryBindingState.LINKED
    assert field.get_value() == ReferenceValue("drive_lib", _value(0.25))


def test_missing_linked_reference_stays_keyed_and_relinks_when_restored() -> None:
    ports = BindingPorts()
    ports.references[("module", "drive_lib")] = ResolvedReference(
        _shape().label, _value(0.25)
    )
    draft = _draft(ports, ReferenceValue("drive_lib", _value(0.25)))
    field = _field(draft)

    del ports.references[("module", "drive_lib")]
    draft.refresh_references("module")
    assert field.get_chosen_key() == "drive_lib"
    assert field.has_missing_library_ref()
    assert not field.is_valid()

    ports.references[("module", "drive_lib")] = ResolvedReference(
        _shape().label, _value(0.5)
    )
    draft.refresh_references("module")
    assert not field.has_missing_library_ref()
    assert field.is_valid()
    assert field.get_value() == ReferenceValue("drive_lib", _value(0.5))


def test_modified_reference_deletion_heals_to_custom_and_keeps_edits() -> None:
    ports = BindingPorts()
    ports.references[("module", "drive_lib")] = ResolvedReference(
        _shape().label, _value(0.25)
    )
    draft = _draft(ports, ReferenceValue("drive_lib", _value(0.25)))
    field = _field(draft)
    assert field.sub_field is not None
    gain = cast(ScalarField, field.sub_field.fields["gain"])
    gain.set_value(0.75)
    assert field.is_modified()

    del ports.references[("module", "drive_lib")]
    draft.refresh_references("module")

    assert field.get_chosen_key() == "<Custom:Pulse>"
    assert not field.has_missing_library_ref()
    assert field.get_value() == ReferenceValue("<Custom:Pulse>", _value(0.75))


def test_persisted_modified_missing_reference_remains_relinkable() -> None:
    ports = BindingPorts()
    draft = _draft(
        ports,
        ReferenceValue("drive_lib", _value(0.75), is_overridden=True),
    )
    field = _field(draft)

    assert field.get_chosen_key() == "drive_lib"
    assert field.has_missing_library_ref()
    assert field.is_modified()
    assert not field.is_valid()

    ports.references[("module", "drive_lib")] = ResolvedReference(
        _shape().label, _value(0.5)
    )
    draft.refresh_references("module")
    assert field._binding_state is LibraryBindingState.LINKED
    assert field.get_value() == ReferenceValue("drive_lib", _value(0.5))


@pytest.mark.parametrize(
    ("key", "message"),
    [
        ("<Custom:Pulse", "Invalid custom reference key"),
        ("<Custom:Unknown>", "Unknown custom reference label"),
    ],
)
def test_custom_reference_key_must_be_well_formed_and_known(
    key: str, message: str
) -> None:
    with pytest.raises(RuntimeError, match=message):
        _draft(BindingPorts(), ReferenceValue(key, _value(0.25)))


def test_optional_reference_disable_and_reenable_preserves_inline_value() -> None:
    spec = ReferenceSpec("module", [_shape()], label="Drive", optional=True)
    draft = _draft(
        BindingPorts(),
        ReferenceValue("<Custom:Pulse>", _value(0.25)),
        reference_spec=spec,
    )
    field = _field(draft)

    field.set_enabled(False)
    assert field.get_value() is None
    assert field.is_valid()

    field.set_enabled(True)
    assert field.get_value() == ReferenceValue("<Custom:Pulse>", _value(0.25))


def test_catalog_value_aligns_locked_literals_before_binding() -> None:
    ports = BindingPorts()
    ports.references[("module", "drive_lib")] = ResolvedReference(
        "Pulse",
        CfgSectionValue(
            {"type": DirectValue("corrupt-type"), "gain": DirectValue(0.5)}
        ),
    )
    draft = _draft(
        ports,
        ReferenceValue("<Custom:Pulse>", _value(0.25)),
    )
    field = _field(draft)

    field.set_chosen_key("drive_lib")

    assert field.get_value() == ReferenceValue("drive_lib", _value(0.5))


def test_reference_key_refresh_is_observable_without_rebuilding_custom_value() -> None:
    ports = BindingPorts()
    draft = _draft(
        ports,
        ReferenceValue("<Custom:Pulse>", _value(0.25)),
    )
    field = _field(draft)
    changed = MagicMock()
    field.on_change.connect(changed)

    ports.references[("module", "drive_lib")] = ResolvedReference("Pulse", _value(0.5))
    field.refresh_references("module")

    assert field.available_keys() == ("drive_lib",)
    assert field.get_value() == ReferenceValue("<Custom:Pulse>", _value(0.25))
    changed.assert_called_once()


def test_catalog_unsupported_shape_fast_fails() -> None:
    ports = BindingPorts()
    ports.references[("module", "readout_lib")] = ResolvedReference(
        "Direct Readout", CfgSectionValue()
    )

    with pytest.raises(RuntimeError, match="unsupported shape 'Direct Readout'"):
        _draft(ports, ReferenceValue("readout_lib", _value(0.25)))


def test_catalog_supported_shape_without_materialized_value_fast_fails() -> None:
    ports = BindingPorts()
    ports.references[("module", "drive_lib")] = ResolvedReference("Pulse", None)

    with pytest.raises(RuntimeError, match="cannot materialize supported shape"):
        _draft(ports, ReferenceValue("drive_lib", _value(0.25)))


class _ExplodingCatalog(BindingPorts):
    error: ValueError | None = None

    def resolve(self, kind: str, key: str) -> ResolvedReference | None:
        if self.error is not None:
            raise self.error
        return super().resolve(kind, key)


class _CountingCatalog(BindingPorts):
    resolve_count: int = 0

    def resolve(self, kind: str, key: str) -> ResolvedReference | None:
        self.resolve_count += 1
        return super().resolve(kind, key)


class _AllCountingPorts(BindingPorts):
    def __init__(self) -> None:
        super().__init__()
        self.keys_count = 0
        self.resolve_count = 0
        self.options_count = 0
        self.evaluate_count = 0

    def reset_counts(self) -> None:
        self.keys_count = 0
        self.resolve_count = 0
        self.options_count = 0
        self.evaluate_count = 0

    def keys(self, kind: str, allowed_labels: frozenset[str]) -> Sequence[str]:
        self.keys_count += 1
        return super().keys(kind, allowed_labels)

    def resolve(self, kind: str, key: str) -> ResolvedReference | None:
        self.resolve_count += 1
        return super().resolve(kind, key)

    def provide(self, source_id: str) -> Sequence[object]:
        self.options_count += 1
        return super().provide(source_id)

    def evaluate(self, expression: str) -> int | float:
        self.evaluate_count += 1
        return super().evaluate(expression)


def test_reference_payload_unknown_key_has_zero_side_effects() -> None:
    ports = _AllCountingPorts()
    draft = _draft(
        ports,
        ReferenceValue("<Custom:Pulse>", _value(0.25)),
    )
    field = _field(draft)
    old_sub_field = field.sub_field
    assert old_sub_field is not None
    old_value = field.get_value()
    old_binding_state = field._binding_state
    old_sub_change_callbacks = tuple(old_sub_field.on_change._callbacks)
    old_sub_validity_callbacks = tuple(old_sub_field.on_validity_changed._callbacks)
    field_changed = MagicMock()
    root_changed = MagicMock()
    draft_changed = MagicMock()
    field.on_change.connect(field_changed)
    draft.root.on_change.connect(root_changed)
    draft.on_change.connect(draft_changed)
    ports.reset_counts()

    with pytest.raises(KeyError, match="unknown field.*'missing'"):
        draft.root.set_value(
            CfgSectionValue(
                {
                    "drive": ReferenceValue(
                        "<Custom:Pulse>",
                        CfgSectionValue(
                            {
                                "type": DirectValue("pulse"),
                                "gain": DirectValue(0.5),
                                "missing": DirectValue(1),
                            }
                        ),
                    )
                }
            )
        )

    assert (
        ports.keys_count,
        ports.resolve_count,
        ports.options_count,
        ports.evaluate_count,
    ) == (0, 0, 0, 0)
    assert field.get_value() == old_value
    assert field._binding_state is old_binding_state
    assert field.sub_field is old_sub_field
    assert tuple(old_sub_field.on_change._callbacks) == old_sub_change_callbacks
    assert (
        tuple(old_sub_field.on_validity_changed._callbacks)
        == old_sub_validity_callbacks
    )
    field_changed.assert_not_called()
    root_changed.assert_not_called()
    draft_changed.assert_not_called()


def test_direct_reference_constructor_rejects_payload_before_all_ports() -> None:
    ports = _AllCountingPorts()

    with pytest.raises(KeyError, match="unknown field.*'missing'"):
        ReferenceField(
            ReferenceSpec("module", [_shape()], label="Drive"),
            evaluate_expression=ports.evaluate,
            provide_options=ports.provide,
            references=ports,
            initial_val=ReferenceValue(
                "<Custom:Pulse>",
                CfgSectionValue(
                    {
                        "type": DirectValue("pulse"),
                        "gain": DirectValue(0.5),
                        "missing": DirectValue(1),
                    }
                ),
            ),
        )

    assert (
        ports.keys_count,
        ports.resolve_count,
        ports.options_count,
        ports.evaluate_count,
    ) == (0, 0, 0, 0)


def test_direct_reference_setter_rejects_payload_with_zero_side_effects() -> None:
    ports = _AllCountingPorts()
    field = ReferenceField(
        ReferenceSpec("module", [_shape()], label="Drive"),
        evaluate_expression=ports.evaluate,
        provide_options=ports.provide,
        references=ports,
        initial_val=ReferenceValue("<Custom:Pulse>", _value(0.25)),
    )
    old_sub_field = field.sub_field
    assert old_sub_field is not None
    old_value = field.get_value()
    old_binding_state = field._binding_state
    old_change_callbacks = tuple(old_sub_field.on_change._callbacks)
    old_validity_callbacks = tuple(old_sub_field.on_validity_changed._callbacks)
    changed = MagicMock()
    enabled_changed = MagicMock()
    validity_changed = MagicMock()
    field.on_change.connect(changed)
    field.on_enabled_changed.connect(enabled_changed)
    field.on_validity_changed.connect(validity_changed)
    ports.reset_counts()

    with pytest.raises(KeyError, match="unknown field.*'missing'"):
        field.set_value(
            ReferenceValue(
                "<Custom:Pulse>",
                CfgSectionValue(
                    {
                        "type": DirectValue("pulse"),
                        "gain": DirectValue(0.5),
                        "missing": DirectValue(1),
                    }
                ),
            )
        )

    assert (
        ports.keys_count,
        ports.resolve_count,
        ports.options_count,
        ports.evaluate_count,
    ) == (0, 0, 0, 0)
    assert field.get_value() == old_value
    assert field._binding_state is old_binding_state
    assert field.is_enabled
    assert not field.has_missing_library_ref()
    assert field.sub_field is old_sub_field
    assert tuple(old_sub_field.on_change._callbacks) == old_change_callbacks
    assert tuple(old_sub_field.on_validity_changed._callbacks) == old_validity_callbacks
    changed.assert_not_called()
    enabled_changed.assert_not_called()
    validity_changed.assert_not_called()


@pytest.mark.parametrize("operation", ["set_chosen_key", "set_value"])
def test_direct_reference_failed_rebuild_preserves_state_callbacks_and_events(
    operation: str,
) -> None:
    ports = BindingPorts()
    field = ReferenceField(
        _failing_reference_spec(optional=True),
        evaluate_expression=ports.evaluate,
        provide_options=ports.provide,
        references=ports,
        initial_val=_good_reference_value(),
    )
    field.set_enabled(False)
    old_sub_field = field.sub_field
    assert old_sub_field is not None
    old_sub_value = old_sub_field.get_value()
    old_change_callbacks = tuple(old_sub_field.on_change._callbacks)
    old_validity_callbacks = tuple(old_sub_field.on_validity_changed._callbacks)
    changed = MagicMock()
    enabled_changed = MagicMock()
    validity_changed = MagicMock()
    field.on_change.connect(changed)
    field.on_enabled_changed.connect(enabled_changed)
    field.on_validity_changed.connect(validity_changed)

    with pytest.raises(RuntimeError, match="unknown option source 'missing'"):
        if operation == "set_chosen_key":
            field.set_chosen_key("<Custom:Bad>")
        else:
            field.set_value(
                ReferenceValue(
                    "<Custom:Bad>",
                    CfgSectionValue({"mode": DirectValue("bad")}),
                )
            )

    assert field.get_chosen_key() == "<Custom:Good>"
    assert field._binding_state is LibraryBindingState.CUSTOM
    assert not field.has_missing_library_ref()
    assert not field.is_enabled
    assert field.sub_field is old_sub_field
    assert old_sub_field.get_value() == old_sub_value
    assert tuple(old_sub_field.on_change._callbacks) == old_change_callbacks
    assert tuple(old_sub_field.on_validity_changed._callbacks) == old_validity_callbacks
    changed.assert_not_called()
    enabled_changed.assert_not_called()
    validity_changed.assert_not_called()


def test_catalog_rebuild_failure_preserves_reference_and_resolves_once() -> None:
    ports = _CountingCatalog()
    ports.references[("module", "entry")] = ResolvedReference(
        "Good",
        _good_reference_value().value,
    )
    field = _field(
        _draft(
            ports,
            _good_reference_value("entry"),
            reference_spec=_failing_reference_spec(),
        )
    )
    old_sub_field = field.sub_field
    assert old_sub_field is not None
    old_value = field.get_value()
    old_change_callbacks = tuple(old_sub_field.on_change._callbacks)
    old_validity_callbacks = tuple(old_sub_field.on_validity_changed._callbacks)
    changed = MagicMock()
    field.on_change.connect(changed)
    ports.references[("module", "entry")] = ResolvedReference(
        "Bad",
        CfgSectionValue(
            {
                "type": DirectValue("bad"),
                "mode": DirectValue("bad"),
            }
        ),
    )
    ports.resolve_count = 0

    with pytest.raises(RuntimeError, match="unknown option source 'missing'"):
        field.refresh_references("module")

    assert ports.resolve_count == 1
    assert field.get_chosen_key() == "entry"
    assert field._binding_state is LibraryBindingState.LINKED
    assert not field.has_missing_library_ref()
    assert field.is_enabled
    assert field.sub_field is old_sub_field
    assert field.get_value() == old_value
    assert tuple(old_sub_field.on_change._callbacks) == old_change_callbacks
    assert tuple(old_sub_field.on_validity_changed._callbacks) == old_validity_callbacks
    changed.assert_not_called()


def test_nested_section_reference_build_failure_preserves_live_tree() -> None:
    ports = BindingPorts()
    draft = _draft(
        ports,
        _good_reference_value(),
        reference_spec=_failing_reference_spec(),
    )
    field = _field(draft)
    old_sub_field = field.sub_field
    assert old_sub_field is not None
    old_root_value = draft.root.get_value()
    old_change_callbacks = tuple(old_sub_field.on_change._callbacks)
    old_validity_callbacks = tuple(old_sub_field.on_validity_changed._callbacks)
    field_changed = MagicMock()
    root_changed = MagicMock()
    draft_changed = MagicMock()
    field.on_change.connect(field_changed)
    draft.root.on_change.connect(root_changed)
    draft.on_change.connect(draft_changed)

    with pytest.raises(RuntimeError, match="unknown option source 'missing'"):
        draft.root.set_value(
            CfgSectionValue(
                {
                    "drive": ReferenceValue(
                        "<Custom:Bad>",
                        CfgSectionValue({"mode": DirectValue("bad")}),
                    )
                }
            )
        )

    assert draft.root.get_value() == old_root_value
    assert field.get_chosen_key() == "<Custom:Good>"
    assert field._binding_state is LibraryBindingState.CUSTOM
    assert not field.has_missing_library_ref()
    assert field.is_enabled
    assert field.sub_field is old_sub_field
    assert tuple(old_sub_field.on_change._callbacks) == old_change_callbacks
    assert tuple(old_sub_field.on_validity_changed._callbacks) == old_validity_callbacks
    field_changed.assert_not_called()
    root_changed.assert_not_called()
    draft_changed.assert_not_called()


def test_reference_catalog_refresh_resolves_linked_key_once() -> None:
    ports = _CountingCatalog()
    ports.references[("module", "drive_lib")] = ResolvedReference(
        "Pulse",
        _value(0.25),
    )
    field = _field(_draft(ports, ReferenceValue("drive_lib", _value(0.25))))
    ports.resolve_count = 0

    field.refresh_references("module")

    assert ports.resolve_count == 1


def test_catalog_resolver_exception_is_not_downgraded_to_missing() -> None:
    ports = _ExplodingCatalog()
    ports.references[("module", "drive_lib")] = ResolvedReference("Pulse", _value(0.25))
    field = _field(_draft(ports, ReferenceValue("drive_lib", _value(0.25))))
    ports.error = ValueError("Unknown reference text from corrupt decoder")

    with pytest.raises(ValueError, match="corrupt decoder"):
        field.refresh_references("module")

    assert not field.has_missing_library_ref()


def test_nested_reference_builds_nested_reference_field() -> None:
    nested_spec = ReferenceSpec("module", [_shape()], label="Nested drive")
    container_spec = CfgSectionSpec(
        label="Container",
        fields={"nested": nested_spec},
    )
    nested_value = ReferenceValue("<Custom:Pulse>", _value(0.75))
    container_value = CfgSectionValue({"nested": nested_value})
    ports = BindingPorts()
    ports.references[("module", "container_lib")] = ResolvedReference(
        "Container", container_value
    )
    reference_spec = ReferenceSpec(
        "module",
        [container_spec],
        label="Container reference",
    )
    field = _field(
        _draft(
            ports,
            ReferenceValue("container_lib", container_value),
            reference_spec=reference_spec,
        )
    )

    assert field.sub_field is not None
    nested = field.sub_field.fields["nested"]
    assert isinstance(nested, ReferenceField)
    assert nested.get_value() == nested_value
