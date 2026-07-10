from __future__ import annotations

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
