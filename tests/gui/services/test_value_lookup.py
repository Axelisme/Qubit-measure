from __future__ import annotations

from typing import cast

import pytest
from zcu_tools.gui.session.value_lookup import (
    DuplicateValueKey,
    EmptyValueLookup,
    MissingValue,
    ProviderError,
    UnavailableValue,
    ValueInfo,
    ValueKey,
    ValueProviderSpec,
    ValueRef,
    ValueRegistry,
    ValueTypeError,
    decode_value_ref,
    parse_value_ref_text,
    resolve_value_ref,
)


def test_register_get_and_describe_value_source() -> None:
    registry = ValueRegistry()

    registry.register(
        ValueKey("device.flux.value", float),
        lambda: 0.25,
        owner="device:flux",
        description="cached flux value",
    )

    assert registry.get_as("device.flux.value", float) == pytest.approx(0.25)
    assert registry.describe() == (
        ValueInfo(
            key="device.flux.value",
            type_=float,
            owner="device:flux",
            description="cached flux value",
        ),
    )
    assert registry.describe()[0].type_name == "float"


def test_get_default_covers_missing_and_unavailable_values() -> None:
    registry = ValueRegistry()
    registry.register(
        ValueKey("device.flux.value", float),
        lambda: (_ for _ in ()).throw(
            UnavailableValue("device.flux.value", "device disconnected")
        ),
        owner="device:flux",
    )

    assert registry.get_as("missing", float, default=1.0) == pytest.approx(1.0)
    assert registry.get_as("device.flux.value", float, default=2.0) == pytest.approx(
        2.0
    )


def test_missing_unavailable_type_and_provider_errors_are_distinct() -> None:
    registry = ValueRegistry()
    registry.register(
        ValueKey("source.float", float),
        lambda: cast(float, "bad"),
        owner="source",
    )
    registry.register(
        ValueKey("source.broken", float),
        lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        owner="source",
    )

    with pytest.raises(MissingValue):
        registry.get_as("missing", float)
    with pytest.raises(ValueTypeError):
        registry.get_as("source.float", str)
    with pytest.raises(ValueTypeError):
        registry.get_as("source.float", float)
    with pytest.raises(ProviderError) as exc_info:
        registry.get_as("source.broken", float)
    assert exc_info.value.owner == "source"
    assert isinstance(exc_info.value.cause, RuntimeError)


def test_bool_is_not_accepted_as_numeric_value() -> None:
    registry = ValueRegistry()
    registry.register(ValueKey("source.int", int), lambda: True, owner="source")
    registry.register(ValueKey("source.float", float), lambda: False, owner="source")

    with pytest.raises(ValueTypeError):
        registry.get_as("source.int", int)
    with pytest.raises(ValueTypeError):
        registry.get_as("source.float", float)


def test_register_rejects_duplicate_keys() -> None:
    registry = ValueRegistry()
    registry.register(ValueKey("source.value", float), lambda: 1.0, owner="a")

    with pytest.raises(DuplicateValueKey):
        registry.register(ValueKey("source.value", float), lambda: 2.0, owner="b")


def test_replace_owner_is_atomic_on_duplicate_or_conflict() -> None:
    registry = ValueRegistry()
    registry.register(ValueKey("owner.old", float), lambda: 1.0, owner="owner")
    registry.register(ValueKey("other.value", float), lambda: 2.0, owner="other")

    with pytest.raises(DuplicateValueKey):
        registry.replace_owner(
            "owner",
            [
                ValueProviderSpec(
                    ValueKey("owner.new", float), lambda: 3.0, owner="owner"
                ),
                ValueProviderSpec(
                    ValueKey("owner.new", float), lambda: 4.0, owner="owner"
                ),
            ],
        )
    assert registry.get_as("owner.old", float) == pytest.approx(1.0)

    with pytest.raises(DuplicateValueKey):
        registry.replace_owner(
            "owner",
            [
                ValueProviderSpec(
                    ValueKey("other.value", float), lambda: 5.0, owner="owner"
                )
            ],
        )
    assert registry.get_as("owner.old", float) == pytest.approx(1.0)
    assert registry.get_as("other.value", float) == pytest.approx(2.0)


def test_owner_unregister_and_registration_close_are_scoped() -> None:
    registry = ValueRegistry()
    old = registry.register(ValueKey("source.value", float), lambda: 1.0, owner="owner")
    registry.replace_owner(
        "owner",
        [
            ValueProviderSpec(
                ValueKey("source.value", float), lambda: 2.0, owner="owner"
            )
        ],
    )

    old.close()
    assert registry.get_as("source.value", float) == pytest.approx(2.0)

    registry.unregister_owner("owner")
    with pytest.raises(MissingValue):
        registry.get_as("source.value", float)


def test_value_ref_decode_parse_and_resolve() -> None:
    registry = ValueRegistry()
    registry.register(ValueKey("device.flux.value", float), lambda: 0.25, owner="dev")
    registry.register(
        ValueKey("device.flux.name", str), lambda: "flux_yoko", owner="dev"
    )

    assert decode_value_ref(
        {"__kind": "value_ref", "key": "device.flux.value", "type": "float"}
    ) == ValueRef("device.flux.value", "float")
    assert decode_value_ref({"__kind": "eval", "expr": "r_f"}) is None
    assert parse_value_ref_text(" @{device.flux.name} ") == ValueRef("device.flux.name")
    assert resolve_value_ref(ValueRef("device.flux.value"), registry) == pytest.approx(
        0.25
    )
    assert resolve_value_ref(
        ValueRef("device.flux.value", "float"), registry, target_type=float
    ) == pytest.approx(0.25)
    assert resolve_value_ref(ValueRef("device.flux.name"), registry) == "flux_yoko"

    with pytest.raises(ValueTypeError):
        resolve_value_ref(
            ValueRef("device.flux.value", "float"), registry, target_type=str
        )


def test_empty_lookup_uses_default_or_raises_missing() -> None:
    lookup = EmptyValueLookup()

    assert lookup.get_as("missing", float, default=3.0) == pytest.approx(3.0)
    with pytest.raises(MissingValue):
        lookup.get_as("missing", float)
