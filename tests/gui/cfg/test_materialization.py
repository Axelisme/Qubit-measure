from __future__ import annotations

from dataclasses import dataclass

import pytest
from zcu_tools.gui.cfg import (
    RAW_MISSING,
    CenteredSweepSpec,
    CenteredSweepValue,
    CfgSectionSpec,
    CfgSectionValue,
    ChoiceBinding,
    ChoiceSectionSpec,
    DirectValue,
    LiteralSpec,
    RawMissing,
    ReferenceMaterialization,
    ReferenceSpec,
    ReferenceValue,
    ScalarSpec,
    SweepSpec,
    SweepValue,
    make_default_value,
    materialize_spec_value,
)


@dataclass(frozen=True)
class _RecordingPolicy:
    paths: list[tuple[str, ...]]

    def scalar_value(
        self,
        path: tuple[str, ...],
        spec: ScalarSpec,
        raw: object | RawMissing,
    ) -> DirectValue:
        del spec
        self.paths.append(path)
        return DirectValue(None if raw is RAW_MISSING else raw)

    def sweep_value(
        self,
        path: tuple[str, ...],
        spec: SweepSpec,
        raw: object | RawMissing,
    ) -> SweepValue:
        del spec, raw
        self.paths.append(path)
        return SweepValue(0.0, 1.0, 3)

    def centered_sweep_value(
        self,
        path: tuple[str, ...],
        spec: CenteredSweepSpec,
        raw: object | RawMissing,
    ) -> CenteredSweepValue:
        del spec, raw
        self.paths.append(path)
        return CenteredSweepValue(0.5, 1.0, 3)

    def missing_section_value(
        self,
        path: tuple[str, ...],
        spec: CfgSectionSpec,
        raw: object | RawMissing,
    ) -> CfgSectionValue:
        del raw
        self.paths.append(path)
        return make_default_value(spec)

    def reference_value(
        self,
        path: tuple[str, ...],
        spec: ReferenceSpec,
        raw: object | RawMissing,
    ) -> ReferenceMaterialization | None:
        self.paths.append(path)
        return ReferenceMaterialization(
            spec=spec.allowed[0],
            raw=raw,
            chosen_key="<Custom:Child>",
        )


def test_materialize_spec_value_walks_complete_union_without_domain_knowledge() -> None:
    choice = ChoiceSectionSpec(
        label="Choice",
        fields={
            "mode": LiteralSpec("a"),
            "active": ScalarSpec("Active", int),
            "inactive": ScalarSpec("Inactive", int),
        },
        bindings=(
            ChoiceBinding(
                selector_key="mode",
                choices={
                    "a": CfgSectionSpec(fields={"active": ScalarSpec("Active", int)})
                },
            ),
        ),
    )
    child = CfgSectionSpec(
        label="Child",
        fields={"value": ScalarSpec("Value", float)},
    )
    spec = CfgSectionSpec(
        fields={
            "locked": LiteralSpec("fixed"),
            "scalar": ScalarSpec("Scalar", int),
            "sweep": SweepSpec(),
            "centered": CenteredSweepSpec(),
            "choice": choice,
            "missing_section": CfgSectionSpec(
                fields={"ignored": ScalarSpec("Ignored", bool)}
            ),
            "reference": ReferenceSpec(kind="asset", allowed=[child]),
        }
    )
    paths: list[tuple[str, ...]] = []

    value = materialize_spec_value(
        spec,
        {
            "locked": "fixed",
            "scalar": 7,
            "choice": {"mode": "a", "active": 1, "inactive": 9},
            "missing_section": "not-a-mapping",
            "reference": {"value": 2.5},
        },
        policy=_RecordingPolicy(paths),
    )

    assert tuple(value.fields) == tuple(spec.fields)
    assert value.fields["locked"] == DirectValue("fixed")
    assert value.fields["scalar"] == DirectValue(7)
    assert isinstance(value.fields["sweep"], SweepValue)
    assert isinstance(value.fields["centered"], CenteredSweepValue)
    choice_value = value.fields["choice"]
    assert isinstance(choice_value, CfgSectionValue)
    assert tuple(choice_value.fields) == ("mode", "active", "inactive")
    assert choice_value.fields["inactive"] == DirectValue(9)
    assert value.fields["missing_section"] == CfgSectionValue(
        fields={"ignored": DirectValue(False)}
    )
    reference = value.fields["reference"]
    assert isinstance(reference, ReferenceValue)
    assert reference.value.fields["value"] == DirectValue(2.5)
    assert ("choice", "inactive") in paths
    assert ("missing_section",) in paths
    assert ("reference", "value") in paths


def test_materialize_spec_value_always_applies_literal_lock() -> None:
    spec = CfgSectionSpec(fields={"kind": LiteralSpec("expected")})

    value = materialize_spec_value(
        spec,
        {"kind": "other"},
        policy=_RecordingPolicy([]),
    )

    assert value.fields["kind"] == DirectValue("expected")


def test_materialize_spec_value_rejects_unknown_spec_type() -> None:
    class _UnknownSpec:
        pass

    spec = CfgSectionSpec(fields={"unknown": _UnknownSpec()})  # type: ignore[dict-item]

    with pytest.raises(TypeError, match="Unsupported cfg spec node"):
        materialize_spec_value(spec, {}, policy=_RecordingPolicy([]))


@dataclass(frozen=True)
class _MalformedMissingPolicy(_RecordingPolicy):
    malformed: object

    def missing_section_value(
        self,
        path: tuple[str, ...],
        spec: CfgSectionSpec,
        raw: object | RawMissing,
    ) -> CfgSectionValue:
        del path, spec, raw
        return self.malformed  # type: ignore[return-value]


@pytest.mark.parametrize("raw", [RAW_MISSING, "not-a-mapping"])
def test_policy_supplied_ordinary_section_requires_exact_fields_and_order(
    raw: object | RawMissing,
) -> None:
    nested = CfgSectionSpec(
        fields={
            "first": ScalarSpec("First", int),
            "second": ScalarSpec("Second", int),
        }
    )
    spec = CfgSectionSpec(fields={"nested": nested})
    root_raw = {} if raw is RAW_MISSING else {"nested": raw}
    malformed = CfgSectionValue(
        fields={
            "second": DirectValue(2),
            "first": DirectValue(1),
        }
    )

    with pytest.raises(
        ValueError,
        match=r"nested.*expected fields/order \('first', 'second'\).*actual fields/order \('second', 'first'\)",
    ):
        materialize_spec_value(
            spec,
            root_raw,
            policy=_MalformedMissingPolicy([], malformed),
        )


@pytest.mark.parametrize("raw", [RAW_MISSING, "not-a-mapping"])
def test_policy_supplied_reference_section_requires_exact_shape(
    raw: object | RawMissing,
) -> None:
    child = CfgSectionSpec(
        label="Child",
        fields={"expected": ScalarSpec("Expected", int)},
    )
    spec = CfgSectionSpec(
        fields={"reference": ReferenceSpec(kind="asset", allowed=[child])}
    )
    root_raw = {} if raw is RAW_MISSING else {"reference": raw}

    with pytest.raises(
        ValueError,
        match=r"reference.*expected fields/order \('expected',\).*actual fields/order \('wrong',\)",
    ):
        materialize_spec_value(
            spec,
            root_raw,
            policy=_MalformedMissingPolicy(
                [],
                CfgSectionValue(fields={"wrong": DirectValue(1)}),
            ),
        )


def test_policy_supplied_missing_choice_requires_complete_union() -> None:
    choice = ChoiceSectionSpec(
        fields={
            "mode": LiteralSpec("a"),
            "active": ScalarSpec("Active", int),
            "inactive": ScalarSpec("Inactive", int),
        },
        bindings=(
            ChoiceBinding(
                selector_key="mode",
                choices={
                    "a": CfgSectionSpec(fields={"active": ScalarSpec("Active", int)})
                },
            ),
        ),
    )
    spec = CfgSectionSpec(fields={"choice": choice})

    with pytest.raises(
        ValueError,
        match=r"choice.*expected fields/order.*inactive.*actual fields/order",
    ):
        materialize_spec_value(
            spec,
            {},
            policy=_MalformedMissingPolicy(
                [],
                CfgSectionValue(
                    fields={
                        "mode": DirectValue("a"),
                        "active": DirectValue(1),
                    }
                ),
            ),
        )


def test_policy_supplied_section_validates_nested_node_kind() -> None:
    nested = CfgSectionSpec(
        fields={"inner": CfgSectionSpec(fields={"leaf": ScalarSpec("Leaf", int)})}
    )
    spec = CfgSectionSpec(fields={"outer": nested})

    with pytest.raises(
        ValueError,
        match=r"outer\.inner.*expected CfgSectionValue.*actual DirectValue",
    ):
        materialize_spec_value(
            spec,
            {},
            policy=_MalformedMissingPolicy(
                [],
                CfgSectionValue(fields={"inner": DirectValue(1)}),
            ),
        )
