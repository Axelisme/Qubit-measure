from __future__ import annotations

import pytest
from zcu_tools.gui.cfg import (
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    ReferenceSpec,
    ReferenceValue,
    ScalarSpec,
    SweepSpec,
    read_value_path,
    replace_value_path,
    resolve_spec_path,
)


def _spec_tree() -> CfgSectionSpec:
    return CfgSectionSpec(
        fields={
            "section": CfgSectionSpec(
                fields={"count": ScalarSpec(label="Count", type=int)}
            ),
            "module": ReferenceSpec(
                kind="module",
                allowed=[
                    CfgSectionSpec(
                        label="A",
                        fields={"gain": ScalarSpec(label="Gain", type=float)},
                    ),
                    CfgSectionSpec(
                        label="B",
                        fields={"gain": ScalarSpec(label="Gain", type=float)},
                    ),
                ],
            ),
        }
    )


def _value_tree() -> CfgSectionValue:
    return CfgSectionValue(
        fields={
            "section": CfgSectionValue(fields={"count": DirectValue(1)}),
            "module": ReferenceValue(
                chosen_key="<Custom:A>",
                value=CfgSectionValue(fields={"gain": DirectValue(0.25)}),
            ),
        }
    )


def test_resolve_spec_path_descends_section() -> None:
    leaf = resolve_spec_path(_spec_tree(), "section.count")

    assert isinstance(leaf, ScalarSpec)
    assert leaf.type is int


def test_resolve_spec_path_descends_all_matching_reference_shapes() -> None:
    leaf = resolve_spec_path(_spec_tree(), "module.gain")

    assert isinstance(leaf, ScalarSpec)
    assert leaf.type is float


def test_resolve_spec_path_rejects_inconsistent_reference_leaf_types() -> None:
    spec = CfgSectionSpec(
        fields={
            "module": ReferenceSpec(
                kind="module",
                allowed=[
                    CfgSectionSpec(
                        label="A",
                        fields={"value": ScalarSpec(label="Value", type=float)},
                    ),
                    CfgSectionSpec(
                        label="B", fields={"value": SweepSpec(label="Value")}
                    ),
                ],
            )
        }
    )

    with pytest.raises(TypeError, match="inconsistent spec types"):
        resolve_spec_path(spec, "module.value")


def test_resolve_spec_path_rejects_inconsistent_scalar_types() -> None:
    spec = CfgSectionSpec(
        fields={
            "module": ReferenceSpec(
                kind="module",
                allowed=[
                    CfgSectionSpec(
                        label="Integer",
                        fields={"value": ScalarSpec(label="Value", type=int)},
                    ),
                    CfgSectionSpec(
                        label="String",
                        fields={"value": ScalarSpec(label="Value", type=str)},
                    ),
                ],
            )
        }
    )

    with pytest.raises(TypeError, match=r"ScalarSpec\[int\].*ScalarSpec\[str\]"):
        resolve_spec_path(spec, "module.value")


def test_resolve_spec_path_rejects_inconsistent_reference_kinds() -> None:
    spec = CfgSectionSpec(
        fields={
            "module": ReferenceSpec(
                kind="module",
                allowed=[
                    CfgSectionSpec(
                        label="A",
                        fields={
                            "nested": ReferenceSpec(
                                kind="module",
                                allowed=[CfgSectionSpec(label="Nested module")],
                            )
                        },
                    ),
                    CfgSectionSpec(
                        label="B",
                        fields={
                            "nested": ReferenceSpec(
                                kind="waveform",
                                allowed=[CfgSectionSpec(label="Nested waveform")],
                            )
                        },
                    ),
                ],
            )
        }
    )

    with pytest.raises(
        TypeError, match=r"ReferenceSpec\[module\].*ReferenceSpec\[waveform\]"
    ):
        resolve_spec_path(spec, "module.nested")


def test_resolve_spec_path_rejects_missing_path() -> None:
    with pytest.raises(KeyError, match="not found"):
        resolve_spec_path(_spec_tree(), "section.missing")


def test_resolve_spec_path_rejects_bad_descent() -> None:
    with pytest.raises(RuntimeError, match="cannot descend"):
        resolve_spec_path(_spec_tree(), "section.count.child")


def test_read_value_path_descends_section_and_reference() -> None:
    value = _value_tree()

    assert read_value_path(value, "section.count") == DirectValue(1)
    assert read_value_path(value, "module.gain") == DirectValue(0.25)


def test_replace_value_path_replaces_existing_leaf() -> None:
    value = _value_tree()

    replace_value_path(value, "module.gain", DirectValue(0.5))

    assert read_value_path(value, "module.gain") == DirectValue(0.5)


def test_replace_value_path_rejects_missing_leaf_without_creating_it() -> None:
    value = _value_tree()

    with pytest.raises(KeyError, match="leaf 'missing' not found"):
        replace_value_path(value, "section.missing", DirectValue(2))

    section = value.fields["section"]
    assert isinstance(section, CfgSectionValue)
    assert "missing" not in section.fields


def test_tree_paths_reject_empty_path() -> None:
    with pytest.raises(RuntimeError, match="must not be empty"):
        resolve_spec_path(_spec_tree(), "")
    with pytest.raises(RuntimeError, match="must not be empty"):
        read_value_path(_value_tree(), "")
