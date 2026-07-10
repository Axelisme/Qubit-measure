from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.cfg import (
    CenteredSweepSpec,
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    EvalValue,
    ReferenceSpec,
    ReferenceValue,
    ScalarSpec,
    SweepSpec,
    make_default_value,
)
from zcu_tools.gui.cfg.binding import (
    CenteredSweepField,
    CfgDraft,
    ReferenceField,
    ScalarField,
    SectionField,
    SweepField,
)

from ._fakes import BindingPorts


def _new_draft(
    ports: BindingPorts,
    spec: CfgSectionSpec,
    value: CfgSectionValue | None = None,
) -> CfgDraft:
    return CfgDraft(
        CfgSchema(spec, value or make_default_value(spec)),
        evaluate_expression=ports.evaluate,
        provide_options=ports.provide,
        references=ports,
    )


def test_close_invalidates_cached_root_and_scalar_and_is_idempotent() -> None:
    ports = BindingPorts()
    ports.expressions["x"] = 1
    spec = CfgSectionSpec(fields={"x": ScalarSpec("X", int)})
    value = make_default_value(spec).with_field("x", EvalValue("x"))
    draft = _new_draft(ports, spec, value)
    root = draft.root
    child = cast(ScalarField, root.fields["x"])

    draft.close()
    draft.close()

    draft_operations = (
        lambda: draft.root,
        draft.snapshot,
        draft.is_valid,
        draft.refresh_expressions,
        draft.refresh_options,
        draft.refresh_references,
    )
    field_operations = (
        root.get_value,
        lambda: root.set_value(CfgSectionValue()),
        root.is_valid,
        root.refresh_expressions,
        root.refresh_options,
        root.refresh_references,
        child.get_value,
        lambda: child.set_value(2),
        child.is_valid,
        child.available_options,
        child.refresh_expressions,
        child.refresh_options,
        child.refresh_references,
    )
    for operation in (*draft_operations, *field_operations):
        with pytest.raises(RuntimeError, match="closed"):
            operation()


def test_close_invalidates_range_fields_and_cached_range_children() -> None:
    ports = BindingPorts()
    spec = CfgSectionSpec(
        fields={
            "sweep": SweepSpec(),
            "centered": CenteredSweepSpec(),
        }
    )
    draft = _new_draft(ports, spec)
    sweep = cast(SweepField, draft.root.fields["sweep"])
    start = sweep.start_field
    centered = cast(CenteredSweepField, draft.root.fields["centered"])
    center = centered.center_field

    draft.close()

    operations = (
        sweep.get_value,
        lambda: sweep.update_expts(5),
        lambda: sweep.update_step(0.5),
        sweep.refresh_expressions,
        lambda: start.set_value(DirectValue(2.0)),
        start.get_value,
        centered.get_value,
        lambda: centered.update_span(2.0),
        lambda: centered.update_expts(5),
        lambda: centered.update_step(0.5),
        centered.refresh_expressions,
        lambda: center.set_value(DirectValue(2.0)),
        center.get_value,
    )
    for operation in operations:
        with pytest.raises(RuntimeError, match="closed"):
            operation()


def test_close_invalidates_reference_public_surface_and_nested_field() -> None:
    ports = BindingPorts()
    shape = CfgSectionSpec(
        label="Pulse",
        fields={"gain": ScalarSpec("Gain", float)},
    )
    spec = CfgSectionSpec(
        fields={
            "drive": ReferenceSpec(
                "module",
                [shape],
                label="Drive",
                optional=True,
            )
        }
    )
    value = CfgSectionValue(
        {
            "drive": ReferenceValue(
                "<Custom:Pulse>",
                CfgSectionValue({"gain": DirectValue(0.25)}),
            )
        }
    )
    draft = _new_draft(ports, spec, value)
    reference = cast(ReferenceField, draft.root.fields["drive"])
    assert reference.sub_field is not None
    nested = cast(ScalarField, reference.sub_field.fields["gain"])

    draft.close()

    operations = (
        reference.available_keys,
        reference.is_modified,
        reference.has_missing_library_ref,
        reference.get_chosen_key,
        lambda: reference.is_enabled,
        lambda: reference.set_chosen_key("other"),
        lambda: reference.set_enabled(False),
        reference.get_value,
        lambda: reference.set_value(None),
        reference.is_valid,
        reference.refresh_expressions,
        reference.refresh_options,
        reference.refresh_references,
        nested.get_value,
        lambda: nested.set_value(0.5),
    )
    for operation in operations:
        with pytest.raises(RuntimeError, match="closed"):
            operation()


def test_nested_section_transaction_has_no_transient_draft_validity_event() -> None:
    ports = BindingPorts()
    nested_spec = CfgSectionSpec(
        fields={
            "left": ScalarSpec("Left", int),
            "right": ScalarSpec("Right", int),
        }
    )
    spec = CfgSectionSpec(fields={"nested": nested_spec})
    draft = _new_draft(
        ports,
        spec,
        CfgSectionValue(
            {
                "nested": CfgSectionValue(
                    {
                        "left": DirectValue(None),
                        "right": DirectValue(2),
                    }
                )
            }
        ),
    )
    nested = cast(SectionField, draft.root.fields["nested"])
    nested_validity = MagicMock()
    root_validity = MagicMock()
    draft_validity = MagicMock()
    root_changed = MagicMock()
    draft_changed = MagicMock()
    nested.on_validity_changed.connect(nested_validity)
    draft.root.on_validity_changed.connect(root_validity)
    draft.on_validity_changed.connect(draft_validity)
    draft.root.on_change.connect(root_changed)
    draft.on_change.connect(draft_changed)

    nested.set_value(
        CfgSectionValue(
            {
                "left": DirectValue(1),
                "right": DirectValue(None),
            }
        )
    )

    assert not nested.is_valid()
    assert not draft.is_valid()
    nested_validity.assert_not_called()
    root_validity.assert_not_called()
    draft_validity.assert_not_called()
    root_changed.assert_called_once_with()
    draft_changed.assert_called_once_with()


def test_section_constructor_closes_completed_children_on_later_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ports = BindingPorts()
    ports.options["good"] = ("a",)
    closed_labels: list[str] = []
    original_teardown = ScalarField.teardown

    def record_teardown(field: ScalarField) -> None:
        closed_labels.append(field.spec.label)
        original_teardown(field)

    monkeypatch.setattr(ScalarField, "teardown", record_teardown)
    spec = CfgSectionSpec(
        fields={
            "first": ScalarSpec("First", str, choices_source="good"),
            "second": ScalarSpec("Second", str, choices_source="missing"),
        }
    )

    with pytest.raises(RuntimeError, match="unknown option source 'missing'"):
        _new_draft(ports, spec)

    assert closed_labels == ["First"]
