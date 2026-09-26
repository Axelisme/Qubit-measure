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
    LiteralSpec,
    ReferenceSpec,
    ReferenceValue,
    ScalarSpec,
    SweepSpec,
    SweepValue,
    make_default_value,
)
from zcu_tools.gui.cfg.binding import (
    CenteredSweepField,
    CfgDraft,
    ReferenceField,
    ResolvedReference,
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


@pytest.mark.parametrize("required", [False, True])
def test_observation_includes_locked_values_and_cached_input_state(
    required: bool,
) -> None:
    ports = BindingPorts()
    ports.expressions["freq"] = 2.0
    ports.options["rig"] = ("flux",)
    spec = CfgSectionSpec(
        fields={
            "fixed": LiteralSpec(7),
            "freq": ScalarSpec("Frequency", float, editable=False),
            "device": ScalarSpec(
                "Device", str, choices_source="rig", required=required
            ),
            "axis": SweepSpec(),
        }
    )
    value = CfgSectionValue(
        {
            "fixed": DirectValue(7),
            "freq": EvalValue("freq"),
            "device": DirectValue("flux"),
            "axis": SweepValue(0.0, 1.0, 5),
        }
    )
    draft = _new_draft(ports, spec, value)
    try:
        sweep = draft.root.fields["axis"]
        assert isinstance(sweep, SweepField)
        sweep.set_text("expts", "1e")
        ports.expressions["freq"] = 9.0
        ports.options["rig"] = ("bias",)
        observed = draft.observe()
        assert not observed.valid
        assert observed.value == draft.snapshot().value
        assert observed.children["fixed"].value == DirectValue(7)
        frequency = observed.children["freq"]
        assert isinstance(frequency.spec, ScalarSpec)
        assert not frequency.spec.editable
        assert frequency.value == EvalValue("freq", resolved=2.0)
        expected_options = ("flux",) if required else ("", "flux")
        assert observed.children["device"].options == expected_options
        axis = observed.children["axis"].value
        assert isinstance(axis, SweepValue)
        assert isinstance(axis.expts, DirectValue)
        assert axis.expts.raw == "1e" and axis.expts.error is not None
        draft.refresh_expressions()
        draft.refresh_options("rig")
        current = draft.observe()
        assert current.children["freq"].value == EvalValue("freq", resolved=9.0)
        assert frequency.value == EvalValue("freq", resolved=2.0)
        device = current.children["device"]
        assert not device.valid
        assert device.options == (("bias",) if required else ("", "bias"))
        assert isinstance(device.value, DirectValue)
        assert device.value.validation_error is not None
    finally:
        draft.close()


def test_observation_preserves_reference_shape_and_disabled_optional_state() -> None:
    ports = BindingPorts()
    shape = CfgSectionSpec(label="Pulse", fields={"gain": ScalarSpec("Gain", float)})
    gain = CfgSectionValue({"gain": DirectValue(0.25)})
    ports.references[("module", "drive_lib")] = ResolvedReference("Pulse", gain)
    spec = CfgSectionSpec(
        fields={"drive": ReferenceSpec("module", [shape], optional=True)}
    )
    draft = _new_draft(
        ports, spec, CfgSectionValue({"drive": ReferenceValue("drive_lib", gain)})
    )
    try:
        ports.references.clear()
        previous = draft.observe().children["drive"]
        assert previous.valid
        assert previous.options == ("Pulse", "drive_lib")
        assert previous.children["gain"].value == DirectValue(0.25)
        assert isinstance(previous.value, ReferenceValue)
        assert previous.value.resolved_label == "Pulse"
        draft.refresh_references()
        missing = draft.observe().children["drive"]
        assert not missing.valid
        assert isinstance(missing.value, ReferenceValue)
        assert missing.value.error is not None
        assert previous.value.error is None
        assert missing.children["gain"].value == DirectValue(0.25)
        reference = draft.root.fields["drive"]
        assert isinstance(reference, ReferenceField)
        reference.set_enabled(False)
        disabled = draft.observe().children["drive"]
        assert disabled.valid
        assert disabled.value is None
        assert disabled.children == {}
    finally:
        draft.close()


def test_observation_data_and_model_are_detached_in_both_directions() -> None:
    draft = _new_draft(
        BindingPorts(),
        CfgSectionSpec(fields={"count": ScalarSpec("Count", int)}),
        CfgSectionValue({"count": DirectValue(2)}),
    )
    try:
        observed = draft.observe()
        draft.set_target("count", 3)
        assert observed.children["count"].value == DirectValue(2)
        assert isinstance(observed.spec, CfgSectionSpec)
        assert isinstance(observed.value, CfgSectionValue)
        observed.spec.fields.clear()
        observed.value.fields["count"] = DirectValue(99)
        observed.children.clear()
        current = draft.observe()
        assert set(current.children) == {"count"}
        assert current.children["count"].value == DirectValue(3)
    finally:
        draft.close()


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
        draft.observe,
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
        reference.available_options,
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
