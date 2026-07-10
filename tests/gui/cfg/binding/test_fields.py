from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.cfg import (
    CenteredSweepSpec,
    CenteredSweepValue,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    EvalValue,
    ScalarSpec,
    SweepSpec,
    SweepValue,
)
from zcu_tools.gui.cfg.binding import (
    CenteredSweepField,
    ScalarField,
    SectionField,
    SweepField,
)

from ._fakes import BindingPorts


def test_scalar_field_resolves_expressions_and_refreshes_snapshot() -> None:
    ports = BindingPorts()
    ports.expressions["freq"] = 5
    field = ScalarField(
        ScalarSpec("Frequency", float),
        ports.evaluate,
        ports.provide,
        EvalValue("freq"),
    )

    assert field.get_value() == EvalValue("freq", resolved=5.0)
    ports.expressions["freq"] = 7
    field.refresh_expressions()
    assert field.get_value() == EvalValue("freq", resolved=7.0)


def test_scalar_field_dynamic_options_drive_membership_and_observability() -> None:
    ports = BindingPorts()
    ports.options["rig_devices"] = ("flux",)
    field = ScalarField(
        ScalarSpec(
            "Flux device",
            str,
            choices_source="rig_devices",
            required=True,
        ),
        ports.evaluate,
        ports.provide,
        DirectValue("flux"),
    )
    changed = MagicMock()
    validity_changed = MagicMock()
    field.on_change.connect(changed)
    field.on_validity_changed.connect(validity_changed)

    assert field.available_options() == ("flux",)
    assert field.is_valid()
    ports.options["rig_devices"] = ("bias",)
    field.refresh_options("rig_devices")

    assert field.available_options() == ("bias",)
    assert not field.is_valid()
    changed.assert_called_once()
    validity_changed.assert_called_once_with(False)


def test_scalar_field_optional_unset_remains_valid() -> None:
    ports = BindingPorts()
    field = ScalarField(
        ScalarSpec("Mixer", float, optional=True),
        ports.evaluate,
        ports.provide,
        DirectValue(None),
    )

    assert field.is_valid()


@pytest.mark.parametrize(
    ("declared_type", "initial", "actual_type"),
    [
        (int, DirectValue(True), "bool"),
        (bool, DirectValue(1), "int"),
    ],
)
def test_scalar_field_constructor_strictly_separates_bool_and_int(
    declared_type: type,
    initial: DirectValue,
    actual_type: str,
) -> None:
    ports = BindingPorts()

    with pytest.raises(
        TypeError,
        match=rf"Value.*expects {declared_type.__name__}, got {actual_type}",
    ):
        ScalarField(
            ScalarSpec("Value", declared_type),
            ports.evaluate,
            ports.provide,
            initial,
        )


@pytest.mark.parametrize(
    ("value", "actual_type"),
    [
        (True, "bool"),
        ("7", "str"),
    ],
)
def test_scalar_field_rejects_wrong_runtime_type_before_mutation(
    value: object, actual_type: str
) -> None:
    ports = BindingPorts()
    field = ScalarField(
        ScalarSpec("Repetitions", int),
        ports.evaluate,
        ports.provide,
        DirectValue(7),
    )
    changed = MagicMock()
    field.on_change.connect(changed)

    with pytest.raises(
        TypeError,
        match=rf"Repetitions.*expects int, got {actual_type}",
    ):
        field.set_value(value)

    assert field.get_value() == DirectValue(7)
    changed.assert_not_called()


def test_required_scalar_allows_none_as_existing_invalid_state() -> None:
    ports = BindingPorts()
    field = ScalarField(
        ScalarSpec("Repetitions", int),
        ports.evaluate,
        ports.provide,
        DirectValue(7),
    )

    field.set_value(None)

    assert field.get_value() == DirectValue(None)
    assert not field.is_valid()


def test_section_set_value_rolls_back_invalid_child_and_emits_once_on_success() -> None:
    ports = BindingPorts()
    section = SectionField(
        CfgSectionSpec(
            fields={
                "reps": ScalarSpec("Repetitions", int),
                "name": ScalarSpec("Name", str),
            }
        ),
        evaluate_expression=ports.evaluate,
        provide_options=ports.provide,
        references=ports,
        initial_val=CfgSectionValue(
            {
                "reps": DirectValue(7),
                "name": DirectValue("before"),
            }
        ),
    )
    changed = MagicMock()
    section.on_change.connect(changed)

    with pytest.raises(TypeError, match="Name.*expects str, got int"):
        section.set_value(
            CfgSectionValue(
                {
                    "reps": DirectValue(8),
                    "name": DirectValue(9),
                }
            )
        )

    assert section.get_value() == CfgSectionValue(
        {
            "reps": DirectValue(7),
            "name": DirectValue("before"),
        }
    )
    changed.assert_not_called()

    section.set_value(
        CfgSectionValue(
            {
                "reps": DirectValue(8),
                "name": DirectValue("after"),
            }
        )
    )

    assert section.get_value() == CfgSectionValue(
        {
            "reps": DirectValue(8),
            "name": DirectValue("after"),
        }
    )
    changed.assert_called_once_with()


def test_section_preflight_evaluates_candidate_and_commit_once_each() -> None:
    evaluate_count = 0

    def evaluate(expression: str) -> int:
        nonlocal evaluate_count
        assert expression == "next_reps"
        evaluate_count += 1
        return 11

    ports = BindingPorts()
    section = SectionField(
        CfgSectionSpec(fields={"reps": ScalarSpec("Repetitions", int)}),
        evaluate_expression=evaluate,
        provide_options=ports.provide,
        references=ports,
        initial_val=CfgSectionValue({"reps": DirectValue(7)}),
    )

    section.set_value(CfgSectionValue({"reps": EvalValue("next_reps")}))

    assert evaluate_count == 2
    assert section.get_value() == CfgSectionValue(
        {"reps": EvalValue("next_reps", resolved=11)}
    )


def test_section_unknown_key_fails_before_provider_evaluation_or_mutation() -> None:
    evaluate_count = 0
    provide_count = 0

    def evaluate(expression: str) -> int:
        nonlocal evaluate_count
        assert expression == "next_reps"
        evaluate_count += 1
        return 11

    def provide(source_id: str) -> tuple[str, ...]:
        nonlocal provide_count
        assert source_id == "modes"
        provide_count += 1
        return ("fast",)

    ports = BindingPorts()
    section = SectionField(
        CfgSectionSpec(
            fields={
                "nested": CfgSectionSpec(
                    fields={"reps": ScalarSpec("Repetitions", int)}
                ),
                "mode": ScalarSpec("Mode", str, choices_source="modes"),
            }
        ),
        evaluate_expression=evaluate,
        provide_options=provide,
        references=ports,
        initial_val=CfgSectionValue(
            {
                "nested": CfgSectionValue({"reps": DirectValue(7)}),
                "mode": DirectValue("fast"),
            }
        ),
    )
    evaluate_count = 0
    provide_count = 0
    changed = MagicMock()
    validity_changed = MagicMock()
    section.on_change.connect(changed)
    section.on_validity_changed.connect(validity_changed)
    original = section.get_value()

    with pytest.raises(KeyError, match="unknown field.*'missing'"):
        section.set_value(
            CfgSectionValue(
                {
                    "nested": CfgSectionValue(
                        {
                            "reps": EvalValue("next_reps"),
                            "missing": DirectValue(1),
                        }
                    )
                }
            )
        )

    assert evaluate_count == 0
    assert provide_count == 0
    assert section.get_value() == original
    changed.assert_not_called()
    validity_changed.assert_not_called()


@pytest.mark.parametrize(
    ("spec", "initial", "expected_options", "expected_valid"),
    [
        (
            ScalarSpec(
                "Required",
                str,
                choices_source="empty",
                required=True,
            ),
            DirectValue(""),
            (),
            False,
        ),
        (
            ScalarSpec("Non-required", str, choices_source="empty"),
            DirectValue(""),
            ("",),
            True,
        ),
        (
            ScalarSpec(
                "Optional",
                str,
                choices_source="empty",
                optional=True,
            ),
            DirectValue(None),
            ("",),
            True,
        ),
    ],
)
def test_scalar_field_empty_dynamic_options_respect_required_and_optional_semantics(
    spec: ScalarSpec,
    initial: DirectValue,
    expected_options: tuple[object, ...],
    expected_valid: bool,
) -> None:
    ports = BindingPorts()
    ports.options["empty"] = ()

    field = ScalarField(spec, ports.evaluate, ports.provide, initial)

    assert field.available_options() == expected_options
    assert field.is_valid() is expected_valid


def test_sweep_fields_keep_canonical_step_rules() -> None:
    ports = BindingPorts()
    sweep = SweepField(
        SweepSpec(),
        ports.evaluate,
        SweepValue(0.0, 1.0, 5),
    )
    centered = CenteredSweepField(
        CenteredSweepSpec(),
        ports.evaluate,
        CenteredSweepValue(2.0, 4.0, 5),
    )

    assert sweep.get_value().step == pytest.approx(0.25)
    assert centered.get_value().step == pytest.approx(1.0)
