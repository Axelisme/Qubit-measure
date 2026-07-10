from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.cfg import (
    CenteredSweepSpec,
    CenteredSweepValue,
    CfgSectionSpec,
    DirectValue,
    EvalValue,
    ScalarSpec,
    SweepSpec,
    SweepValue,
)
from zcu_tools.gui.cfg.binding import (
    CenteredSweepField,
    ScalarField,
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
