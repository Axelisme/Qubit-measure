"""Drive-line operator and calibration units at the decoded YAML boundary."""

import pytest
from zcu_tools.resources.ftf import parse_ftf

from tests.resources._ftf_data import ftf_document


def _document(line: dict[str, object]) -> dict[str, object]:
    return ftf_document() | {"drive_lines": {"control": line}}


def _coupling() -> dict[str, object]:
    return {
        "f1": {"real": 0.02, "imag": -0.01},
        "coupler": {"real": -0.03, "imag": 0.04},
        "f2": {"real": 0.0, "imag": 0.0},
    }


@pytest.mark.parametrize("operator", [None, "charge", "flux"])
def test_drive_line_selects_operator_and_units(operator: str | None) -> None:
    field = (
        "flux_drive_phi0_per_gain"
        if operator == "flux"
        else "charge_drive_ghz_per_gain"
    )
    line: dict[str, object] = {"gen_ch": 2, field: _coupling()}
    if operator is not None:
        line["operator"] = operator
    try:
        settings = parse_ftf(_document(line))
    except ValueError as error:
        pytest.fail(f"supported drive operator must validate: {error}")
    control = settings.drive_lines["control"]
    assert control.operator == (operator or "charge")
    # JSON replay must retain the chosen physical calibration without aliases.
    restored = parse_ftf(settings.model_dump(mode="json"))
    assert restored == settings


@pytest.mark.parametrize("operator", ["phase", "", None, 1])
def test_drive_line_rejects_unknown_or_null_operator(operator: object) -> None:
    line = {"gen_ch": 2, "operator": operator, "charge_drive_ghz_per_gain": _coupling()}
    with pytest.raises(ValueError, match=r"drive_lines.control.operator"):
        parse_ftf(_document(line))


@pytest.mark.parametrize(
    "operator,fields,error_field",
    [
        (
            "flux",
            {"charge_drive_ghz_per_gain": _coupling()},
            "flux_drive_phi0_per_gain",
        ),
        (
            "charge",
            {"flux_drive_phi0_per_gain": _coupling()},
            "charge_drive_ghz_per_gain",
        ),
        ("charge", {}, "charge_drive_ghz_per_gain"),
        ("flux", {}, "flux_drive_phi0_per_gain"),
        (
            "charge",
            {
                "charge_drive_ghz_per_gain": _coupling(),
                "flux_drive_phi0_per_gain": _coupling(),
            },
            "flux_drive_phi0_per_gain",
        ),
        (
            "flux",
            {
                "charge_drive_ghz_per_gain": _coupling(),
                "flux_drive_phi0_per_gain": _coupling(),
            },
            "charge_drive_ghz_per_gain",
        ),
    ],
)
def test_drive_line_requires_only_its_operator_calibration(
    operator: str, fields: dict[str, object], error_field: str
) -> None:
    with pytest.raises(ValueError, match=rf"drive_lines.control.*{error_field}"):
        parse_ftf(_document({"gen_ch": 2, "operator": operator, **fields}))


@pytest.mark.parametrize("value", [None, float("nan"), float("inf")])
def test_flux_calibration_requires_finite_complex_components(value: object) -> None:
    coupling = _coupling() | {"f1": {"real": value, "imag": 0.0}}
    line: dict[str, object] = {
        "gen_ch": 2,
        "operator": "flux",
        "flux_drive_phi0_per_gain": coupling,
    }
    with pytest.raises(ValueError, match=r"flux_drive_phi0_per_gain.f1.real"):
        parse_ftf(_document(line))
