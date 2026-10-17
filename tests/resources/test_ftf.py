"""Named FTF input owner rejects invalid wiring/units with field paths."""

from pathlib import Path

import pytest
import yaml

from tests.resources._ftf_data import ftf_document


def test_singular_receiver_spectral_density_is_legal() -> None:
    from zcu_tools.resources.ftf import parse_ftf

    document = ftf_document()
    _set_field(document, "readout.adc.gain", 0.0)
    _set_field(document, "readout.adc.noise_psd_adc2_ns", [[1.0, 0.1], [0.1, 0.01]])
    config = parse_ftf(document)
    assert config.readout.adc.noise_psd_adc2_ns == ((1.0, 0.1), (0.1, 0.01))


def test_ftf_loads_detached_named_settings(tmp_path: Path) -> None:
    from zcu_tools.resources.ftf import load_ftf

    path = tmp_path / "model.yaml"
    path.write_text(yaml.safe_dump(ftf_document()), encoding="utf-8")
    config = load_ftf(path)
    assert config.chip.g_f1_f2_ghz == -0.02
    assert config.dc_bias.sources["bias"].unit == "V"
    assert config.readout.adc.noise_psd_adc2_ns == ((0.0, 0.0), (0.0, 0.0))
    assert config.chip.coupler.ng == 0


def _set_field(document: dict[str, object], path: str, value: object) -> None:
    node = document
    *parents, field = path.split(".")
    for key in parents:
        child = node[key]
        if not isinstance(child, dict):
            raise TypeError("synthetic field path must traverse mappings")
        node = child
    node[field] = value


@pytest.mark.parametrize(
    "path,value,error_path",
    [
        ("mystery", 1, "mystery"),
        ("dc_bias.sources.bias.unit", "mA", "dc_bias.sources.bias.unit"),
        ("dc_bias.sources.bias.initial", 2, "dc_bias.sources.bias.initial"),
        ("dc_bias.sources.bias.limits", [1, -1], "dc_bias.sources.bias"),
        ("dc_bias.sources.bias.ramp_step", 0, "dc_bias.sources.bias.ramp_step"),
        (
            "dc_bias.sources.bias.flux_per_unit",
            {"f1": 0, "coupler": 1},
            "dc_bias.sources.bias.flux_per_unit.f2",
        ),
        (
            "dc_bias.sources.bias.flux_per_unit.f1",
            float("inf"),
            "dc_bias.sources.bias.flux_per_unit.f1",
        ),
        ("drive_lines.mw.gen_ch", 0, "drive_lines.mw.gen_ch"),
        ("drive_lines.mw.gen_ch", -1, "drive_lines.mw.gen_ch"),
        (
            "drive_lines.mw.charge_drive_ghz_per_gain.f1",
            {"real": 0},
            "drive_lines.mw.charge_drive_ghz_per_gain.f1.imag",
        ),
        (
            "readout.adc.noise_psd_adc2_ns",
            [[1, 2], [2, 1]],
            "readout.adc.noise_psd_adc2_ns",
        ),
        ("numerics.solver.rtol", 0, "numerics.solver.rtol"),
        ("numerics.truncation.f1.levels", 9, "numerics.truncation.f1"),
        ("numerics.truncation.coupler.levels", 8, "numerics.truncation.coupler"),
        ("numerics.truncation.retained_levels", 9, "numerics.truncation"),
        ("numerics.seed", -1, "numerics.seed"),
        ("chip.coupler.flux_resistance_ohm", 50, "chip.coupler"),
        ("version", 2.0, "version"),
    ],
)
def test_ftf_field_path_diagnostics(path: str, value: object, error_path: str) -> None:
    from zcu_tools.resources.ftf import parse_ftf

    document = ftf_document()
    _set_field(document, path, value)
    with pytest.raises(ValueError, match=error_path.replace(".", r"\.")):
        parse_ftf(document)


def test_duplicate_dc_and_drive_names_are_rejected() -> None:
    from zcu_tools.resources.ftf import parse_ftf

    document = ftf_document()
    _set_field(
        document,
        "drive_lines.bias",
        {
            "gen_ch": 1,
            "charge_drive_ghz_per_gain": {
                "f1": {"real": 0, "imag": 0},
                "coupler": {"real": 0, "imag": 0},
                "f2": {"real": 0, "imag": 0},
            },
        },
    )
    with pytest.raises(ValueError, match=r"drive_lines.bias: duplicate source name"):
        parse_ftf(document)
