"""Single-fluxonium settings validate external values before resource creation."""

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import yaml
from pydantic import ValidationError

if TYPE_CHECKING:
    from zcu_tools.resources.single_fluxonium.schema import SingleFluxoniumSettings


DOCUMENT = """
format: zcu.single-fluxonium
version: 1
chip:
  qubit: {ej_ghz: 4.0, ec_ghz: 1.0, el_ghz: 1.0}
  temperature_k: 0.05
dc_bias:
  offset_phi0: 0.5
  source:
    name: bias
    unit: A
    limits: [-0.02, 0.02]
    initial: 0.001
    ramp_step: 0.000001
    flux_per_unit: 100.0
drive_line:
  name: mw
  gen_ch: 1
  charge_drive_ghz_per_gain: {real: 0.1, imag: -0.02}
flux_line:
  name: ff
  gen_ch: 2
  flux_phi0_per_gain: -0.2
  delay_ns: 3.0
  tau_ns: 2.0
  distortion_terms: [{amplitude: -0.1, tau_ns: 10.0}]
readout:
  drive: {gen_ch: 0, input_per_gain: 10.0}
  resonator:
    frequency_ghz: 7.0
    charge_coefficient_ghz: 0.03
    kappa_external_mhz: 1.0
    kappa_internal_mhz: 0.1
  adc:
    ro_ch: 0
    gain: 100.0
    noise_psd_adc2_ns: [[1.0, 0.5], [0.5, 0.25]]
numerics:
  truncation: {native_cutoff: 12, levels: 4}
  solver: {rtol: 0.00000001, atol: 0.0000000001, max_step_ns: 0.1, bath_segment_ns: 2.0}
  seed: 123
"""


def _parse(document: object) -> "SingleFluxoniumSettings":
    # Import inside the call so an absent new API fails a test, not collection.
    try:
        from zcu_tools.resources.single_fluxonium import parse_single_fluxonium
    except ModuleNotFoundError as error:
        if error.name != "zcu_tools.resources.single_fluxonium":
            raise
        pytest.fail("single-fluxonium parse seam is not implemented")
    return parse_single_fluxonium(document)


def _load(path: Path) -> "SingleFluxoniumSettings":
    try:
        from zcu_tools.resources.single_fluxonium import load_single_fluxonium
    except ModuleNotFoundError as error:
        if error.name != "zcu_tools.resources.single_fluxonium":
            raise
        pytest.fail("single-fluxonium load seam is not implemented")
    return load_single_fluxonium(path)


def test_parse_returns_named_immutable_settings_and_replayable_values() -> None:
    settings = _parse(yaml.safe_load(DOCUMENT))
    assert settings.chip.qubit.ej_ghz == 4.0
    assert settings.dc_bias.source.flux_per_unit == 100.0
    assert settings.drive_line.charge_drive_ghz_per_gain.imag == -0.02
    assert settings.flux_line.distortion_terms[0].amplitude == -0.1
    assert settings.readout.adc.noise_psd_adc2_ns == ((1.0, 0.5), (0.5, 0.25))
    assert settings.numerics.truncation.levels == 4
    assert _parse(settings.model_dump(mode="json")) == settings
    with pytest.raises(ValidationError, match="frozen"):
        settings.flux_line.tau_ns = 5.0


def test_load_reads_utf8_and_honors_yaml_merge_overrides(tmp_path: Path) -> None:
    path = tmp_path / "synthetic.yaml"
    path.write_text(
        DOCUMENT.replace("name: bias", "name: 偏壓").replace(
            "  offset_phi0: 0.5",
            "  <<: {offset_phi0: 0.1}\n  offset_phi0: 0.5",
        ),
        encoding="utf-8",
    )
    settings = _load(path)
    assert settings.dc_bias.source.name == "偏壓"
    assert settings.dc_bias.offset_phi0 == 0.5
    assert settings.flux_line.delay_ns == 3.0


@pytest.mark.parametrize(
    "old,new,field",
    [
        ("version: 1", "version: true", "version"),
        ("version: 1", "version: 1.0", "version"),
        ("version: 1", "version: 2", "version"),
        ("zcu.single-fluxonium", "zcu.ftf", "format"),
        ("ej_ghz: 4.0", "ej_ghz: 0", "chip.qubit.ej_ghz"),
        ("ec_ghz: 1.0", "ec_ghz: '1.0'", "chip.qubit.ec_ghz"),
        ("el_ghz: 1.0", "el_ghz: true", "chip.qubit.el_ghz"),
        ("temperature_k: 0.05", "temperature_k: .nan", "chip.temperature_k"),
        ("unit: A", "unit: mA", "dc_bias.source.unit"),
        ("initial: 0.001", "initial: 1.0", "dc_bias.source.initial"),
        ("[-0.02, 0.02]", "[0.02, -0.02]", "dc_bias.source.limits"),
        ("ramp_step: 0.000001", "ramp_step: 0", "dc_bias.source.ramp_step"),
        ("name: mw", "name: bias", "drive_line.name"),
        ("name: ff", "name: mw", "flux_line.name"),
        ("gen_ch: 1", "gen_ch: 0", "drive_line.gen_ch"),
        ("gen_ch: 2", "gen_ch: 1", "flux_line.gen_ch"),
        ("gen_ch: 2", "gen_ch: -1", "flux_line.gen_ch"),
        ("delay_ns: 3.0", "delay_ns: -1", "flux_line.delay_ns"),
        ("tau_ns: 2.0", "tau_ns: .inf", "flux_line.tau_ns"),
        ("amplitude: -0.1", "amplitude: .nan", "flux_line.distortion_terms.0.amplitude"),
        ("tau_ns: 10.0", "tau_ns: 0", "flux_line.distortion_terms.0.tau_ns"),
        ("kappa_external_mhz: 1.0", "kappa_external_mhz: 0", "readout.resonator.kappa_external_mhz"),
        ("[[1.0, 0.5], [0.5, 0.25]]", "[[1, 2], [2, 1]]", "readout.adc.noise_psd_adc2_ns"),
        ("[[1.0, 0.5], [0.5, 0.25]]", "[[1, 0], [1, 1]]", "readout.adc.noise_psd_adc2_ns"),
        ("levels: 4", "levels: 13", "numerics.truncation.levels"),
        ("levels: 4", "levels: 1", "numerics.truncation.levels"),
        ("rtol: 0.00000001", "rtol: 0", "numerics.solver.rtol"),
        ("seed: 123", "seed: -1", "numerics.seed"),
        ("name: ff", "name: ''", "flux_line.name"),
        ("version: 1", "version: 1\ncoupler: {}", "coupler"),
        ("drive: {gen_ch: 0,", "drive: {kind: mux, gen_ch: 0,", "readout.drive.kind"),
    ],
)
def test_parse_rejects_out_of_contract_fields_with_paths(
    old: str, new: str, field: str
) -> None:
    document = yaml.safe_load(DOCUMENT.replace(old, new))
    # Resolve the seam before raises: an absent API must not satisfy rejection.
    baseline = _parse(yaml.safe_load(DOCUMENT))
    assert baseline.version == 1
    with pytest.raises(ValueError, match=field.replace(".", r"\.")):
        _parse(document)


@pytest.mark.parametrize(
    "text,diagnostic",
    [
        (DOCUMENT.replace("version: 1", "version: 1\nversion: 1"), "duplicate key"),
        (DOCUMENT.replace("tau_ns: 2.0", "tau_ns: -1"), r"flux_line\.tau_ns"),
        ("chip: [", "invalid YAML"),
        ("[]", "model"),
    ],
)
def test_load_reports_file_and_input_diagnostics(
    tmp_path: Path, text: str, diagnostic: str
) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(DOCUMENT, encoding="utf-8")
    assert _load(path).version == 1
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=diagnostic) as error:
        _load(path)
    assert str(path) in str(error.value)


def test_zero_filter_and_noise_and_omitted_optional_values_are_legal() -> None:
    settings = _parse(
        yaml.safe_load(
            DOCUMENT.replace("  distortion_terms: [{amplitude: -0.1, tau_ns: 10.0}]\n", "")
            .replace("tau_ns: 2.0", "tau_ns: 0")
            .replace("[[1.0, 0.5], [0.5, 0.25]]", "[[0, 0], [0, 0]]")
            .replace("  seed: 123\n", "")
        )
    )
    assert settings.flux_line.distortion_terms == ()
    assert settings.flux_line.tau_ns == 0
    assert settings.chip.qubit.q_cap is None
    assert settings.readout.resonator.thermal_occupation is None
    assert settings.numerics.seed is None
