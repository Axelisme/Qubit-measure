"""Single-fluxonium settings validate external values before resource creation."""

from collections.abc import MutableMapping
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import yaml
from pydantic import ValidationError

if TYPE_CHECKING:
    from zcu_tools.simulators.single_fluxonium.config.schema import (
        FluxLine,
        SingleFluxoniumSettings,
    )


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
    thermal_occupation: null
  adc:
    ro_ch: 0
    gain: 100.0
    background: {ripple_period_mhz: 1000.0, pivot_ghz: 0.0}
    noise_psd_adc2_ns: [[1.0, 0.5], [0.5, 0.25]]
numerics:
  slow_envelope_segments_per_period: 20
  projection_prune_floor: 1.0e-9
  truncation: {native_cutoff: 12, levels: 4}
  solver: {rtol: 0.00000001, atol: 0.0000000001, max_step_ns: 0.1, bath_segment_ns: 2.0}
  seed: 123
"""


@pytest.mark.parametrize(
    "path",
    ["numerics.seed", "numerics.slow_envelope_segments_per_period", "numerics.projection_prune_floor", "readout.resonator.thermal_occupation"],
)
def test_loader_requires_model_inputs(tmp_path: Path, path: str) -> None:
    document = _parse(yaml.safe_load(DOCUMENT)).model_dump(mode="json")
    parent = document
    *parts, field = path.split(".")
    for part in parts:
        parent = parent[part]
    del parent[field]
    model_path = tmp_path / "missing-model-input.yaml"
    model_path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match=path):
        _load(model_path)


@pytest.mark.parametrize("floor", [0.0, 0.25])
def test_projection_floor_accepts_finite_probability_threshold(floor):
    document = yaml.safe_load(DOCUMENT)
    document["numerics"]["projection_prune_floor"] = floor
    assert _parse(document).numerics.projection_prune_floor == floor


@pytest.mark.parametrize("floor", [-0.1, 1.0, float("inf"), float("nan")])
def test_projection_floor_rejects_invalid_probability_threshold(floor):
    document = yaml.safe_load(DOCUMENT)
    document["numerics"]["projection_prune_floor"] = floor
    with pytest.raises(ValueError, match="numerics.projection_prune_floor"):
        _parse(document)


def test_optional_drift_validates_log_ou_parameters_and_positive_loss():
    document = yaml.safe_load(DOCUMENT)
    document["chip"]["qubit"].update(q_cap=1e6, xqp=1e-7, aphi=1e-6)
    document["drift"] = {"q_cap": {"sigma": 0.2, "tau_s": 3600.0}}
    parsed = _parse(document)
    assert parsed.model_dump(mode="json")["drift"]["q_cap"] == {
        "sigma": 0.2,
        "tau_s": 3600.0,
    }
    for invalid in ({"sigma": -0.1, "tau_s": 1.0}, {"sigma": 0.1, "tau_s": 0.0}):
        document["drift"]["q_cap"] = invalid
        with pytest.raises(ValueError, match="drift.q_cap"):
            _parse(document)
    document["drift"] = {"xqp": {"sigma": 0.2, "tau_s": 10.0}}
    document["chip"]["qubit"]["xqp"] = 0.0
    with pytest.raises(ValueError, match="drift.xqp.*positive"):
        _parse(document)
    document["drift"] = {"q_cap": {"sigma": 0.2, "tau_s": 10.0}}
    document["chip"]["qubit"]["q_cap"] = None
    with pytest.raises(ValueError, match="drift.q_cap.*positive"):
        _parse(document)


def test_fixed_round_duration_reports_time_scale_migration():
    document = yaml.safe_load(DOCUMENT)
    document["numerics"]["min_round_duration_s"] = 0.2
    with pytest.raises(ValueError, match=r"numerics.min_round_duration_s.*time_scale"):
        _parse(document)


def _parse(document: object) -> "SingleFluxoniumSettings":
    # Import inside the call so an absent new API fails a test, not collection.
    try:
        from zcu_tools.simulators.single_fluxonium.config import parse_single_fluxonium
    except ModuleNotFoundError as error:
        if error.name != "zcu_tools.simulators.single_fluxonium.config":
            raise
        pytest.fail("single-fluxonium parse seam is not implemented")
    return parse_single_fluxonium(document)


def _load(path: Path) -> "SingleFluxoniumSettings":
    try:
        from zcu_tools.simulators.single_fluxonium.config import load_single_fluxonium
    except ModuleNotFoundError as error:
        if error.name != "zcu_tools.simulators.single_fluxonium.config":
            raise
        pytest.fail("single-fluxonium load seam is not implemented")
    return load_single_fluxonium(path)


def _bound_flux(settings: "SingleFluxoniumSettings") -> "FluxLine":
    if settings.flux_line is None:
        raise AssertionError("pulsed fixture requires a bound fast-flux line")
    return settings.flux_line


def test_named_drive_lines_and_legacy_input_share_immutable_replay_format(
    tmp_path: Path,
) -> None:
    legacy = _parse(yaml.safe_load(DOCUMENT))
    document = yaml.safe_load(DOCUMENT)
    line = document.pop("drive_line")
    document["drive_lines"] = {
        "mw": line,
        "auxiliary": {
            "name": "second",
            "gen_ch": 5,
            "charge_drive_ghz_per_gain": {"real": -0.04, "imag": 0.03},
        },
    }
    parsed = _parse(document)
    values = parsed.model_dump(mode="json")
    assert (
        values["drive_lines"]["mw"]
        == legacy.model_dump(mode="json")["drive_lines"]["mw"]
    )
    assert values["drive_lines"]["auxiliary"]["name"] == "second"
    assert values["drive_lines"]["auxiliary"]["charge_drive_ghz_per_gain"] == {
        "real": -0.04,
        "imag": 0.03,
    }
    assert _parse(values) == parsed
    path = tmp_path / "multiple.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    assert _load(path) == parsed
    assert not isinstance(parsed.drive_lines, MutableMapping)


@pytest.mark.parametrize(
    "field,value,diagnostic",
    [
        ("name", "mw", r"drive_lines.second.name.*drive_lines.mw.name"),
        ("name", "bias", r"drive_lines.second.name.*dc_bias.source.name"),
        ("gen_ch", 1, r"drive_lines.second.gen_ch.*channel 1.*drive_lines.mw.gen_ch"),
        ("gen_ch", 0, r"drive_lines.second.gen_ch.*channel 0.*readout.drive.gen_ch"),
        ("gen_ch", 2, r"flux_line.gen_ch.*channel 2.*drive_lines.second.gen_ch"),
    ],
)
def test_named_drive_lines_reject_collisions_with_both_binding_paths(
    field, value, diagnostic
) -> None:
    document = yaml.safe_load(DOCUMENT)
    document["drive_lines"] = {"mw": document.pop("drive_line")}
    document["drive_lines"]["second"] = {
        "name": "second",
        "gen_ch": 5,
        "charge_drive_ghz_per_gain": {"real": 0.1, "imag": 0.0},
    }
    document["drive_lines"]["second"][field] = value
    with pytest.raises(ValueError, match=diagnostic):
        _parse(document)


def test_drive_line_formats_are_exclusive_and_named_map_must_be_nonempty() -> None:
    document = yaml.safe_load(DOCUMENT)
    document["drive_lines"] = {"mw": document["drive_line"]}
    with pytest.raises(ValueError, match=r"drive_lines.*drive_line"):
        _parse(document)
    del document["drive_line"]
    document["drive_lines"] = {}
    with pytest.raises(ValueError, match="drive_lines"):
        _parse(document)


def test_parse_returns_named_immutable_settings_and_replayable_values() -> None:
    settings = _parse(yaml.safe_load(DOCUMENT))
    assert settings.chip.qubit.ej_ghz == 4.0
    assert settings.dc_bias.source.flux_per_unit == 100.0
    assert settings.drive_lines["mw"].charge_drive_ghz_per_gain.imag == -0.02
    assert _bound_flux(settings).distortion_terms[0].amplitude == -0.1
    assert settings.readout.adc.noise_psd_adc2_ns == ((1.0, 0.5), (0.5, 0.25))
    assert settings.numerics.truncation.levels == 4
    assert _parse(settings.model_dump(mode="json")) == settings
    with pytest.raises(ValidationError, match="frozen"):
        _bound_flux(settings).tau_ns = 5.0


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
    assert _bound_flux(settings).delay_ns == 3.0


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
        ("name: mw", "name: bias", "drive_lines.bias.name"),
        ("name: ff", "name: mw", "flux_line.name"),
        ("gen_ch: 1", "gen_ch: 0", "drive_lines.mw.gen_ch"),
        ("gen_ch: 2", "gen_ch: 1", "flux_line.gen_ch"),
        ("gen_ch: 2", "gen_ch: -1", "flux_line.gen_ch"),
        ("delay_ns: 3.0", "delay_ns: -1", "flux_line.delay_ns"),
        ("tau_ns: 2.0", "tau_ns: .inf", "flux_line.tau_ns"),
        (
            "amplitude: -0.1",
            "amplitude: .nan",
            "flux_line.distortion_terms.0.amplitude",
        ),
        ("tau_ns: 10.0", "tau_ns: 0", "flux_line.distortion_terms.0.tau_ns"),
        (
            "kappa_external_mhz: 1.0",
            "kappa_external_mhz: 0",
            "readout.resonator.kappa_external_mhz",
        ),
        (
            "[[1.0, 0.5], [0.5, 0.25]]",
            "[[1, 2], [2, 1]]",
            "readout.adc.noise_psd_adc2_ns",
        ),
        (
            "[[1.0, 0.5], [0.5, 0.25]]",
            "[[1, 0], [1, 1]]",
            "readout.adc.noise_psd_adc2_ns",
        ),
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
            DOCUMENT.replace(
                "  distortion_terms: [{amplitude: -0.1, tau_ns: 10.0}]\n", ""
            )
            .replace("tau_ns: 2.0", "tau_ns: 0")
            .replace("[[1.0, 0.5], [0.5, 0.25]]", "[[0, 0], [0, 0]]")
            .replace("  seed: 123\n", "  seed: null\n")
        )
    )
    assert _bound_flux(settings).distortion_terms == ()
    assert _bound_flux(settings).tau_ns == 0
    assert settings.chip.qubit.q_cap is None
    assert settings.readout.resonator.thermal_occupation is None
    assert settings.numerics.seed is None
