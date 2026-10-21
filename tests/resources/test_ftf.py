"""Named FTF input owner rejects invalid wiring/units with field paths."""

from pathlib import Path

import pytest
import yaml
from zcu_tools.resources.ftf import load_ftf, parse_ftf

from tests.resources._ftf_data import dual_readout_document, ftf_document


def test_singular_receiver_spectral_density_is_legal() -> None:
    document = ftf_document()
    _set_field(document, "readout.adc.gain", 0.0)
    _set_field(document, "readout.adc.noise_psd_adc2_ns", [[1.0, 0.1], [0.1, 0.01]])
    config = parse_ftf(document)
    assert config.readout.receivers[0].noise_psd_adc2_ns == ((1.0, 0.1), (0.1, 0.01))


def test_ftf_loads_detached_named_settings(tmp_path: Path) -> None:
    from zcu_tools.resources.ftf import load_ftf

    document = ftf_document()
    path = tmp_path / "model.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    config = load_ftf(path)
    document.clear()
    assert config.chip.g_f1_f2_ghz == -0.02
    assert config.dc_bias.sources["bias"].unit == "V"
    assert config.readout.receivers[0].noise_psd_adc2_ns == ((0.0, 0.0), (0.0, 0.0))
    assert config.chip.coupler.ng == 0
    assert config.dc_bias.sources["bias"].flux_per_unit.coupler == 0.1
    assert config.dc_bias.offset_phi0.f2 == 0.4
    assert config.drive_lines["mw"].coefficients().f2.imag == 0.02
    assert config.readout.resonators.r2.frequency_ghz == 6.2
    assert config.numerics.truncation.f1.native_cutoff == 8
    assert config.numerics.truncation.retained_levels == 8
    assert config.readout.receivers[0].offset_adc == (0.0, 0.0)


def test_ftf_load_preserves_merge_defaults_and_explicit_overrides(
    tmp_path: Path,
) -> None:
    from zcu_tools.resources.ftf import load_ftf

    text = (
        yaml.safe_dump(ftf_document())
        .replace(
            "    bias:\n",
            "    bias:\n      <<: &defaults {initial: 0.6, ramp_step: 0.3}\n",
        )
        .replace("      ramp_step: 0.1\n", "")
    )
    path = tmp_path / "merged.yaml"
    path.write_text(text, encoding="utf-8")

    config = load_ftf(path)

    assert config.dc_bias.sources["bias"].initial == 0.2
    assert config.dc_bias.sources["bias"].ramp_step == 0.3
    assert config.dc_bias.sources["bias"].unit == "V"


@pytest.mark.parametrize(
    "name,key_line,first_value",
    [
        (
            "bias",
            "    bias:\n",
            "{unit: A, limits: [-2, 2], initial: 0.5, ramp_step: 0.2, "
            "flux_per_unit: {f1: 0, coupler: 0.2, f2: 0}}",
        ),
        (
            "mw",
            "  mw:\n",
            "{gen_ch: 3, charge_drive_ghz_per_gain: "
            "{f1: {real: 0.1, imag: 0}, coupler: {real: 0, imag: 0}, "
            "f2: {real: 0, imag: 0}}}",
        ),
    ],
)
def test_duplicate_yaml_keys_are_rejected_before_overwrite(
    tmp_path: Path, name: str, key_line: str, first_value: str
) -> None:
    from zcu_tools.resources.ftf import load_ftf

    text = yaml.safe_dump(ftf_document())
    text = text.replace(key_line, f"{key_line.rstrip()} {first_value}\n{key_line}")
    path = tmp_path / "duplicate.yaml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(
        ValueError, match=rf"(?s)duplicate.*{name}.*line [0-9]+"
    ) as error:
        load_ftf(path)
    assert str(path) in str(error.value)


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
    document = ftf_document()
    _set_field(document, path, value)
    with pytest.raises(ValueError, match=error_path.replace(".", r"\.")):
        parse_ftf(document)


def test_duplicate_dc_and_drive_names_are_rejected() -> None:
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


@pytest.mark.parametrize(
    "field,value",
    [
        ("flux_lines", {}),
        ("flux_lines", {"line": {"gen_ch": 1}}),
        ("flux_sources", []),
        ("channels", {"generators": 3}),
    ],
)
def test_v2_rejects_removed_and_unknown_top_level_fields(
    field: str, value: object
) -> None:
    document = ftf_document()
    try:
        parse_ftf(document)
    except ValueError as error:
        pytest.fail(f"v2 base must validate before checking {field}: {error}")
    with pytest.raises(ValueError, match=field):
        parse_ftf(document | {field: value})


def test_version_one_is_rejected_at_version_field() -> None:
    with pytest.raises(ValueError, match=r"version:.*2"):
        parse_ftf(ftf_document() | {"version": 1})


@pytest.mark.parametrize(
    "section,path",
    [
        ("dc_bias", "offset_phi0.f2"),
        ("drive_lines", "mw.charge_drive_ghz_per_gain.f2"),
        ("readout", "resonators.r1.charge_coefficients_ghz.f2"),
    ],
)
def test_v2_requires_every_named_loop(section: str, path: str) -> None:
    document = ftf_document()
    try:
        parse_ftf(document)
    except ValueError as error:
        pytest.fail(f"v2 base must validate before checking loop maps: {error}")
    # Work with decoded YAML just as a caller editing a file would.
    node = document[section]
    for key in path.split(".")[:-1]:
        assert isinstance(node, dict)
        node = node[key]
    assert isinstance(node, dict)
    del node["f2"]
    with pytest.raises(ValueError, match=(section + "." + path).replace(".", r"\.")):
        parse_ftf(document)


def test_receiver_delay_and_independent_mismatch_defaults_and_roundtrip() -> None:
    defaults = parse_ftf(ftf_document())
    assert defaults.readout.receivers[0].time_of_flight_ns == 0.0
    assert defaults.readout.resonators.r1.mismatch_phase_rad == 0.0
    assert defaults.readout.resonators.r2.mismatch_phase_rad == 0.0
    document = ftf_document()
    _set_field(document, "readout.adc.time_of_flight_ns", 400.0)
    _set_field(document, "readout.resonators.r1.mismatch_phase_rad", 0.6)
    _set_field(document, "readout.resonators.r2.mismatch_phase_rad", -0.4)
    configured = parse_ftf(document)
    restored = parse_ftf(configured.model_dump(mode="json"))
    assert restored.readout.receivers[0].time_of_flight_ns == 400.0
    assert restored.readout.resonators.r1.mismatch_phase_rad == 0.6
    assert restored.readout.resonators.r2.mismatch_phase_rad == -0.4


@pytest.mark.parametrize("value", [None, -1.0, float("nan"), float("inf"), "400"])
def test_receiver_delay_invalid_values_report_field_path(value: object) -> None:
    # A valid nonzero value must parse before testing rejection diagnostics.
    document = ftf_document()
    _set_field(document, "readout.adc.time_of_flight_ns", 400.0)
    assert parse_ftf(document).readout.receivers[0].time_of_flight_ns == 400.0
    _set_field(document, "readout.adc.time_of_flight_ns", value)
    with pytest.raises(ValueError, match=r"readout\.adc\.time_of_flight_ns"):
        parse_ftf(document)


@pytest.mark.parametrize("resonator", ["r1", "r2"])
@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), "0.2"])
def test_mismatch_invalid_values_report_resonator_field_path(
    resonator: str, value: object
) -> None:
    document = ftf_document()
    path = f"readout.resonators.{resonator}.mismatch_phase_rad"
    _set_field(document, path, -0.4)
    parse_ftf(document)
    _set_field(document, path, value)
    with pytest.raises(ValueError, match=path.replace(".", r"\.")):
        parse_ftf(document)


@pytest.mark.parametrize("channels", [None, (4, 1), (1,), tuple(range(1, 9))])
def test_dual_readout_yaml_loads_with_optional_mux(
    tmp_path: Path, channels: tuple[int, ...] | None
) -> None:
    document = ftf_document() if channels is None else dual_readout_document(channels)
    path = tmp_path / "readout.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    try:
        settings = load_ftf(path)
    except ValueError as error:
        pytest.fail(f"pulse with optional independent mux must load: {error}")
    assert settings.readout.adc.ro_ch == 0
    mux = settings.readout.mux
    if channels is None:
        assert mux is None, "omitting mux must disable only the mux path"
    else:
        assert mux is not None, "mux must retain its binding"
        assert mux.gen_ch == 1
        assert mux.input_per_gain == -2.0
        assert tuple(receiver.ro_ch for receiver in mux.adc) == channels
        assert parse_ftf(settings.model_dump(mode="json")) == settings


@pytest.mark.parametrize("kind", ["pulse", "mux", None])
def test_removed_readout_kind_reports_migration_error(
    tmp_path: Path, kind: object
) -> None:
    document = ftf_document()
    _set_field(document, "readout.drive.kind", kind)
    path = tmp_path / "old.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match=r"readout\.drive\.kind:.*removed"):
        load_ftf(path)


@pytest.mark.parametrize(
    "path,value,error_path",
    [
        ("readout.mux.gen_ch", 0, "readout.mux.gen_ch"),
        ("readout.mux.gen_ch", 2, "drive_lines.mw.gen_ch"),
        ("drive_lines.mw.gen_ch", 0, "drive_lines.mw.gen_ch"),
        ("readout.mux.adc", [], "readout.mux.adc"),
        (
            "readout.mux.adc",
            [
                {
                    "ro_ch": ch,
                    "gain": 1.0,
                    "noise_psd_adc2_ns": [[0.0, 0.0], [0.0, 0.0]],
                }
                for ch in range(1, 10)
            ],
            "readout.mux.adc",
        ),
        (
            "readout.mux.adc",
            [{"ro_ch": 1, "gain": 1.0, "noise_psd_adc2_ns": [[0.0, 0.0], [0.0, 0.0]]}]
            * 2,
            "readout.mux.adc",
        ),
        (
            "readout.mux.adc",
            [{"ro_ch": 0, "gain": 1.0, "noise_psd_adc2_ns": [[0.0, 0.0], [0.0, 0.0]]}],
            "readout.mux.adc",
        ),
        ("readout.adc", [], "readout.adc"),
        ("readout.mux.input_per_gain", float("inf"), "readout.mux.input_per_gain"),
        (
            "readout.mux.adc",
            [{"ro_ch": 1, "gain": 1.0, "noise_psd_adc2_ns": [[1.0, 2.0], [2.0, 1.0]]}],
            "readout.mux.adc.0.noise_psd_adc2_ns",
        ),
    ],
)
def test_dual_readout_invalid_bindings_report_field_path(
    path: str, value: object, error_path: str
) -> None:
    document = dual_readout_document()
    try:
        parse_ftf(document)
    except ValueError as error:
        pytest.fail(f"valid dual-readout must parse before invalid binding: {error}")
    _set_field(document, path, value)
    with pytest.raises(ValueError, match=error_path.replace(".", r"\.")):
        parse_ftf(document)


@pytest.mark.parametrize("value", [-1, 1.5, True, "1"])
@pytest.mark.parametrize(
    "path",
    [
        "readout.drive.gen_ch",
        "readout.mux.gen_ch",
        "drive_lines.mw.gen_ch",
        "readout.adc.ro_ch",
        "readout.mux.adc",
    ],
)
def test_dual_readout_channel_indices_require_nonnegative_integers(
    path: str, value: object
) -> None:
    document = dual_readout_document()
    try:
        parse_ftf(document)
    except ValueError as error:
        pytest.fail(f"valid dual-readout must parse before invalid index: {error}")
    error_path = path
    if path.endswith(".adc"):
        value = [
            {"ro_ch": value, "gain": 1.0, "noise_psd_adc2_ns": [[0.0, 0.0], [0.0, 0.0]]}
        ]
        error_path += ".0.ro_ch"
    _set_field(document, path, value)
    with pytest.raises(ValueError, match=error_path.replace(".", r"\.")):
        parse_ftf(document)


def test_dual_readout_channel_indices_have_no_fixed_upper_bound() -> None:
    document = dual_readout_document((100, 200))
    for path, value in [
        ("readout.drive.gen_ch", 100),
        ("readout.mux.gen_ch", 200),
        ("drive_lines.mw.gen_ch", 300),
        ("readout.adc.ro_ch", 300),
    ]:
        _set_field(document, path, value)
    try:
        settings = parse_ftf(document)
    except ValueError as error:
        pytest.fail(f"legal high indices must not be hardware-limited: {error}")
    assert settings.readout.drive.gen_ch == 100
    assert settings.drive_lines["mw"].gen_ch == 300
    assert settings.readout.adc.ro_ch == 300
    assert settings.readout.mux is not None
    assert settings.readout.mux.gen_ch == 200
    assert tuple(r.ro_ch for r in settings.readout.mux.adc) == (100, 200)
