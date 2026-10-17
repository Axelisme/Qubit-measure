"""V2 domain records through the public YAML validation seam."""

from pathlib import Path

import pytest
import yaml
from zcu_tools.resources.ftf import load_ftf, parse_ftf

from tests.resources._ftf_data import ftf_v2_document


def test_v2_loads_named_domain_values_and_detaches_input(tmp_path: Path) -> None:
    document = ftf_v2_document()
    path = tmp_path / "sample.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    try:
        settings = load_ftf(path)
    except ValueError as error:
        pytest.fail(f"v2 domain document must load: {error}")
    document.clear()
    assert settings.chip.g_f1_f2_ghz == -0.02
    assert settings.dc_bias.sources["bias"].unit == "V"
    assert settings.dc_bias.sources["bias"].flux_per_unit.coupler == 0.1
    assert settings.dc_bias.offset_phi0.f2 == 0.4
    assert settings.drive_lines["mw"].charge_drive_ghz_per_gain.f2.imag == 0.02
    assert settings.readout.resonators.r2.frequency_ghz == 6.2
    assert settings.numerics.truncation.f1.native_cutoff == 8
    assert settings.numerics.truncation.retained_levels == 8
    assert settings.readout.adc.offset_adc == (0.0, 0.0)


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
    document = ftf_v2_document()
    try:
        parse_ftf(document)
    except ValueError as error:
        pytest.fail(f"v2 base must validate before checking {field}: {error}")
    with pytest.raises(ValueError, match=field):
        parse_ftf(document | {field: value})


def test_version_one_is_rejected_at_version_field() -> None:
    with pytest.raises(ValueError, match=r"version:.*2"):
        parse_ftf(ftf_v2_document() | {"version": 1})


@pytest.mark.parametrize(
    "section,path",
    [
        ("dc_bias", "offset_phi0.f2"),
        ("drive_lines", "mw.charge_drive_ghz_per_gain.f2"),
        ("readout", "resonators.r1.charge_coefficients_ghz.f2"),
    ],
)
def test_v2_requires_every_named_loop(section: str, path: str) -> None:
    document = ftf_v2_document()
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
