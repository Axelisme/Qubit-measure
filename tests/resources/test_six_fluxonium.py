"""External six-cavity documents fail before allocating simulation resources."""

from pathlib import Path

import pytest
import yaml
from zcu_tools.resources import six_fluxonium as api

from ._six_fluxonium_data import six_document


def parse(document):
    try:
        return api.parse_six_fluxonium(document)
    except NotImplementedError:
        pytest.fail("six-fluxonium parser must validate complete independent material")


def test_parse_and_load_preserve_complete_independent_material(tmp_path: Path):
    document = six_document()
    parsed = parse(document)
    path = tmp_path / "material.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    loaded = api.load_six_fluxonium(path)
    assert parsed.model_dump(mode="json") == document
    assert loaded == parsed
    document["qubits"]["Q1"]["chip"]["qubit"]["ej_ghz"] = 99
    assert loaded.model_dump()["qubits"]["Q1"]["chip"]["qubit"]["ej_ghz"] == 4


@pytest.mark.parametrize(
    "defect, diagnostic",
    [
        ("missing", "Q6"),
        ("extra", "Q7"),
        ("unknown", "surprise"),
        ("version", "version"),
        ("drive", "drive_line.gen_ch"),
        ("adc", "readout.adc.ro_ch"),
        ("source", "dc_source.name"),
        ("mux", "mux_gen_ch"),
        ("pulse", "pulse_gen_ch"),
        ("pulse_adc", "pulse_ro_ch"),
        ("nan", "frequency_ghz"),
    ],
)
def test_invalid_topology_or_material_is_rejected(defect, diagnostic):
    document = six_document()
    if defect == "missing":
        del document["qubits"]["Q6"]
    elif defect == "extra":
        document["qubits"]["Q7"] = document["qubits"]["Q1"]
    elif defect == "unknown":
        document["surprise"] = True
    elif defect == "version":
        document["version"] = True
    elif defect in ("mux", "pulse", "pulse_adc"):
        field = {
            "mux": "mux_gen_ch",
            "pulse": "pulse_gen_ch",
            "pulse_adc": "pulse_ro_ch",
        }[defect]
        document[field] = 99
    elif defect == "source":
        document["dc_source"]["name"] = "other"
    else:
        q = document["qubits"]["Q2"]
        if defect == "drive":
            q["drive_line"]["gen_ch"] = 12
        elif defect == "adc":
            q["readout"]["adc"]["ro_ch"] = 0
        else:
            q["readout"]["resonator"]["frequency_ghz"] = float("nan")
    with pytest.raises(ValueError, match=diagnostic):
        parse(document)
