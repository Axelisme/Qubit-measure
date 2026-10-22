"""External six-cavity documents fail before allocating simulation resources."""

from importlib import import_module
from pathlib import Path

import pytest
import yaml

from ._six_fluxonium_data import six_document


def resource_api():
    try:
        return import_module("zcu_tools.resources.six_fluxonium")
    except ModuleNotFoundError:
        pytest.fail("six-fluxonium documents need a public parser and loader")


def test_parse_and_load_preserve_complete_independent_material(tmp_path: Path):
    api = resource_api()
    document = six_document()
    parsed = api.parse_six_fluxonium(document)
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
        ("flux", "flux_line.gen_ch"),
        ("adc", "readout.adc.ro_ch"),
        ("source", "dc_bias.source.name"),
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
        field = {"mux": "mux_gen_ch", "pulse": "pulse_gen_ch", "pulse_adc": "pulse_ro_ch"}[defect]
        document[field] = 99
    else:
        q = document["qubits"]["Q2"]
        if defect == "drive":
            q["drive_line"]["gen_ch"] = 12
        elif defect == "flux":
            q["flux_line"]["gen_ch"] = 11
        elif defect == "adc":
            q["readout"]["adc"]["ro_ch"] = 0
        elif defect == "source":
            q["dc_bias"]["source"]["name"] = "flux_bias_1"
        else:
            q["readout"]["resonator"]["frequency_ghz"] = float("nan")
    with pytest.raises(ValueError, match=diagnostic):
        resource_api().parse_six_fluxonium(document)
