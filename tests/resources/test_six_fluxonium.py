"""External six-cavity documents fail before allocating simulation resources."""

from pathlib import Path

import pytest
import yaml
from zcu_tools.resources import six_fluxonium as api

from ._six_fluxonium_data import six_document


def parse(document):
    try:
        return api.parse_six_fluxonium(document)
    except (NotImplementedError, ValueError) as error:
        pytest.fail(
            f"six-fluxonium parser must accept independent local and coil material: {error}"
        )


def test_parse_and_load_preserve_complete_independent_material(tmp_path: Path):
    document = six_document()
    parsed = parse(document)
    path = tmp_path / "material.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    loaded = api.load_six_fluxonium(path)
    assert parsed.model_dump(mode="json", exclude={"numerics"}) == document
    assert loaded == parsed
    document["qubits"]["Q1"]["chip"]["qubit"]["ej_ghz"] = 99
    assert loaded.model_dump()["qubits"]["Q1"]["chip"]["qubit"]["ej_ghz"] == 4


def test_per_qubit_drift_accepts_positive_losses_and_rejects_disabled_targets():
    document = six_document()
    document["qubits"]["Q2"]["chip"]["qubit"]["xqp"] = 1e-7
    document["qubits"]["Q2"] = {
        **document["qubits"]["Q2"],
        "drift": {"xqp": {"sigma": 0.3, "tau_s": 10.0}},
    }
    parsed = api.parse_six_fluxonium(document)
    assert parsed.qubits.Q2.drift is not None
    assert parsed.qubits.Q2.drift.xqp is not None
    assert parsed.qubits.Q2.drift.xqp.tau_s == 10.0
    document["qubits"]["Q2"]["chip"]["qubit"]["xqp"] = 0.0
    with pytest.raises(ValueError, match=r"qubits.Q2.*drift.xqp.*positive"):
        api.parse_six_fluxonium(document)


@pytest.mark.parametrize(
    "field",
    [
        "routing_cache_bytes",
    ],
)
@pytest.mark.parametrize("value", [-1, True, 1.5, "1024"])
def test_cache_budgets_reject_noninteger_or_negative_bytes(field, value):
    document: dict[str, object] = {**six_document(), "numerics": {field: value}}
    with pytest.raises(ValueError, match=rf"numerics\.{field}"):
        api.parse_six_fluxonium(document)


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
        ("local_source", "qubits.Q2.dc_bias.source.name"),
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
        if defect == "local_source":
            q["dc_bias"]["source"]["name"] = "Q1_bias"
        elif defect == "drive":
            q["drive_line"]["gen_ch"] = 12
        elif defect == "adc":
            q["readout"]["adc"]["ro_ch"] = 0
        else:
            q["readout"]["resonator"]["frequency_ghz"] = float("nan")
    with pytest.raises(ValueError, match=diagnostic):
        api.parse_six_fluxonium(document)


@pytest.mark.parametrize("field", ["offset", "period", "coil_offset", "coil_period"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "0.1"])
def test_current_calibrations_require_finite_numeric_values(field, value):
    document = six_document()
    document["qubits"]["Q3"]["dc_bias"][field] = value
    reason = "finite number" if isinstance(value, float) else "valid number"
    with pytest.raises(ValueError, match=rf"qubits.Q3.dc_bias.{field}: [^;]*{reason}"):
        api.parse_six_fluxonium(document)


@pytest.mark.parametrize("field", ["period", "coil_period"])
def test_current_calibrations_reject_zero_period(field):
    document = six_document()
    document["qubits"]["Q3"]["dc_bias"][field] = 0.0
    with pytest.raises(ValueError, match=rf"qubits.Q3.dc_bias.{field}: [^;]*nonzero"):
        api.parse_six_fluxonium(document)
