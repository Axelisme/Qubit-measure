"""Named FTF input owner rejects invalid wiring/units with field paths."""

from pathlib import Path

import pytest
import yaml

from tests.resources._ftf_data import ftf_document


def test_ftf_loads_detached_named_settings(tmp_path: Path) -> None:
    from zcu_tools.resources.ftf import load_ftf

    path = tmp_path / "model.yaml"
    path.write_text(yaml.safe_dump(ftf_document()), encoding="utf-8")
    config = load_ftf(path)
    assert config.model.g12_ghz == -.02
    assert config.devices[0].unit == "V"
    assert config.flux_sources[0].delay_ns == 0
    assert config.flux_sources[0].tau_ns == 0
    assert config.receiver.noise_psd_adc2_ns == ((0., 0.), (0., 0.))
    assert config.model.coupler.ng == 0


@pytest.mark.parametrize("patch,path", [
    ({"mystery": 1}, "mystery"),
    ({"channels": {"generators": 3, "readouts": 1, "clock": 10}}, "channels.clock"),
    ({"devices": [{"name": "bias", "unit": "mA", "limits": [-1, 1], "initial": 0, "rampstep": .1}]}, "devices.0.unit"),
    ({"devices": [{"name": "bias", "unit": "V", "limits": [-1, 1], "initial": 2, "rampstep": .1}]}, "devices.0.initial"),
    ({"dc_mapping": {"offset": [0, 0, 0], "columns": [{"source": "missing", "unit": "Phi0/V", "coefficients": [0, 1, 0]}]}}, "dc_mapping.columns.0.source"),
    ({"dc_mapping": {"offset": [0, 0, 0], "columns": [{"source": "bias", "unit": "Phi0/A", "coefficients": [0, 1, 0]}]}}, "dc_mapping.columns.0.unit"),
    ({"flux_sources": [{"name": "flux", "gen_ch": 0, "coefficients_phi0_per_gain": [0, 1, 0]}]}, "flux_sources.0.gen_ch"),
    ({"flux_sources": [{"name": "flux", "gen_ch": 1, "coefficients_phi0_per_gain": [0, 1]}]}, "flux_sources.0.coefficients_phi0_per_gain"),
    ({"receiver": {"ro_ch": 0, "gain": 1, "noise_psd_adc2_ns": [[1, 2], [2, 1]]}}, "receiver.noise_psd_adc2_ns"),
    ({"solver": {"rtol": 0, "atol": 1e-8, "max_step_ns": 1, "bath_segment_ns": 4}}, "solver.rtol"),
])
def test_ftf_field_path_diagnostics(patch: dict[str, object], path: str) -> None:
    from zcu_tools.resources.ftf import parse_ftf

    with pytest.raises(ValueError, match=path.replace(".", r"\.")):
        parse_ftf(ftf_document() | patch)
