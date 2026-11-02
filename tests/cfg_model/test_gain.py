"""DAC full-scale gain validation through public module models."""

import pytest
from pydantic import ValidationError
from zcu_tools.cfg_model import (
    MuxReadoutTone,
    PulseCfg,
    PulseReadoutCfg,
    PulseReadoutSpec,
    PulseSpec,
)


@pytest.fixture(
    params=[PulseCfg, PulseSpec, PulseReadoutCfg, PulseReadoutSpec, MuxReadoutTone]
)
def gain_model(request):
    model = request.param
    if model is MuxReadoutTone:
        raw = {"freq": 100.0, "ro_ch": 0}
    else:
        raw = {"waveform": {"style": "const", "length": 0.2}, "nqz": 1, "freq": 100.0}
        if model in (PulseCfg, PulseSpec):
            raw["ch"] = 0
        else:
            raw.update(gen_ch=0, ro_ch=0, ro_length=0.2)
    return model, raw


def test_module_gain_rejects_out_of_range_and_accepts_boundaries(gain_model):
    model, raw = gain_model
    for gain in (-1.0, 0.0, 1.0):
        cfg = model.model_validate({**raw, "gain": gain})
        assert cfg.gain == gain
    for gain in (-1.1, 1.1):
        with pytest.raises(ValidationError) as exc:
            model.model_validate({**raw, "gain": gain})
        assert exc.value.errors()[0]["loc"][0] == "gain"


@pytest.mark.parametrize("model", [PulseCfg, PulseReadoutCfg])
def test_runtime_gain_accepts_table_values_without_range_checks(model):
    from zcu_tools.program.v2 import TableParam

    raw = {"waveform": {"style": "const", "length": 0.2}, "nqz": 1, "freq": 100.0}
    if model is PulseCfg:
        raw["ch"] = 0
    else:
        raw.update(gen_ch=0, ro_ch=0, ro_length=0.2)
    gain = TableParam("point", [0.0, 1.1, -1.2])
    cfg = model.model_validate({**raw, "gain": gain})
    assert cfg.gain is gain
    replacement = TableParam("point", [1.4, -1.5, 0.0])
    cfg.set_param("gain", replacement)
    assert cfg.gain is replacement
