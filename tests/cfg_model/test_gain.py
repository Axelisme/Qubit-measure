"""DAC full-scale gain validation through public module models."""

import pytest
from pydantic import ValidationError
from qick.asm_v2 import QickParam
from zcu_tools.cfg_model import (
    ConstWaveformCfg,
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
        with pytest.raises(
            ValidationError, match=r"Gain must be within \[-1, 1\].*" + str(gain)
        ) as exc:
            model.model_validate({**raw, "gain": gain})
        assert exc.value.errors()[0]["loc"][0] == "gain"


@pytest.mark.parametrize("model", [PulseCfg, PulseReadoutCfg])
def test_runtime_gain_checks_entire_multidimensional_sweep_and_assignments(model):
    raw = {"waveform": ConstWaveformCfg(length=0.2), "nqz": 1, "freq": 100.0}
    if model is PulseCfg:
        raw["ch"] = 0
    else:
        raw.update(gen_ch=0, ro_ch=0, ro_length=0.2)
    valid = QickParam(-1.0, {"a": 1.0, "b": 1.0})
    cfg = model.model_validate({**raw, "gain": valid})
    assert cfg.gain is valid
    for gain in (
        1.1,
        QickParam(0.0, {"a": 0.6, "b": 0.6}),
        QickParam(0.0, {"a": -0.6, "b": -0.6}),
    ):
        with pytest.raises(ValidationError, match=r"Gain must be within \[-1, 1\]"):
            cfg.set_param("gain", gain)
        assert cfg.gain is valid
    with pytest.raises(ValidationError, match=r"Gain must be within \[-1, 1\].*1.2"):
        model.model_validate({**raw, "gain": QickParam(1.2)})
