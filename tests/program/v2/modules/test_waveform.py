import pytest
from qick.asm_v2 import QickParam
from zcu_tools.program.v2.modules.waveform import (
    ArbWaveformCfg,
    ConstWaveform,
    ConstWaveformCfg,
    CosineWaveform,
    CosineWaveformCfg,
    DragWaveformCfg,
    GaussWaveform,
    GaussWaveformCfg,
    WaveformCfg,
)


def test_const_waveform_length_set_param():
    cfg = ConstWaveformCfg(length=1.0)
    cfg.set_param("length", 2.5)
    assert cfg.length == 2.5


def test_cosine_waveform_rejects_qickparam_length():
    cfg = CosineWaveformCfg(length=1.0)
    with pytest.raises(ValueError):
        cfg.set_param("length", QickParam(start=1.0, spans={"a": 0.1}))


def test_gauss_waveform_length_rescales_sigma():
    cfg = GaussWaveformCfg(length=1.0, sigma=0.2)
    cfg.set_param("length", 2.0)
    assert cfg.length == 2.0
    assert cfg.sigma == pytest.approx(0.4)


def test_gauss_waveform_only_length_preserves_sigma():
    cfg = GaussWaveformCfg(length=1.0, sigma=0.2)
    cfg.set_param("only_length", 3.0)
    assert cfg.length == 3.0
    assert cfg.sigma == 0.2


def test_drag_waveform_rejects_qickparam():
    cfg = DragWaveformCfg(length=1.0, sigma=0.2, delta=0.1, alpha=0.0)
    with pytest.raises(ValueError):
        cfg.set_param("length", QickParam(start=1.0, spans={"a": 0.1}))


def test_arb_waveform_set_param_rejected():
    cfg = ArbWaveformCfg(length=1.0, data="dummy")
    with pytest.raises(ValueError):
        cfg.set_param("length", 2.0)


def test_waveform_build_dispatch_const():
    wf = ConstWaveformCfg(length=1.0).build("n")
    assert isinstance(wf, ConstWaveform)
    assert wf.length == 1.0
    assert wf.name == "n"


def test_waveform_build_dispatch_gauss():
    wf = GaussWaveformCfg(length=1.0, sigma=0.2).build("g")
    assert isinstance(wf, GaussWaveform)


def test_waveform_build_dispatch_cosine():
    wf = CosineWaveformCfg(length=1.0).build("c")
    assert isinstance(wf, CosineWaveform)


def test_waveform_adapter_dispatches_by_style():
    cfg = WaveformCfg.from_raw({"style": "gauss", "length": 1.0, "sigma": 0.2})
    assert isinstance(cfg, GaussWaveformCfg)


def test_waveform_adapter_rejects_unknown_style():
    with pytest.raises(Exception):
        WaveformCfg.from_raw({"style": "no_such_style", "length": 1.0})
