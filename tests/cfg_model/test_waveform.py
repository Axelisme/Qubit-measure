"""Core waveform knobs and non-serialized asset-binding contracts."""

from copy import deepcopy

import pytest
from pydantic import ValidationError
from qick.asm_v2 import QickParam
from zcu_tools.cfg_model import (
    ArbWaveformCfg,
    CfgValidationContext,
    ConstWaveformCfg,
    CosineWaveformCfg,
    DragWaveformCfg,
    FlatTopWaveformCfg,
    GaussWaveformCfg,
    WaveformCfgFactory,
)

from ._support import AssetInfo


def test_deepcopy_preserves_current_asset_binding():
    assets = AssetInfo()
    cfg = WaveformCfgFactory.from_raw(
        {"style": "arb", "data": "shape"},
        context=CfgValidationContext(assets=assets),
    )
    copied = deepcopy(cfg)

    assets.current_duration = 4.0

    assert cfg.length == 4.0
    assert copied.length == 4.0
    assert copied.to_dict() == {"style": "arb", "data": "shape"}


def test_const_waveform_length_set_param():
    cfg = ConstWaveformCfg(length=1.0)
    cfg.set_param("length", 2.5)
    assert cfg.length == 2.5


def test_cosine_waveform_rejects_qickparam_length():
    cfg = CosineWaveformCfg(length=1.0)
    with pytest.raises(
        ValueError, match="Cosine waveform length must be a float value"
    ):
        cfg.set_param("length", QickParam(start=1.0, spans={"a": 0.1}))


def test_const_waveform_rejects_unknown_param():
    cfg = ConstWaveformCfg(length=1.0)
    with pytest.raises(ValueError, match="Unknown parameter"):
        cfg.set_param("gain", 1.0)


def test_cosine_waveform_rejects_unknown_param():
    cfg = CosineWaveformCfg(length=1.0)
    with pytest.raises(ValueError, match="Unknown parameter"):
        cfg.set_param("sigma", 0.1)


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
    with pytest.raises(
        ValueError, match="Drag waveform length must not be a QickParam"
    ):
        cfg.set_param("length", QickParam(start=1.0, spans={"a": 0.1}))


def test_gauss_waveform_rejects_qickparam_and_unknown_param():
    cfg = GaussWaveformCfg(length=1.0, sigma=0.2)

    with pytest.raises(ValueError, match="must not be a QickParam"):
        cfg.set_param("sigma", QickParam(start=1.0, spans={"a": 0.1}))

    with pytest.raises(ValueError, match="Unknown parameter"):
        cfg.set_param("delta", 0.1)


def test_drag_waveform_length_rescales_sigma_and_updates_fields():
    cfg = DragWaveformCfg(length=1.0, sigma=0.2, delta=0.1, alpha=0.0)

    cfg.set_param("length", 2.0)
    assert cfg.length == 2.0
    assert cfg.sigma == pytest.approx(0.4)

    cfg.set_param("sigma", 0.6)
    cfg.set_param("delta", 0.3)
    cfg.set_param("alpha", 0.5)
    cfg.set_param("only_length", 1.5)

    assert cfg.length == 1.5
    assert cfg.sigma == 0.6
    assert cfg.delta == 0.3
    assert cfg.alpha == 0.5


def test_drag_waveform_rejects_unknown_param():
    cfg = DragWaveformCfg(length=1.0, sigma=0.2, delta=0.1, alpha=0.0)
    with pytest.raises(ValueError, match="Unknown parameter"):
        cfg.set_param("phase", 90.0)


def test_arb_waveform_set_param_rejected():
    cfg = ArbWaveformCfg(data="dummy")
    with pytest.raises(ValueError, match="Arb waveform length is asset-derived"):
        cfg.set_param("length", 2.0)


def test_arb_waveform_cfg_rejects_legacy_length_field():
    with pytest.raises(ValidationError, match="length"):
        ArbWaveformCfg.model_validate({"style": "arb", "length": 1.0, "data": "dummy"})


def test_flat_top_waveform_set_param_rejects_unknown_name():
    cfg = FlatTopWaveformCfg(
        length=1.0,
        raise_waveform=GaussWaveformCfg(length=0.2, sigma=0.05),
    )
    cfg.set_param("length", 2.0)
    assert cfg.length == 2.0

    with pytest.raises(ValueError, match="Unknown parameter"):
        cfg.set_param("sigma", 0.1)
