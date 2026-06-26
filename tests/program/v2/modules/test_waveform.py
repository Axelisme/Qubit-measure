from unittest.mock import MagicMock

import numpy as np
import pytest
from qick.asm_v2 import QickParam
from zcu_tools.program.v2.modules import (
    WaveformCfgFactory,  # ensures leaf subclass registration
)
from zcu_tools.program.v2.modules.waveform import (
    AbsWaveformCfg,
    ArbWaveform,
    ArbWaveformCfg,
    ConstWaveform,
    ConstWaveformCfg,
    CosineWaveform,
    CosineWaveformCfg,
    DragWaveform,
    DragWaveformCfg,
    FlatTopWaveform,
    FlatTopWaveformCfg,
    GaussWaveform,
    GaussWaveformCfg,
    resolve_waveform_ref,
)


def test_const_waveform_length_set_param():
    cfg = ConstWaveformCfg(length=1.0)
    cfg.set_param("length", 2.5)
    assert cfg.length == 2.5


def test_cosine_waveform_rejects_qickparam_length():
    cfg = CosineWaveformCfg(length=1.0)
    with pytest.raises(ValueError):
        cfg.set_param("length", QickParam(start=1.0, spans={"a": 0.1}))


def test_abs_waveform_cfg_default_methods_raise():
    cfg = AbsWaveformCfg(style="const")

    with pytest.raises(NotImplementedError):
        cfg.build("base")

    with pytest.raises(NotImplementedError):
        cfg.set_param("length", 2.0)


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
    with pytest.raises(ValueError):
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
    with pytest.raises(ValueError):
        cfg.set_param("length", 2.0)


def test_arb_waveform_cfg_rejects_legacy_length_field():
    with pytest.raises(ValueError):
        ArbWaveformCfg.model_validate({"style": "arb", "length": 1.0, "data": "dummy"})


def test_arb_waveform_cfg_length_comes_from_data_duration(monkeypatch):
    class Info:
        duration = 2.5

    monkeypatch.setattr(
        "zcu_tools.meta_tool.arb_waveform.ArbWaveformDatabase.inspect",
        lambda key: Info(),
    )

    cfg = ArbWaveformCfg(data="dummy")

    assert cfg.length == 2.5


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


def test_waveform_build_dispatch_drag_arb_and_flat_top():
    drag = DragWaveformCfg(length=1.0, sigma=0.2, delta=0.1, alpha=0.0).build("d")
    arb = ArbWaveformCfg(data="demo").build("a")
    flat_top = FlatTopWaveformCfg(
        length=1.0,
        raise_waveform=GaussWaveformCfg(length=0.2, sigma=0.05),
    ).build("f")

    assert isinstance(drag, DragWaveform)
    assert isinstance(arb, ArbWaveform)
    assert isinstance(flat_top, FlatTopWaveform)
    assert isinstance(flat_top.raise_waveform, GaussWaveform)


def test_waveform_adapter_dispatches_by_style():
    cfg = WaveformCfgFactory.from_raw({"style": "gauss", "length": 1.0, "sigma": 0.2})
    assert isinstance(cfg, GaussWaveformCfg)


def test_waveform_adapter_rejects_unknown_style():
    with pytest.raises(Exception):
        WaveformCfgFactory.from_raw({"style": "no_such_style", "length": 1.0})


def test_waveform_factory_string_ref_requires_module_library():
    with pytest.raises(ValueError, match="ModuleLibrary is not provided"):
        WaveformCfgFactory.from_raw("stored_waveform")


def test_waveform_factory_string_ref_uses_module_library():
    ml = MagicMock()
    ml.get_waveform.return_value = ConstWaveformCfg(length=0.5)

    cfg = WaveformCfgFactory.from_raw("stored_waveform", ml=ml)

    assert isinstance(cfg, ConstWaveformCfg)
    ml.get_waveform.assert_called_once_with("stored_waveform")


def test_waveform_type_adapter_rejects_string_ref_without_context():
    info = MagicMock()
    info.context = None

    with pytest.raises(ValueError, match="ModuleLibrary context not found"):
        resolve_waveform_ref("named_waveform", info)


def test_waveform_type_adapter_resolves_nested_raise_waveform_ref():
    ml = MagicMock()
    ml.get_waveform.return_value = GaussWaveformCfg(length=0.2, sigma=0.05)

    cfg = WaveformCfgFactory.from_raw(
        {"style": "flat_top", "length": 1.0, "raise_waveform": "gauss_rise"},
        ml=ml,
    )

    assert isinstance(cfg, FlatTopWaveformCfg)
    assert isinstance(cfg.raise_waveform, GaussWaveformCfg)
    ml.get_waveform.assert_called_once_with("gauss_rise")


def test_waveform_to_wav_kwargs_and_create_methods():
    prog = MagicMock()

    cosine = CosineWaveformCfg(length=1.0).build("cos")
    gauss = GaussWaveformCfg(length=1.2, sigma=0.3).build("gau")
    drag = DragWaveformCfg(length=1.5, sigma=0.4, delta=0.2, alpha=0.1).build("drg")
    const = ConstWaveformCfg(length=2.0).build("cst")

    assert const.to_wav_kwargs() == {"style": "const", "length": 2.0}
    assert cosine.to_wav_kwargs() == {"style": "arb", "envelope": "cos"}
    assert gauss.to_wav_kwargs() == {"style": "arb", "envelope": "gau"}
    assert drag.to_wav_kwargs() == {"style": "arb", "envelope": "drg"}

    cosine.create(prog, 1, phrase="kept")
    gauss.create(prog, 2, marker="g")
    drag.create(prog, 3, marker="d")
    const.create(prog, 4)

    prog.add_cosine.assert_called_once_with(1, "cos", length=1.0, phrase="kept")
    prog.add_gauss.assert_called_once_with(2, "gau", sigma=0.3, length=1.2, marker="g")
    prog.add_DRAG.assert_called_once_with(
        3,
        "drg",
        sigma=0.4,
        length=1.5,
        delta=0.2,
        alpha=0.1,
        marker="d",
    )


def test_arb_waveform_make_iqdata_and_create(monkeypatch):
    class FakeSoccfg(dict):
        def get_maxv(self, ch):
            assert ch == 0
            return 8.0

    prog = MagicMock()
    prog.soccfg = FakeSoccfg({"gens": [{"samps_per_clk": 2}]})
    prog.us2cycles.side_effect = lambda *, gen_ch, us: 2 if us == 2.0 else 3

    cfg = ArbWaveformCfg(data="demo")
    waveform = cfg.build("arb")

    idata_raw = np.array([0.0, 1.0, 0.5])
    qdata_raw = np.array([0.0, -1.0, -0.5])
    time_raw = np.array([0.0, 1.0, 2.0])

    monkeypatch.setattr(
        "zcu_tools.meta_tool.arb_waveform.ArbWaveformDatabase.get",
        lambda key: (idata_raw, qdata_raw, time_raw),
    )

    idata, qdata = waveform.make_iqdata(0, prog)
    assert len(idata) == 4
    assert qdata is not None
    assert len(qdata) == 4
    assert np.allclose(idata, np.array([0.0, 4.0, 8.0, 6.0]))
    assert np.allclose(qdata, np.array([0.0, -4.0, -8.0, -6.0]))

    even_idata, even_qdata = waveform.make_iqdata(0, prog, even_length=True)
    assert len(even_idata) == 12
    assert even_qdata is not None
    assert len(even_qdata) == 12

    waveform.create(prog, 0)
    prog.add_envelope.assert_called_once()
    _, kwargs = prog.add_envelope.call_args
    assert kwargs["idata"].shape == (4,)
    assert kwargs["qdata"].shape == (4,)
    assert waveform.to_wav_kwargs() == {"style": "arb", "envelope": "arb"}


def test_arb_waveform_make_iqdata_handles_missing_q_channel(monkeypatch):
    class FakeSoccfg(dict):
        def get_maxv(self, ch):
            return 4.0

    prog = MagicMock()
    prog.soccfg = FakeSoccfg({"gens": [{"samps_per_clk": 1}]})
    prog.us2cycles.return_value = 3

    waveform = ArbWaveformCfg(data="demo").build("arb")

    monkeypatch.setattr(
        "zcu_tools.meta_tool.arb_waveform.ArbWaveformDatabase.get",
        lambda key: (
            np.array([0.0, 1.0, 0.0]),
            None,
            np.array([0.0, 1.0, 2.0]),
        ),
    )

    idata, qdata = waveform.make_iqdata(0, prog)

    assert np.allclose(idata, np.array([0.0, 8.0 / 3.0, 8.0 / 3.0]))
    assert qdata is None


def test_arb_waveform_make_iqdata_uses_full_data_duration(monkeypatch):
    class FakeSoccfg(dict):
        def get_maxv(self, ch):
            return 1.0

    prog = MagicMock()
    prog.soccfg = FakeSoccfg({"gens": [{"samps_per_clk": 1}]})
    prog.us2cycles.side_effect = lambda *, gen_ch, us: 2 if us == 2.0 else 99

    waveform = ArbWaveformCfg(data="long_data").build("arb")

    monkeypatch.setattr(
        "zcu_tools.meta_tool.arb_waveform.ArbWaveformDatabase.get",
        lambda key: (
            np.array([0.0, 1.0, 0.0]),
            None,
            np.array([0.0, 1.0, 2.0]),
        ),
    )

    idata, qdata = waveform.make_iqdata(0, prog)

    assert qdata is None
    assert np.allclose(idata, np.array([0.0, 1.0]))
    prog.us2cycles.assert_called_once_with(gen_ch=0, us=2.0)


def test_flat_top_waveform_create_and_kwargs():
    prog = MagicMock()
    waveform = FlatTopWaveformCfg(
        length=1.0,
        raise_waveform=GaussWaveformCfg(length=0.2, sigma=0.05),
    ).build("ft")

    waveform.create(prog, 1, marker="flat")
    assert waveform.to_wav_kwargs() == {
        "style": "flat_top",
        "envelope": "ft",
        "length": 0.8,
    }
    prog.add_gauss.assert_called_once_with(
        1,
        "ft",
        sigma=0.05,
        length=0.2,
        even_length=True,
        marker="flat",
    )


def test_flat_top_waveform_set_param_rejects_unknown_name():
    cfg = FlatTopWaveformCfg(
        length=1.0,
        raise_waveform=GaussWaveformCfg(length=0.2, sigma=0.05),
    )
    cfg.set_param("length", 2.0)
    assert cfg.length == 2.0

    with pytest.raises(ValueError, match="Unknown parameter"):
        cfg.set_param("sigma", 0.1)
