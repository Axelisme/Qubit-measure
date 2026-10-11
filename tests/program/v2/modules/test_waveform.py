from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from pydantic import ValidationError
from zcu_tools.cfg_model import ArbWaveformCfg as CoreArbWaveformCfg
from zcu_tools.cfg_model import CfgValidationContext
from zcu_tools.cfg_model import FlatTopWaveformCfg as CoreFlatTopWaveformCfg
from zcu_tools.core.waveform_format import WaveformFormatError
from zcu_tools.program.v2.modules.waveform import (
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
    WaveformCfgFactory,
)
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase, ArbWaveformError


@pytest.fixture(scope="module", autouse=True)
def repository_state_guard() -> Iterator[None]:
    before = vars(ArbWaveformDatabase)["_database_path"]
    yield
    assert vars(ArbWaveformDatabase)["_database_path"] == before


def _asset(
    tmp_path: Path,
    *,
    idata: object = (0.0, 1.0, 0.0),
    qdata: object = (0.0, 0.0, 0.0),
    recipe_json: str | None = None,
) -> str:
    path = tmp_path / "demo.npz"
    if recipe_json is None:
        np.savez(path, idata=np.asarray(idata), qdata=np.asarray(qdata), time=[0, 1, 2])
    else:
        np.savez(
            path,
            idata=np.asarray(idata),
            qdata=np.asarray(qdata),
            time=[0, 1, 2],
            recipe_json=np.array(recipe_json),
        )
    return path.resolve().as_posix()


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


def test_waveform_build_dispatch_drag_arb_and_flat_top(tmp_path: Path):
    drag = DragWaveformCfg(length=1.0, sigma=0.2, delta=0.1, alpha=0.0).build("d")
    arb = ArbWaveformCfg(data="demo", path=_asset(tmp_path)).build("a")
    flat_top = FlatTopWaveformCfg(
        length=1.0,
        raise_waveform=GaussWaveformCfg(length=0.2, sigma=0.05),
    ).build("f")

    assert isinstance(drag, DragWaveform)
    assert isinstance(arb, ArbWaveform)
    assert isinstance(flat_top, FlatTopWaveform)
    assert isinstance(flat_top.raise_waveform, GaussWaveform)


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


def test_arb_waveform_make_iqdata_and_create(tmp_path: Path):
    class FakeSoccfg(dict[str, object]):
        def get_maxv(self, ch):
            assert ch == 0
            return 8.0

    prog = MagicMock()
    prog.soccfg = FakeSoccfg({"gens": [{"samps_per_clk": 2}]})
    prog.us2cycles.side_effect = lambda *, gen_ch, us: 2 if us == 2.0 else 3

    cfg = ArbWaveformCfg(
        data="demo",
        path=_asset(tmp_path, idata=[0.0, 0.6, 0.3], qdata=[0.0, -0.8, -0.4]),
    )
    waveform = cfg.build("arb")

    idata, qdata = waveform.make_iqdata(0, prog)
    assert len(idata) == 4
    assert qdata is not None
    assert len(qdata) == 4
    assert np.allclose(idata, np.array([0.0, 2.4, 4.8, 3.6]))
    assert np.allclose(qdata, np.array([0.0, -3.2, -6.4, -4.8]))

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


def test_arb_waveform_make_iqdata_handles_missing_q_channel(tmp_path: Path):
    class FakeSoccfg(dict[str, object]):
        def get_maxv(self, ch):
            return 4.0

    prog = MagicMock()
    prog.soccfg = FakeSoccfg({"gens": [{"samps_per_clk": 1}]})
    prog.us2cycles.return_value = 3

    waveform = ArbWaveformCfg(data="demo", path=_asset(tmp_path)).build("arb")

    idata, qdata = waveform.make_iqdata(0, prog)

    assert np.allclose(idata, np.array([0.0, 8.0 / 3.0, 8.0 / 3.0]))
    assert qdata is None


def test_arb_waveform_make_iqdata_uses_full_data_duration(tmp_path: Path):
    class FakeSoccfg(dict[str, object]):
        def get_maxv(self, ch):
            return 1.0

    prog = MagicMock()
    prog.soccfg = FakeSoccfg({"gens": [{"samps_per_clk": 1}]})
    prog.us2cycles.side_effect = lambda *, gen_ch, us: 2 if us == 2.0 else 99

    waveform = ArbWaveformCfg(data="long_data", path=_asset(tmp_path)).build("arb")

    idata, qdata = waveform.make_iqdata(0, prog)

    assert qdata is None
    assert np.allclose(idata, np.array([0.0, 1.0]))
    prog.us2cycles.assert_called_once_with(gen_ch=0, us=2.0)


def test_arb_path_controls_length_samples_and_flat_top_after_global_db_switch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ArbWaveformDatabase, "_database_path", None)
    path = _asset(tmp_path, idata=[0.0, 0.5, 0.0])
    ArbWaveformDatabase.init(tmp_path / "other")
    ArbWaveformDatabase.save(
        "demo", idata=np.array([0.0, 0.9]), time=np.array([0.0, 9.0])
    )
    cfg = ArbWaveformCfg(data="demo", path=path)
    assert cfg.length == 2.0
    prog = MagicMock()
    prog.soccfg.get_maxv.return_value = 1.0
    prog.soccfg.__getitem__.return_value = [{"samps_per_clk": 1}]
    prog.us2cycles.return_value = 2
    data, qdata = cfg.build("arb").make_iqdata(0, prog)
    np.testing.assert_allclose(data, [0.0, 0.5])
    assert qdata is None
    flat = FlatTopWaveformCfg(length=5.0, raise_waveform=cfg).build("flat")
    assert flat.to_wav_kwargs().get("length") == 3.0
    # Playback reads the current file; serialization preserves locator values.
    restored = ArbWaveformCfg.model_validate_json(cfg.model_dump_json())
    assert restored.data == "demo" and restored.path == path
    _asset(tmp_path, idata=[0.0, 0.2, 0.0])
    data, _ = restored.build("updated").make_iqdata(0, prog)
    np.testing.assert_allclose(data, [0.0, 0.2])


@pytest.mark.parametrize("flat_top", [False, True])
def test_converted_arb_copy_reads_current_capture_duration_and_iq(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flat_top: bool
) -> None:
    monkeypatch.setattr(ArbWaveformDatabase, "_database_path", None)
    first = tmp_path / "first"
    ArbWaveformDatabase.init(first)
    ArbWaveformDatabase.save(
        "demo",
        time=np.array([0.0, 1.0, 2.0]),
        idata=np.array([0.0, 0.3, 0.0]),
        qdata=np.array([0.0, 0.4, 0.0]),
    )
    context = CfgValidationContext(assets=ArbWaveformDatabase.bound_info())
    core = CoreArbWaveformCfg(data="demo")
    source = (
        CoreFlatTopWaveformCfg(length=6.0, raise_waveform=core) if flat_top else core
    )
    copied = deepcopy(WaveformCfgFactory.from_core(source, context=context))
    native = copied.raise_waveform if isinstance(copied, FlatTopWaveformCfg) else copied
    assert isinstance(native, ArbWaveformCfg)
    ArbWaveformDatabase.init(tmp_path / "second")
    ArbWaveformDatabase.save(
        "demo",
        time=np.array([0.0, 9.0]),
        idata=np.array([0.9, 0.9]),
    )
    prog = MagicMock()
    prog.soccfg.get_maxv.return_value = 1.0
    prog.soccfg.__getitem__.return_value = [{"samps_per_clk": 1}]
    prog.us2cycles.side_effect = lambda *, gen_ch, us: round(us)
    waveform = native.build("captured")
    assert waveform.length == 2.0
    idata, qdata = waveform.make_iqdata(0, prog)
    np.testing.assert_allclose(idata, [0.0, 0.3])
    assert qdata is not None
    np.testing.assert_allclose(qdata, [0.0, 0.4])

    path = first / "demo.npz"
    np.savez(path, time=[0, 2, 4], idata=[0, 0.6, 0], qdata=[0, 0.8, 0])
    assert waveform.length == 4.0
    idata, qdata = waveform.make_iqdata(0, prog, even_length=flat_top)
    np.testing.assert_allclose(idata, [0.0, 0.3, 0.6, 0.3])
    assert qdata is not None
    np.testing.assert_allclose(qdata, [0.0, 0.4, 0.8, 0.4])
    if isinstance(copied, FlatTopWaveformCfg):
        assert copied.build("flat").to_wav_kwargs().get("length") == 2.0

    path.unlink()
    with pytest.raises(FileNotFoundError):
        _ = waveform.length
    with pytest.raises(FileNotFoundError):
        waveform.make_iqdata(0, prog)
    np.savez(path, time=[0, 1], idata=[0, 2], qdata=[0, 0])
    with pytest.raises(WaveformFormatError):
        _ = waveform.length
    with pytest.raises(WaveformFormatError):
        waveform.make_iqdata(0, prog)


@pytest.mark.parametrize(
    "path",
    [
        "",
        "relative.npz",
        "C:/foreign/demo.npz" if Path("/").is_absolute() else "/foreign/demo.npz",
    ],
)
def test_native_arb_rejects_nonlocal_or_relative_paths(path: str) -> None:
    with pytest.raises(ValidationError):
        ArbWaveformCfg.model_validate({"data": "demo", "path": path})


def test_native_arb_requires_path_and_propagates_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        ArbWaveformCfg.model_validate({"data": "demo"})
    cfg = ArbWaveformCfg(data="demo", path=(tmp_path / "absent.npz").as_posix())
    with pytest.raises(FileNotFoundError):
        _ = cfg.length


def test_native_arb_plays_samples_without_interpreting_formula_recipe(
    tmp_path: Path,
) -> None:
    cfg = ArbWaveformCfg(
        data="demo", path=_asset(tmp_path, recipe_json='{"future_recipe": true}')
    )
    assert cfg.length == 2.0


def test_native_arb_retains_codec_failure_instead_of_empty_samples(
    tmp_path: Path,
) -> None:
    cfg = ArbWaveformCfg(data="demo", path=_asset(tmp_path, idata=[0.0, 2.0, 0.0]))
    with pytest.raises(WaveformFormatError) as error:
        _ = cfg.length
    assert error.value.reason == "amplitude_out_of_range"


@pytest.mark.parametrize("style", ["arb", "flat_top"])
def test_waveform_factory_converts_core_using_explicit_asset_path(
    tmp_path: Path, style: str
) -> None:
    path = _asset(tmp_path)
    context = CfgValidationContext(assets=ArbWaveformDatabase.info_at(tmp_path))
    core = CoreArbWaveformCfg(data="demo")
    source = (
        core
        if style == "arb"
        else CoreFlatTopWaveformCfg(length=5.0, raise_waveform=core)
    )
    converted = WaveformCfgFactory.from_core(source, context=context)
    native = (
        converted.raise_waveform
        if isinstance(converted, FlatTopWaveformCfg)
        else converted
    )
    assert isinstance(native, ArbWaveformCfg)
    assert native.path == path and native.data == "demo"
    assert core.has_asset_binding is False


@pytest.mark.parametrize("context", [None, CfgValidationContext()])
def test_waveform_factory_requires_assets_instead_of_global_fallback(
    context: CfgValidationContext | None,
) -> None:
    with pytest.raises(ValueError, match="assets"):
        WaveformCfgFactory.from_core(CoreArbWaveformCfg(data="demo"), context=context)


def test_waveform_factory_propagates_missing_asset(tmp_path: Path) -> None:
    context = CfgValidationContext(assets=ArbWaveformDatabase.info_at(tmp_path))
    with pytest.raises(ArbWaveformError) as error:
        WaveformCfgFactory.from_core(
            CoreArbWaveformCfg(data="missing"), context=context
        )
    assert error.value.reason == "data_key_not_found"


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
