"""Tests for the Phase 162C singleshot adapters: ac_stark, mist/power_freq,
t1_tone_sweep_{gain,freq}.

Covers:
- cfg_spec structure (2D = two sweep axes; t1_tone_sweep = length + one outer)
  + make_default_cfg().validate(ml) + filename stem
- run override: md has GE centres → correct domain run call (mock)
- run override: md missing centres → fast-fail
- ac_stark analyze: chi/rf_w present → forwarded into (chi positional, kappa kw);
  missing chi or rf_w → fast-fail; ac_stark_coeff writeback
- mist/power_freq + t1_tone_sweep analyze: figure-only result
- t1_tone_sweep: each adapter lowers exactly one outer sweep (domain
  _resolve_outer_sweep satisfied) + uniform default True + pop-before-lowering
- canonical save: domain writes one ADR-0027 native HDF5 from one data_path
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import numpy as np
import pytest
from matplotlib.figure import Figure
from zcu_tools.experiment.v2.singleshot import AcStarkExp
from zcu_tools.experiment.v2.singleshot.mist import FreqPowerExp
from zcu_tools.experiment.v2.singleshot.t1 import (
    T1WithToneSweepCfg,
    T1WithToneSweepExp,
)
from zcu_tools.experiment.v2_gui.adapters.singleshot import (
    MistPowerFreqAdapter,
    SsAcStarkAdapter,
    SsT1ToneSweepFreqAdapter,
    SsT1ToneSweepGainAdapter,
)
from zcu_tools.experiment.v2_gui.adapters.singleshot.ac_stark import (
    SsAcStarkAnalyzeResult,
)
from zcu_tools.experiment.v2_gui.adapters.singleshot.mist.power_freq import (
    MistPowerFreqAnalyzeResult,
)
from zcu_tools.experiment.v2_gui.adapters.singleshot.t1_tone_sweep import (
    SsT1ToneSweepAnalyzeResult,
)
from zcu_tools.gui.app.main.adapter import (
    AnalyzeRequest,
    MetaDictWriteback,
    NoAnalyzeParams,
    RunRequest,
    WritebackRequest,
)
from zcu_tools.meta_tool import MetaDict
from zcu_tools.utils.datasaver import load_labber_data

# ---------------------------------------------------------------------------
# Fixtures / shared helpers (mirror test_phase162b_adapters)
# ---------------------------------------------------------------------------


def _make_ml() -> MagicMock:
    ml = MagicMock()
    ml.modules = {}
    ml.waveforms = {}
    ml.make_cfg.return_value = MagicMock()
    return ml


def _make_ctx(ml: MagicMock | None = None) -> MagicMock:
    ctx = MagicMock()
    ctx.ml = ml or _make_ml()
    ctx.md = MetaDict()
    ctx.qub_name = "Q1"
    return ctx


def _md_with_centers() -> MetaDict:
    md = MetaDict()
    md.g_center = -1.5 + 2.0j
    md.e_center = 1.2 - 0.7j
    md.ge_radius = 0.42
    return md


def _run_req(md: MetaDict, ml: MagicMock) -> RunRequest:
    soc, soccfg = MagicMock(), MagicMock()
    return RunRequest(md=md, ml=ml, soc=soc, soccfg=soccfg)


def _analyze_req(run_result: Any, md: MetaDict) -> AnalyzeRequest[Any, NoAnalyzeParams]:
    return AnalyzeRequest(
        run_result=run_result,
        analyze_params=NoAnalyzeParams(),
        md=md,
        ml=_make_ml(),
        predictor=None,
    )


# ---------------------------------------------------------------------------
# cfg_spec structure (2D sweeps) + validate(ml) + filename stem
# ---------------------------------------------------------------------------


def test_ac_stark_cfg_validates_2d_sweep() -> None:
    ml = _make_ml()
    adapter = SsAcStarkAdapter()
    schema = adapter.make_default_cfg(_make_ctx(ml))
    schema.validate(ml)
    raw = schema.to_raw_dict(None, ml)
    modules = cast(dict[str, Any], raw["modules"])
    assert "stark_pulse1" in modules
    assert "stark_pulse2" in modules
    assert "readout" in modules
    sweep = cast(dict[str, Any], raw["sweep"])
    assert set(sweep) == {"gain", "freq"}  # 2D


def test_mist_power_freq_cfg_validates_2d_sweep() -> None:
    ml = _make_ml()
    adapter = MistPowerFreqAdapter()
    schema = adapter.make_default_cfg(_make_ctx(ml))
    schema.validate(ml)
    raw = schema.to_raw_dict(None, ml)
    modules = cast(dict[str, Any], raw["modules"])
    assert "probe_pulse" in modules
    assert "readout" in modules
    sweep = cast(dict[str, Any], raw["sweep"])
    assert set(sweep) == {"freq", "gain"}  # 2D


@pytest.mark.parametrize(
    ("adapter_cls", "outer_key"),
    [
        (SsT1ToneSweepGainAdapter, "gain"),
        (SsT1ToneSweepFreqAdapter, "freq"),
    ],
)
def test_t1_tone_sweep_cfg_has_one_outer_sweep(
    adapter_cls: Any, outer_key: str
) -> None:
    # Each split adapter exposes length + exactly its one outer sweep — so the
    # lowered cfg satisfies the domain's "exactly one sweep besides length".
    ml = _make_ml()
    adapter = adapter_cls()
    schema = adapter.make_default_cfg(_make_ctx(ml))
    schema.validate(ml)
    raw = schema.to_raw_dict(None, ml)
    sweep = cast(dict[str, Any], raw["sweep"])
    assert set(sweep) == {"length", outer_key}
    # uniform extra present, default True (domain default).
    assert raw["uniform"] is True


@pytest.mark.parametrize(
    ("adapter", "prefix"),
    [
        (SsAcStarkAdapter(), "Q1_sh_ac_stark_"),
        (MistPowerFreqAdapter(), "Q1_mist_power_freq_"),
        (SsT1ToneSweepGainAdapter(), "Q1_ss_t1_tone_sweep_gain_"),
        (SsT1ToneSweepFreqAdapter(), "Q1_ss_t1_tone_sweep_freq_"),
    ],
)
def test_filename_stem(adapter: Any, prefix: str) -> None:
    assert adapter.make_filename_stem(_make_ctx()).startswith(prefix)


# ---------------------------------------------------------------------------
# t1_tone_sweep: domain _resolve_outer_sweep satisfied by the lowered cfg
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("adapter_cls", "outer_key"),
    [
        (SsT1ToneSweepGainAdapter, "gain"),
        (SsT1ToneSweepFreqAdapter, "freq"),
    ],
)
def test_t1_tone_sweep_resolves_single_outer(
    adapter_cls: Any, outer_key: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The adapter exposes only one outer sweep, so the lowered cfg's sweep dict has
    # exactly {length, <outer_key>} — confirm the domain _resolve_outer_sweep picks
    # exactly that outer axis (its "exactly one besides length" contract). The
    # sweep *values* are irrelevant here, so stub sweep2array.
    import zcu_tools.experiment.v2.singleshot.t1.t1_with_tone_sweep as mod

    sweep_dict: dict[str, Any] = {"length": object(), outer_key: object()}
    cfg = MagicMock()
    cfg.sweep.model_dump.return_value = sweep_dict
    cfg.modules.probe_pulse.ch = 0
    monkeypatch.setattr(mod, "sweep2array", lambda *a, **k: np.zeros(3))
    name, _xs = T1WithToneSweepExp()._resolve_outer_sweep(cfg, MagicMock())
    assert name == outer_key


# ---------------------------------------------------------------------------
# run override — forwards GE centres (md present) / fast-fails (md absent)
# ---------------------------------------------------------------------------


def test_ac_stark_run_forwards_centers(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_run(self: Any, soc: Any, soccfg: Any, cfg: Any, g, e, radius) -> Any:
        del self, cfg
        captured.update(soc=soc, soccfg=soccfg, g=g, e=e, radius=radius)
        return MagicMock()

    monkeypatch.setattr(AcStarkExp, "run", fake_run, raising=True)
    ml = _make_ml()
    md = _md_with_centers()
    req = _run_req(md, ml)
    adapter = SsAcStarkAdapter()
    schema = adapter.make_default_cfg(_make_ctx(ml))
    adapter.run(req, schema)
    assert captured["g"] == -1.5 + 2.0j
    assert captured["e"] == 1.2 - 0.7j
    assert captured["radius"] == pytest.approx(0.42)


def test_ac_stark_run_fast_fails_without_centers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        AcStarkExp, "run", lambda *a, **k: pytest.fail("domain run should not run")
    )
    ml = _make_ml()
    req = _run_req(MetaDict(), ml)
    schema = SsAcStarkAdapter().make_default_cfg(_make_ctx(ml))
    with pytest.raises(RuntimeError, match="missing 'g_center'.*singleshot/ge"):
        SsAcStarkAdapter().run(req, schema)


def test_mist_power_freq_run_forwards_centers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_run(self: Any, soc: Any, soccfg: Any, cfg: Any, g, e, radius) -> Any:
        del self, cfg, soc, soccfg
        captured.update(g=g, e=e, radius=radius)
        return MagicMock()

    monkeypatch.setattr(FreqPowerExp, "run", fake_run, raising=True)
    ml = _make_ml()
    req = _run_req(_md_with_centers(), ml)
    adapter = MistPowerFreqAdapter()
    schema = adapter.make_default_cfg(_make_ctx(ml))
    adapter.run(req, schema)
    assert captured["g"] == -1.5 + 2.0j
    assert captured["radius"] == pytest.approx(0.42)


def test_mist_power_freq_run_fast_fails_without_centers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        FreqPowerExp, "run", lambda *a, **k: pytest.fail("domain run should not run")
    )
    ml = _make_ml()
    req = _run_req(MetaDict(), ml)
    schema = MistPowerFreqAdapter().make_default_cfg(_make_ctx(ml))
    with pytest.raises(RuntimeError, match="missing 'g_center'.*singleshot/ge"):
        MistPowerFreqAdapter().run(req, schema)


@pytest.mark.parametrize(
    "adapter_cls", [SsT1ToneSweepGainAdapter, SsT1ToneSweepFreqAdapter]
)
@pytest.mark.parametrize("uniform", [False, True])
def test_t1_tone_sweep_run_forwards_centers_and_uniform(
    adapter_cls: Any, uniform: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    def fake_run(
        self: Any,
        soc: Any,
        soccfg: Any,
        cfg: Any,
        g,
        e,
        radius,
        *,
        uniform: bool = True,
    ) -> Any:
        del self, cfg, soc, soccfg
        captured.update(g=g, e=e, radius=radius, uniform=uniform)
        return MagicMock()

    monkeypatch.setattr(T1WithToneSweepExp, "run", fake_run, raising=True)
    ml = _make_ml()
    req = _run_req(_md_with_centers(), ml)
    adapter = adapter_cls()
    schema = adapter.make_default_cfg(_make_ctx(ml))
    schema.value.with_field("uniform", uniform)
    adapter.run(req, schema)
    assert captured["g"] == -1.5 + 2.0j
    assert captured["radius"] == pytest.approx(0.42)
    assert captured["uniform"] is uniform


@pytest.mark.parametrize(
    "adapter_cls", [SsT1ToneSweepGainAdapter, SsT1ToneSweepFreqAdapter]
)
def test_t1_tone_sweep_uniform_default_true(adapter_cls: Any) -> None:
    # No user override → uniform defaults to True (domain default for this exp).
    ml = _make_ml()
    schema = adapter_cls().make_default_cfg(_make_ctx(ml))
    raw = schema.to_raw_dict(None, ml)
    assert raw["uniform"] is True


@pytest.mark.parametrize(
    "adapter_cls", [SsT1ToneSweepGainAdapter, SsT1ToneSweepFreqAdapter]
)
def test_t1_tone_sweep_pops_uniform_before_lowering(adapter_cls: Any) -> None:
    ml = _make_ml()
    adapter = adapter_cls()
    schema = adapter.make_default_cfg(_make_ctx(ml))
    raw = schema.to_raw_dict(None, ml)
    adapter.build_exp_cfg(raw, _run_req(_md_with_centers(), ml))
    passed_dict = ml.make_cfg.call_args[0][0]
    assert "uniform" not in passed_dict
    ml.make_cfg.assert_called_once_with(passed_dict, T1WithToneSweepCfg)


def test_t1_tone_sweep_run_fast_fails_without_centers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        T1WithToneSweepExp,
        "run",
        lambda *a, **k: pytest.fail("domain run should not run"),
    )
    ml = _make_ml()
    req = _run_req(MetaDict(), ml)
    schema = SsT1ToneSweepGainAdapter().make_default_cfg(_make_ctx(ml))
    with pytest.raises(RuntimeError, match="missing 'g_center'.*singleshot/ge"):
        SsT1ToneSweepGainAdapter().run(req, schema)


# ---------------------------------------------------------------------------
# ac_stark analyze — chi/rf_w inputs + writeback
# ---------------------------------------------------------------------------


def test_ac_stark_analyze_forwards_chi_and_kappa(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    fig = Figure()

    def fake_analyze(
        self: Any,
        chi: float,
        result: Any = None,
        *,
        kappa: float,
        confusion_matrix=None,
        cutoff=None,
    ) -> tuple[float, Figure]:
        del self, result
        captured.update(chi=chi, kappa=kappa, confusion=confusion_matrix, cutoff=cutoff)
        return 3.14, fig

    monkeypatch.setattr(AcStarkExp, "analyze", fake_analyze, raising=True)
    md = MetaDict()
    md.chi = 1.5
    md.rf_w = 0.8
    md.confusion_matrix = [[0.9, 0.1], [0.05, 0.95]]
    md.cutoff = 0.05
    out = SsAcStarkAdapter().analyze(_analyze_req(MagicMock(), md))
    assert isinstance(out, SsAcStarkAnalyzeResult)
    assert out.figure is fig
    assert out.ac_stark_coeff == pytest.approx(3.14)
    assert captured["chi"] == pytest.approx(1.5)
    assert captured["kappa"] == pytest.approx(0.8)  # rf_w → kappa
    assert captured["confusion"] == [[0.9, 0.1], [0.05, 0.95]]
    assert captured["cutoff"] == pytest.approx(0.05)


def test_ac_stark_analyze_cutoff_absent_is_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_analyze(
        self: Any, chi, result=None, *, kappa, confusion_matrix=None, cutoff=None
    ) -> tuple[float, Figure]:
        del self, chi, result, kappa, confusion_matrix
        captured["cutoff"] = cutoff
        return 1.0, Figure()

    monkeypatch.setattr(AcStarkExp, "analyze", fake_analyze, raising=True)
    md = MetaDict()
    md.chi = 1.0
    md.rf_w = 1.0
    SsAcStarkAdapter().analyze(_analyze_req(MagicMock(), md))
    assert captured["cutoff"] is None  # absent md.cutoff → domain default


@pytest.mark.parametrize("missing", ["chi", "rf_w"])
def test_ac_stark_analyze_fast_fails_without_chi_or_kappa(
    missing: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        AcStarkExp,
        "analyze",
        lambda *a, **k: pytest.fail("domain analyze should not run"),
    )
    md = MetaDict()
    if missing != "chi":
        md.chi = 1.0
    if missing != "rf_w":
        md.rf_w = 1.0
    with pytest.raises(RuntimeError, match=f"missing '{missing}'.*dispersive"):
        SsAcStarkAdapter().analyze(_analyze_req(MagicMock(), md))


def test_ac_stark_writeback_key_and_value() -> None:
    analyze_result = SsAcStarkAnalyzeResult(ac_stark_coeff=7.2, figure=Figure())
    req: WritebackRequest[Any, SsAcStarkAnalyzeResult] = WritebackRequest(
        run_result=cast(Any, MagicMock()),
        analyze_result=analyze_result,
        ctx=_make_ctx(),
    )
    items = SsAcStarkAdapter().get_writeback_items(req)
    assert len(items) == 1
    item = cast(MetaDictWriteback, items[0])
    assert item.target_name == "ac_stark_coeff"
    assert item.proposed_value == pytest.approx(7.2)


# ---------------------------------------------------------------------------
# figure-only analyze — mist/power_freq + t1_tone_sweep
# ---------------------------------------------------------------------------


def test_mist_power_freq_analyze_figure_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    fig = Figure()

    def fake_analyze(
        self: Any,
        result: Any = None,
        *,
        ac_coeff=None,
        log_scale=False,
        confusion_matrix=None,
    ) -> Figure:
        del self, result
        captured.update(
            ac_coeff=ac_coeff, log_scale=log_scale, confusion=confusion_matrix
        )
        return fig

    monkeypatch.setattr(FreqPowerExp, "analyze", fake_analyze, raising=True)
    md = MetaDict()
    md.ac_stark_coeff = 2.0
    md.confusion_matrix = [[1.0, 0.0], [0.0, 1.0]]
    out = MistPowerFreqAdapter().analyze(_analyze_req(MagicMock(), md))
    assert isinstance(out, MistPowerFreqAnalyzeResult)
    assert out.figure is fig
    assert captured["ac_coeff"] == pytest.approx(2.0)
    assert captured["confusion"] == [[1.0, 0.0], [0.0, 1.0]]


@pytest.mark.parametrize(
    ("adapter_cls", "outer_label"),
    [
        (SsT1ToneSweepGainAdapter, "Probe gain (a.u.)"),
        (SsT1ToneSweepFreqAdapter, "Probe frequency (MHz)"),
    ],
)
def test_t1_tone_sweep_analyze_figure_only_with_xlabel(
    adapter_cls: Any, outer_label: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}
    fig = Figure()

    def fake_analyze(
        self: Any,
        result: Any = None,
        *,
        ac_coeff=None,
        confusion_matrix=None,
        xlabel: str = "",
    ) -> Figure:
        del self, result
        captured.update(ac_coeff=ac_coeff, confusion=confusion_matrix, xlabel=xlabel)
        return fig

    monkeypatch.setattr(T1WithToneSweepExp, "analyze", fake_analyze, raising=True)
    md = MetaDict()
    md.ac_stark_coeff = 1.0
    out = adapter_cls().analyze(_analyze_req(MagicMock(), md))
    assert isinstance(out, SsT1ToneSweepAnalyzeResult)
    assert out.figure is fig
    assert captured["xlabel"] == outer_label  # adapter labels with its outer axis


# ---------------------------------------------------------------------------
# canonical save — one native HDF5 from one data_path
# ---------------------------------------------------------------------------


def test_ac_stark_canonical_save_single_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import zcu_tools.experiment.utils as experiment_utils

    monkeypatch.setattr(experiment_utils, "make_comment", lambda cfg, comment: "")
    exp = AcStarkExp()
    exp.last_result = _make_ac_stark_result()
    base = tmp_path / "Q1_sh_ac_stark@flux0"
    exp.save(filepath=str(base))
    written = sorted(p.name for p in tmp_path.iterdir())
    assert written == ["Q1_sh_ac_stark@flux0.hdf5"]

    raw = load_labber_data(str(tmp_path / written[0]))
    assert [axis.name for axis in raw.axes] == [
        "GE Population",
        "Frequency",
        "Stark Pulse Gain",
    ]
    assert raw.z.shape == (4, 3, 2)
    assert raw.tags == ["singleshot/ac_stark"]


def test_t1_tone_sweep_canonical_save_single_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import zcu_tools.experiment.utils as experiment_utils

    monkeypatch.setattr(experiment_utils, "make_comment", lambda cfg, comment: "")
    exp = T1WithToneSweepExp()
    exp.last_result = _make_t1_tone_sweep_result()
    base = tmp_path / "Q1_ss_t1_tone_sweep_gain@flux0"
    exp.save(filepath=str(base))
    written = sorted(p.name for p in tmp_path.iterdir())
    assert written == ["Q1_ss_t1_tone_sweep_gain@flux0.hdf5"]

    raw = load_labber_data(str(tmp_path / written[0]))
    assert [axis.name for axis in raw.axes] == [
        "GE Population",
        "Time",
        "Initial State",
        "Sweep Value",
    ]
    assert raw.z.shape == (4, 2, 5, 2)
    assert raw.tags == ["singleshot/t1/t1_with_tone_sweep"]


# --- small real-result builders for the save tests -------------------------


def _make_ac_stark_result() -> Any:
    from zcu_tools.experiment.v2.singleshot.ac_stark import AcStarkCfg, AcStarkResult

    gains = np.linspace(0.0, 1.0, 4, dtype=np.float64)
    freqs = np.linspace(-5.0, 5.0, 3, dtype=np.float64)
    populations = np.zeros((len(gains), len(freqs), 2), dtype=np.float64)
    cfg = cast(AcStarkCfg, MagicMock(spec=AcStarkCfg))
    return AcStarkResult(
        gains=gains, freqs=freqs, populations=populations, cfg_snapshot=cfg
    )


def _make_t1_tone_sweep_result() -> Any:
    from zcu_tools.experiment.v2.singleshot.t1.t1_with_tone_sweep import (
        T1WithToneSweepCfg as _Cfg,
    )
    from zcu_tools.experiment.v2.singleshot.t1.t1_with_tone_sweep import (
        T1WithToneSweepResult,
    )

    xs = np.linspace(0.0, 1.0, 4, dtype=np.float64)
    lengths = np.linspace(0.0, 10.0, 5, dtype=np.float64)
    signals = np.zeros((len(xs), 2, len(lengths), 2), dtype=np.float64)
    cfg = cast(_Cfg, MagicMock(spec=_Cfg))
    return T1WithToneSweepResult(
        xs=xs, lengths=lengths, signals=signals, cfg_snapshot=cfg
    )
