"""OneTone GUI captured records, delay policy and readout writeback."""

from __future__ import annotations

from typing import Literal
from unittest.mock import Mock

import pytest
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.stop_signal import StopSignal
from zcu_tools.experiment.v2.onetone.freq import (
    FreqAnalyzeOptions,
    FreqCfg,
    FreqExp,
    FreqResult,
)
from zcu_tools.experiment.v2_gui.measure.adapters.onetone.freq import (
    OneToneFreqAdapter,
    OneToneFreqAnalyzeParams,
)
from zcu_tools.gui.app.measure.adapter import (
    AnalyzeRequest,
    MetaDictWriteback,
    ModuleWriteback,
    RunRequest,
    SessionEnv,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.adapter.lowering import schema_to_raw_dict
from zcu_tools.plotting.plots import NonPresentingHost, Plots
from zcu_tools.program.v2.mocksoc import make_mock_soc

from tests.experiment.v2.onetone._support import make_freq_cfg, make_freq_result


def test_run_captures_matching_cfg_source_and_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = make_freq_cfg(homophasal=True)
    raw_cfg = cfg.to_dict()
    soc, soccfg = make_mock_soc()
    request = RunRequest(soc=soc, soccfg=soccfg, device_snapshot={})
    plots = Plots(NonPresentingHost())
    data = make_freq_result()
    observed: list[tuple[FreqCfg, RunContext]] = []

    def run(_core: FreqExp, config: FreqCfg, *, context: RunContext) -> FreqResult:
        observed.append((config, context))
        return data

    monkeypatch.setattr(FreqExp, "run", run)
    source = OneToneFreqAdapter().run(
        request,
        raw_cfg,
        context=RunContext(
            request.soc, request.soccfg, plots, devices={}, cancel_signal=StopSignal()
        ),
    )
    config, context = observed[0]
    assert source.cfg == config
    assert source.result is data
    assert context.soc is soc and context.soccfg is soccfg and context.plots is plots
    assert config.homophasal == cfg.homophasal
    config.reps = 11
    raw_cfg["reps"] = 99
    assert source.cfg is not None and source.cfg.reps == 2
    plots.finish()
    plots.release()


@pytest.mark.parametrize("mode", ["manual", "calibrated", "auto"])
def test_seeded_real_fit_preserves_delay_policy_and_writeback(
    mode: Literal["manual", "calibrated", "auto"],
) -> None:
    cfg = make_freq_cfg()
    source = RunRecord(cfg=cfg, result=make_freq_result())
    md = Mock()
    md.get.return_value = {"edelay": 0.021, "res_ch": 0, "ro_ch": 0}
    plots = Plots(NonPresentingHost())
    params = OneToneFreqAnalyzeParams(edelay_mode=mode, manual_edelay=0.021)
    adapter = OneToneFreqAdapter()
    answer = adapter.analyze(
        AnalyzeRequest(source, params, md=md, ml=Mock(), predictor=None), plots=plots
    )
    figures = plots.finish()
    assert answer.freq == pytest.approx(6000.0, abs=0.01)
    assert answer.fwhm == pytest.approx(6000.0 / 700.0, rel=0.01)
    assert answer.edelay == pytest.approx(0.021, abs=1e-5)
    assert answer.edelay_source == ("manual" if mode == "manual" else "calibrated")
    assert answer.edelay_persistable
    assert tuple(figures) == ("fit",)
    items = adapter.get_writeback_items(
        WritebackRequest(source, answer, Mock(spec=SessionEnv))
    )
    scalars = {
        item.target_name: item.proposed_value
        for item in items
        if isinstance(item, MetaDictWriteback)
    }
    assert scalars["r_f"] == answer.freq
    assert scalars["rf_w"] == answer.fwhm
    assert scalars["theta0"] == answer.params["theta0"]
    assert scalars["res_edelay_calibration"] == {
        "edelay": answer.edelay,
        "res_ch": 0,
        "ro_ch": 0,
    }
    module = next(item for item in items if isinstance(item, ModuleWriteback))
    assert module.target_name == "readout_rf"
    assert module.edit_schema is not None
    raw = schema_to_raw_dict(module.edit_schema, None, None)
    rebuilt = type(cfg.modules.readout).model_validate(raw)
    assert rebuilt.pulse_cfg.freq == answer.freq
    assert rebuilt.ro_cfg.ro_freq == answer.freq
    assert rebuilt.pulse_cfg.gain == cfg.modules.readout.pulse_cfg.gain
    assert rebuilt.ro_cfg.trig_offset == cfg.modules.readout.ro_cfg.trig_offset
    assert source.cfg == cfg
    plots.release()


def test_uniform_unseeded_fit_does_not_persist_delay_calibration() -> None:
    source = RunRecord(cfg=make_freq_cfg(), result=make_freq_result())
    md = Mock()
    md.get.return_value = None
    adapter = OneToneFreqAdapter()
    plots = Plots(NonPresentingHost())
    answer = adapter.analyze(
        AnalyzeRequest(
            source,
            OneToneFreqAnalyzeParams(fit_bg_amp_slope=False),
            md=md,
            ml=Mock(),
            predictor=None,
        ),
        plots=plots,
    )
    plots.finish()
    assert answer.freq == pytest.approx(6000.0, abs=0.02)
    assert answer.edelay_source == "global"
    assert not answer.edelay_persistable
    items = adapter.get_writeback_items(
        WritebackRequest(source, answer, Mock(spec=SessionEnv))
    )
    assert [item.target_name for item in items] == [
        "r_f",
        "rf_w",
        "theta0",
        "readout_rf",
    ]
    plots.release()


@pytest.mark.parametrize("prior", [None, {"edelay": 0.021, "res_ch": 1, "ro_ch": 0}])
def test_calibrated_mode_rejects_missing_or_other_route_prior(
    prior: dict[str, float] | None,
) -> None:
    source = RunRecord(cfg=make_freq_cfg(), result=make_freq_result())
    md = Mock()
    md.get.return_value = prior
    plots = Plots(NonPresentingHost())
    with pytest.raises(ValueError, match="route-matched"):
        OneToneFreqAdapter().analyze(
            AnalyzeRequest(
                source,
                OneToneFreqAnalyzeParams(edelay_mode="calibrated"),
                md=md,
                ml=Mock(),
                predictor=None,
            ),
            plots=plots,
        )
    plots.finish(present=False)
    plots.release()


def test_enabled_background_options_project_the_same_real_core_fit() -> None:
    source = RunRecord(cfg=make_freq_cfg(), result=make_freq_result(background=True))
    expected_plots = Plots(NonPresentingHost())
    plots = Plots(NonPresentingHost())
    expected = FreqExp().analyze(
        source,
        FreqAnalyzeOptions(
            model_type="hm",
            fit_bg_amp_slope=True,
            fit_bg_phase_curvature=True,
            edelay_branch_seed=0.021,
        ),
        plots=expected_plots,
    )
    actual = OneToneFreqAdapter().analyze(
        AnalyzeRequest(
            source,
            OneToneFreqAnalyzeParams(
                fit_bg_amp_slope=True,
                fit_bg_phase_curvature=True,
                edelay_mode="manual",
                manual_edelay=0.021,
            ),
            md=Mock(),
            ml=Mock(),
            predictor=None,
        ),
        plots=plots,
    )
    expected_plots.finish()
    plots.finish()
    assert actual.freq == pytest.approx(expected.freq, rel=1e-12)
    assert actual.fwhm == pytest.approx(expected.fwhm, rel=1e-12)
    for key in ["edelay", "theta0", "bg_amp_slope", "bg_phase_curvature"]:
        assert actual.params[key] == pytest.approx(expected.params[key], rel=1e-12)
    assert tuple(plots) == ("fit",)
    expected_plots.release()
    plots.release()


def test_nullable_source_can_fit_without_readout_template_writeback() -> None:
    source = RunRecord[FreqCfg, FreqResult](cfg=None, result=make_freq_result())
    plots = Plots(NonPresentingHost())
    adapter = OneToneFreqAdapter()
    answer = adapter.analyze(
        AnalyzeRequest(
            source,
            OneToneFreqAnalyzeParams(
                edelay_mode="manual", manual_edelay=0.021, fit_bg_amp_slope=False
            ),
            md=Mock(),
            ml=Mock(),
            predictor=None,
        ),
        plots=plots,
    )
    plots.finish()
    assert answer.freq == pytest.approx(6000.0, abs=0.02)
    assert not answer.edelay_persistable
    items = adapter.get_writeback_items(
        WritebackRequest(source, answer, Mock(spec=SessionEnv))
    )
    assert [item.target_name for item in items] == ["r_f", "rf_w", "theta0"]
    plots.release()
