"""Zig-zag GUI adapters lower their forms to the core cfgs and reuse its analysis."""

from unittest.mock import MagicMock

import numpy as np
import pytest
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.twotone.zigzag import ZigZagCfg
from zcu_tools.experiment.v2.twotone.zigzag_sweep import (
    ZigZagScanCfg,
    ZigZagScanResult,
)
from zcu_tools.experiment.v2_gui.measure.adapters.twotone.zigzag import (
    ZigZagAdapter,
    ZigZagScanAnalyzeParams,
    ZigZagScanFreqAdapter,
    ZigZagScanGainAdapter,
)
from zcu_tools.experiment.v2_gui.measure.registry import ADAPTERS
from zcu_tools.gui.app.measure.adapter import AnalyzeRequest, RunRequest, SessionEnv
from zcu_tools.gui.app.measure.adapter.lowering import schema_to_resolved_dict
from zcu_tools.gui.app.measure.cfg_binding import MeasureCfgBindings
from zcu_tools.plotting.plots import NonPresentingHost, Plots
from zcu_tools.resources.context import MetaDict, ModuleLibrary


def _lowered_cfg(adapter_type):
    md = MetaDict()
    md.q_f = 4000.0
    md.r_f = 6200.0
    md.qub_ch = 3
    md.res_ch = 2
    md.pi_gain = 0.5
    md.t1 = 20.0
    ml = ModuleLibrary()
    host = MagicMock()
    host.get_current_md.return_value = md
    host.get_current_ml.return_value = ml
    host.list_device_names.return_value = []
    host.arb_waveforms.list_data_keys.return_value = []
    schema = adapter_type.cfg_definition().instantiate(
        SessionEnv(md=md, ml=ml, soc=None, soccfg=None)
    )
    draft = MeasureCfgBindings(host).new_draft(schema)
    try:
        raw = schema_to_resolved_dict(draft.snapshot())
    finally:
        draft.close()
    request = RunRequest(soc=MagicMock(), soccfg=MagicMock(), device_snapshot={})
    return adapter_type().build_exp_cfg(raw, request)


def test_zigzag_adapters_are_registered() -> None:
    assert ADAPTERS["twotone/zigzag"] is ZigZagAdapter
    assert ADAPTERS["twotone/zigzag_scan/gain"] is ZigZagScanGainAdapter
    assert ADAPTERS["twotone/zigzag_scan/freq"] is ZigZagScanFreqAdapter


def test_zigzag_default_form_lowers_to_core_cfg() -> None:
    cfg = _lowered_cfg(ZigZagAdapter)

    assert isinstance(cfg, ZigZagCfg)
    assert cfg.n_times == 10
    assert cfg.repeat_on == "X180_pulse"
    assert cfg.modules.X180_pulse is not None
    assert cfg.relax_delay == pytest.approx(100.0)


@pytest.mark.parametrize(
    ("adapter_type", "swept", "unset", "start", "stop"),
    [
        (ZigZagScanGainAdapter, "gain", "freq", 0.4, 0.6),
        (ZigZagScanFreqAdapter, "freq", "gain", 3998.0, 4002.0),
    ],
)
def test_zigzag_scan_default_form_sets_exactly_one_sweep(
    adapter_type, swept, unset, start, stop
) -> None:
    cfg = _lowered_cfg(adapter_type)

    assert isinstance(cfg, ZigZagScanCfg)
    sweep = getattr(cfg.sweep, swept)
    assert sweep is not None
    assert (sweep.start, sweep.stop, sweep.expts) == pytest.approx((start, stop, 101))
    assert getattr(cfg.sweep, unset) is None
    assert cfg.n_times == 6


def _scan_record(best: float) -> RunRecord[ZigZagScanCfg, ZigZagScanResult]:
    times = np.arange(7)
    values = np.linspace(0.4, 0.6, 41)
    # Error grows with repetitions in proportion to the distance from ``best``.
    signals = np.cos(np.outer(times, values - best) * 20.0).astype(np.complex128)
    return RunRecord(cfg=None, result=ZigZagScanResult(times, values, signals))


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        (ZigZagScanAnalyzeParams(), 0.5),
        (ZigZagScanAnalyzeParams(find_min=0.55), 0.55),
    ],
)
def test_zigzag_scan_analysis_reports_flattest_sweep_value(params, expected) -> None:
    plots = Plots(NonPresentingHost())
    try:
        answer = ZigZagScanGainAdapter().analyze(
            AnalyzeRequest(
                _scan_record(0.5),
                params,
                md=MagicMock(),
                ml=MagicMock(),
                predictor=None,
            ),
            plots=plots,
        )
        assert answer.best_value == pytest.approx(expected, abs=0.006)
        assert tuple(plots) == ("fit",)
    finally:
        plots.finish(present=False)
        plots.release()
