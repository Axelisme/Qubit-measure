"""AllXY GUI adapter lowers its form to the core cfg and reuses its analysis."""

from unittest.mock import MagicMock

import numpy as np
import pytest
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import AnalyzeRequest, RunRequest, SessionEnv
from zcu_tools.gui.app.measure.adapter.lowering import schema_to_resolved_dict
from zcu_tools.gui.app.measure.cfg_binding import MeasureCfgBindings
from zcu_tools.plotting.plots import NonPresentingHost, Plots
from zcu_tools.resources.context import MetaDict, ModuleLibrary

from zcu_lab.definitions import ADAPTERS
from zcu_lab.v2.twotone.allxy.core import (
    ALLXY_SEQUENCE,
    AllXY_Result,
    AllXYCfg,
    predict_state_with_error,
)
from zcu_lab.v2.twotone.allxy.gui import (
    AllXYAdapter,
    AllXYAnalyzeParams,
)


def test_allxy_adapter_is_registered() -> None:
    assert ADAPTERS["twotone/allxy"] is AllXYAdapter


def test_allxy_default_form_lowers_to_core_cfg() -> None:
    md = MetaDict()
    md.q_f = 4000.0
    md.r_f = 6200.0
    md.qub_ch = 3
    md.res_ch = 2
    md.t1 = 20.0
    ml = ModuleLibrary()
    host = MagicMock()
    host.get_current_md.return_value = md
    host.get_current_ml.return_value = ml
    host.list_device_names.return_value = []
    host.arb_waveforms.list_data_keys.return_value = []
    schema = AllXYAdapter.cfg_definition().instantiate(
        SessionEnv(md=md, ml=ml, soc=None, soccfg=None)
    )
    draft = MeasureCfgBindings(host).new_draft(schema)
    try:
        raw = schema_to_resolved_dict(draft.snapshot())
    finally:
        draft.close()
    request = RunRequest(soc=MagicMock(), soccfg=MagicMock(), device_snapshot={})

    cfg = AllXYAdapter().build_exp_cfg(raw, request)

    assert isinstance(cfg, AllXYCfg)
    assert cfg.modules.X90_pulse is not None
    assert cfg.modules.X180_pulse is not None
    assert cfg.modules.I_pulse is None
    assert cfg.relax_delay == pytest.approx(100.0)


@pytest.mark.parametrize("fit_ge", [False, True])
def test_allxy_analysis_reports_fitted_errors(fit_ge: bool) -> None:
    states = np.array(
        [predict_state_with_error(seq, 0.05, 0.0) for seq in ALLXY_SEQUENCE]
    )
    record = RunRecord(
        cfg=None,
        result=AllXY_Result(
            gate_idxs=np.arange(len(ALLXY_SEQUENCE), dtype=np.int64),
            signals=(1.0 + 0.5 * states).astype(np.complex128),
        ),
    )
    plots = Plots(NonPresentingHost())
    try:
        answer = AllXYAdapter().analyze(
            AnalyzeRequest(
                record,
                AllXYAnalyzeParams(fit_ge=fit_ge),
                md=MagicMock(),
                ml=MagicMock(),
                predictor=None,
            ),
            plots=plots,
        )
        assert answer.power_param == pytest.approx(0.05, abs=0.005)
        assert answer.detune_param == pytest.approx(0.0, abs=0.005)
        assert answer.power_err > 0.0
        assert set(answer.to_summary_dict()) == {
            "power_param",
            "detune_param",
            "power_err",
            "detune_err",
        }
        assert tuple(plots) == ("fit",)
    finally:
        plots.finish(present=False)
        plots.release()
