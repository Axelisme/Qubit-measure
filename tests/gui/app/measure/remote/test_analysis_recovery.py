"""Synthetic owner receipts and source-bound Post recovery through RPC."""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pytest
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    PostAnalyzeRequest,
    PostAnalyzeResultBase,
    SessionEnv,
)
from zcu_tools.plotting.plots import Plots

from tests.gui._qt_waits import qt_until
from tests.gui.app.measure._synthetic import (
    EmptyParams,
    SampleAdapter,
    SampleAnalysis,
    SampleResult,
    SampleRun,
)

from ._helpers import Fixture, call, open_client


@dataclass
class TonePostAnalysis(PostAnalyzeResultBase):
    peak: float


class ToneAdapter(SampleAdapter):
    """Declare a synthetic per-tone read cost without any lab definition."""

    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        requires_soc=False, load_data=False, post_analysis=True
    )

    def analysis_result_tone_count(self, result: SampleRun) -> int:
        return len(result.result.data)

    def get_post_analyze_params(
        self, analyze_result: SampleAnalysis, ctx: SessionEnv
    ) -> EmptyParams:
        return EmptyParams()

    def post_analyze(
        self,
        req: PostAnalyzeRequest[SampleRun, SampleAnalysis, EmptyParams],
        *,
        plots: Plots,
    ) -> TonePostAnalysis:
        return TonePostAnalysis(peak=req.analyze_result.peak)


@pytest.mark.parametrize(("tones", "seconds"), [(0, 5), (1, 10), (6, 35), (20, 60)])
def test_post_start_receipt_scales_read_budget_and_binds_recovery(
    tmp_path, qapp, tones, seconds
):
    fx = Fixture(headless=True, project_root=str(tmp_path / "project"))
    fx.start()
    sock = open_client(fx.service.port)
    fx.registry.register("tone-cost", ToneAdapter)
    tab = fx.ctrl.new_tab("tone-cost")
    fx.state.update_tab_result(tab, RunRecord(None, SampleResult(np.zeros(tones))))
    fx.state.update_tab_analyze(tab, SampleAnalysis(peak=1.0), None)
    fx.state.update_tab_post_analyze_param_instance(tab, EmptyParams())
    try:
        response = call(sock, "tab.post_analyze", {"tab_id": tab, "updates": {}})
        assert response["ok"], response
        started = response["result"]
        assert started.get("result_read_timeout_seconds") == seconds
        assert started.get("recovery") == {
            "rpc": "tab.get_post_analyze_result",
            "params": {"tab_id": tab, "operation_id": started["operation_id"]},
            "stage": "post",
        }
        qt_until(lambda: not fx.state.is_tab_busy(tab))
        snapshot = call(sock, "tab.snapshot", {"tab_id": tab})["result"]["tabs"][0]
        assert (
            snapshot["post_analysis_state"]["source_operation_id"]
            == started["operation_id"]
        )
        recovered = call(
            sock, "tab.get_post_analyze_result", started["recovery"]["params"]
        )
        assert recovered["result"]["summary"] == {"peak": 1.0}
        fx.state.update_tab_post_analyze(
            tab, SampleAnalysis(peak=2.0), None, source_operation_id=999
        )
        superseded = call(
            sock, "tab.get_post_analyze_result", started["recovery"]["params"]
        )
        assert superseded["error"]["reason"] == "result_superseded"
    finally:
        sock.close()
        fx.ctrl.close()
        fx.ctrl.tab_control.close_tab(tab)
        fx.stop()
