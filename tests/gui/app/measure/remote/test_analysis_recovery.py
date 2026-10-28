"""Synthetic owner receipts and source-bound Post recovery through RPC."""

from dataclasses import dataclass

import numpy as np
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    PostAnalysis,
    PostAnalyzeRequest,
    SessionEnv,
)
from zcu_tools.gui.app.measure.services.pane_commit import (
    commit_analysis,
    commit_post_analysis,
    commit_run,
)
from zcu_tools.gui.app.measure.state import (
    AnalysisPaneState,
    PostAnalysisPaneState,
    RunPaneState,
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
class TonePostAnalysis:
    peak: float


def ToneAdapter_get_post_analyze_params(
    source, analyze_result: SampleAnalysis, ctx: SessionEnv
) -> EmptyParams:
    return EmptyParams()


def ToneAdapter_post_analyze(
    req: PostAnalyzeRequest[SampleRun, SampleAnalysis, EmptyParams], *, plots: Plots
) -> TonePostAnalysis:
    return TonePostAnalysis(peak=req.analyze_result.peak)


ToneAdapter = Adapter(
    spec=SampleAdapter.spec,
    make_form=SampleAdapter.make_form,
    run=SampleAdapter.run,
    guide=None,
    analysis=SampleAdapter.analysis,
    post_analysis=PostAnalysis(
        fn=ToneAdapter_post_analyze,
        options=ToneAdapter_get_post_analyze_params,
        writeback=None,
    ),
)


def test_post_start_receipt_binds_recovery(tmp_path, qapp):
    fx = Fixture(headless=True, project_root=str(tmp_path / "project"))
    fx.start()
    sock = open_client(fx.service.port)
    fx.registry.register("tone-cost", ToneAdapter)
    tab = fx.ctrl.tab_control.new_tab("tone-cost")
    commit_run(
        fx.state, tab, RunPaneState(result=RunRecord(None, SampleResult(np.zeros(6))))
    )
    commit_analysis(
        fx.state,
        tab,
        AnalysisPaneState(
            result=SampleAnalysis(peak=1.0),
            plots=None,
            params=fx.state.get_tab(tab).analysis.params,
        ),
    )
    fx.state.update_tab_post_analyze_param_instance(tab, EmptyParams())
    try:
        response = call(sock, "tab.post_analyze", {"tab_id": tab, "updates": {}})
        assert response["ok"], response
        started = response["result"]
        assert started.get("result_read_timeout_seconds") == 60
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
        commit_post_analysis(
            fx.state,
            tab,
            PostAnalysisPaneState(
                result=SampleAnalysis(peak=2.0),
                plots=None,
                params=fx.state.get_tab(tab).post_analysis.params,
                source_operation_id=999,
            ),
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
