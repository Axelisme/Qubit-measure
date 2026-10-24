"""Figures-only adapters retain completed analysis without numeric payloads."""

import json
from pathlib import Path
from typing import ClassVar

import pytest
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import AdapterCapabilities, AnalysisMode
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.remote import ControlOptions, RemoteControlAdapter
from zcu_tools.gui.form_adapter import Form
from zcu_tools.gui.session.context_types import EntryRequest, PointUseRequest

from tests.experiment._saving_support import SavingCfg, SavingCore, SavingResult
from tests.gui._qt_waits import qt_until
from tests.gui.app.measure._native_save_support import native_save_session

from ._form_rpc import FormRpc, ImmediateScheduler


class FiguresAdapter(BaseAdapter[SavingCfg, RunRecord[SavingCfg, SavingResult]]):
    exp_cls = SavingCore
    experiment_tag: ClassVar[str | None] = SavingCore.AXES_SPEC.tag
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        analysis=AnalysisMode.FIT, requires_soc=False, load_data=False
    )

    @staticmethod
    def make_form(ctx):
        del ctx
        return Form(SavingCfg, {})

    def run(self, req, cfg, *, context):
        return RunRecord(cfg, SavingCore().run(cfg, context=context))

    def analyze(self, req, *, plots) -> None:
        assert req.analyze_params is None
        _, ax = plots.subplots("fit")
        ax.plot([1, 2], [3, 4])


class PostFiguresAdapter(FiguresAdapter):
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        analysis=AnalysisMode.FIT,
        post_analysis=True,
        requires_soc=False,
        load_data=False,
    )

    def post_analyze(self, req, *, plots) -> None:
        assert req.analyze_result is None
        assert req.post_analyze_params is None
        _, ax = plots.subplots("post")
        ax.plot([1, 2], [4, 5])


@pytest.fixture
def figures_rpc(tmp_path, qapp, post):
    adapter_cls = PostFiguresAdapter if post else FiguresAdapter
    assert adapter_cls.analyze_params_cls() is type(None)
    assert adapter_cls.post_analyze_params_cls() is type(None)
    with native_save_session(tmp_path, qapp) as session:
        session.state.get_tab("tab").adapter = adapter_cls()
        registry = Registry()
        registry.register("saving", adapter_cls)
        controller = Controller(
            session.state, registry, None, session.bus, project_root=str(tmp_path)
        )
        controller.setup_control.open_entry(EntryRequest("original"))
        controller.setup_control.use_point(PointUseRequest("p"))
        remote = RemoteControlAdapter(
            controller, ControlOptions(port=0), ImmediateScheduler()
        )
        remote.save_control = session.control
        rpc = FormRpc(remote)

        def invoke(method, **params):
            reply = rpc.call(method, {"tab_id": "tab", **params})
            assert reply["ok"], reply
            return reply["result"]

        def wait(token):
            control = controller.operation_control
            qt_until(lambda: control.await_operation(token, 0).reason == "completed")
            outcome = control.await_operation(token, 0).outcome
            assert outcome is not None and outcome.status == "finished", outcome

        try:
            yield session, invoke, wait
        finally:
            controller.close()
            remote.on_client_close(rpc.link, on_owner_thread=True)


def assert_saved_panes(session, post):
    saved = session.save_artifacts()
    assert saved.outputs and all(output.status == "saved" for output in saved.outputs)
    paths = [
        Path(output.path)
        for section in session.state.get_artifact_snapshots("tab")
        for artifact in section.artifacts
        for member in artifact.members
        for output in member.outputs
        if output.status == "saved"
    ]
    documents = [
        json.loads(path.read_text()) for path in paths if path.suffix == ".json"
    ]
    for pane in ["analysis", "post_analysis"] if post else ["analysis"]:
        assert any(
            doc["pane"] == pane and doc["parameters"] == {} and doc["summary"] == {}
            for doc in documents
        )
    assert sum(
        path.suffix == ".png" and path.read_bytes().startswith(b"\x89PNG")
        for path in paths
    ) == (4 if post else 2)


@pytest.mark.parametrize("post", [False, True])
def test_figures_only_run_analyze_and_save(figures_rpc, post):
    session, invoke, wait = figures_rpc
    wait(invoke("tab.run_start")["operation_id"])
    started = invoke("tab.analyze")
    assert started["params"] is None
    token = started["operation_id"]
    wait(token)
    result = invoke("tab.get_analyze_result", operation_id=token)
    assert result["summary"] == {}
    assert result["params"] is None
    assert result["presentation"]["figure_names"] == ["fit"]
    assert invoke("tab.writeback_preview", subtab_id="analysis")["items"] == []
    if post:
        post_start = invoke("tab.post_analyze", operation_id=token)
        assert post_start["params"] is None
        post_token = post_start["operation_id"]
        wait(post_token)
        post_result = invoke("tab.get_post_analyze_result", operation_id=post_token)
        assert post_result["summary"] == {}
        assert post_result["params"] is None
        assert post_result["presentation"]["figure_names"] == ["post"]
        assert invoke("tab.writeback_preview", subtab_id="post_analysis")["items"] == []
    assert_saved_panes(session, post)
    if post:
        replacement = invoke("tab.analyze")
        assert "post.result" in replacement["invalidated_on_success"]
        wait(replacement["operation_id"])
        assert invoke("tab.get_post_analyze_result")["summary"] is None
