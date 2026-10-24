"""Registered native analysis results reach the public RPC and artifact seams."""

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from pydantic import BaseModel, JsonValue
from zcu_tools.experiment.artifacts import encode_analysis_document
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    Analysis,
    AnalyzeRequest,
    PostAnalysis,
    PostAnalyzeRequest,
)
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.remote import ControlOptions, RemoteControlAdapter
from zcu_tools.gui.app.measure.state import SessionEnv, State
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.form_adapter import Form
from zcu_tools.gui.remote.rpc_endpoint import ClientLink
from zcu_tools.gui.remote.wire import Request
from zcu_tools.plotting.plots import Plots
from zcu_tools.resources.entry import ArtifactKey

from tests.experiment._saving_support import SavingCfg, SavingCore, SavingResult
from tests.gui._qt_waits import qt_until


class OwnerScheduler:
    """Run request dispatch on the current test owner thread."""

    def is_owner_thread(self) -> bool:
        return True

    def post(self, callback: Callable[[], None]) -> None:
        callback()

    def call[T](self, callback: Callable[[], T]) -> T:
        return callback()


@dataclass(frozen=True)
class NativeFit:
    pi_amp: float
    errors: np.ndarray
    center: complex


def run(cfg: SavingCfg, *, context: RunContext) -> RunRecord[SavingCfg, SavingResult]:
    return RunRecord(cfg, SavingCore().run(cfg, context=context))


def analyze(
    req: AnalyzeRequest[RunRecord[SavingCfg, SavingResult], None], *, plots: Plots
) -> NativeFit:
    return NativeFit(
        req.run_result.result.signal.real[0], np.array([0.0, np.inf]), 1 + 2j
    )


def post_analyze(
    req: PostAnalyzeRequest[RunRecord[SavingCfg, SavingResult], NativeFit, None],
    *,
    plots: Plots,
) -> NativeFit:
    return NativeFit(req.analyze_result.pi_amp + 1, np.array([0.0, np.nan]), 2 + 3j)


class RpcReply(BaseModel):
    """Public request reply with either a result object or a JSON error."""

    ok: bool
    result: dict[str, JsonValue] | None = None
    error: JsonValue = None


@dataclass
class Host:
    state: State
    controller: Controller
    remote: RemoteControlAdapter
    link: ClientLink
    tab: str

    def invoke(self, method: str, **params: object) -> dict[str, JsonValue]:
        self.remote.route(
            self.link, Request("native", method, {"tab_id": self.tab, **params})
        )
        reply = RpcReply.model_validate_json(self.link.outbound.get(timeout=2))
        assert reply.ok, reply.error
        assert reply.result is not None
        return reply.result

    def wait(self, token: object) -> None:
        if not isinstance(token, int):
            pytest.fail(f"Expected operation token, got {token!r}")
        control = self.controller.operation_control
        qt_until(lambda: control.await_operation(token, 0).reason == "completed")
        outcome = control.await_operation(token, 0).outcome
        assert outcome is not None and outcome.status == "finished", outcome


@pytest.fixture
def host(tmp_path: Path, qapp) -> Iterator[Host]:
    definition = Adapter(
        spec=SavingCore.AXES_SPEC,
        make_form=lambda context: Form(SavingCfg, {}),
        run=run,
        guide=None,
        analysis=Analysis(fn=analyze),
        post_analysis=PostAnalysis(fn=post_analyze),
    )
    registry = Registry()
    registry.register("native", definition)
    state = State(SessionEnv(parameters=None, soc=None, soccfg=None))
    controller = Controller(
        state, registry, None, BaseEventBus(), project_root=str(tmp_path)
    )
    tab = controller.new_tab("native")
    cfg = SavingCfg()
    state.update_tab_result(tab, RunRecord(cfg, SavingCore().run(cfg, context=None)))
    remote = RemoteControlAdapter(controller, ControlOptions(port=0), OwnerScheduler())
    link = ClientLink("native-analysis", token_required=False)
    remote.on_client_open(link)
    remote.route(link, Request("connect", "control.connect", {}))
    reply = json.loads(link.outbound.get(timeout=2))
    assert reply["ok"], reply
    try:
        yield Host(state, controller, remote, link, tab)
    finally:
        controller.close()
        remote.on_client_close(link, on_owner_thread=True)


@pytest.mark.parametrize("post", [False, True])
def test_native_analysis_is_identical_in_rpc_and_historical_document(
    host: Host, post: bool
):
    primary = host.invoke("tab.analyze")["operation_id"]
    host.wait(primary)
    token = primary
    if post:
        token = host.invoke("tab.post_analyze", operation_id=primary)["operation_id"]
        host.wait(token)
    method = "tab.get_post_analyze_result" if post else "tab.get_analyze_result"
    reply = host.invoke(method, operation_id=token)
    expected = {
        "pi_amp": 2.0 if post else 1.0,
        "errors": [0.0, None],
        "center": {"real": 2.0, "imag": 3.0} if post else {"real": 1.0, "imag": 2.0},
    }
    assert reply["summary"] == expected
    assert reply["invalid"] == [{"path": "summary.errors[1]", "reason": "non_finite"}]
    tab = host.state.get_tab(host.tab)
    pane_name = "post_analysis" if post else "analysis"
    pane = tab.post_analysis if post else tab.analysis
    item = tab.adapter.analysis_item(
        ArtifactKey(pane_name, "results", "analysis", pane_name),
        pane.result,
        pane.result_params,
        pane=pane_name,
    )
    document = json.loads(encode_analysis_document(item.content))
    assert document == {
        "format": "zcu.analysis-result",
        "format_version": "1.0",
        "pane": pane_name,
        "parameters": {},
        "summary": expected,
    }
