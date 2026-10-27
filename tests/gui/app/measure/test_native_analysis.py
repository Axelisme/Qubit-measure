"""Registered native analysis results reach the public RPC and artifact seams."""

import json
import os
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from matplotlib.figure import Figure
from pydantic import BaseModel, JsonValue
from qtpy.QtCore import Qt
from zcu_tools.experiment.artifacts import ArtifactRef, encode_analysis_document
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.metadata import RunMetadata, RunSnapshot, SoftwareProvenance
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    Analysis,
    AnalyzeRequest,
    Interactive,
    PostAnalysis,
    PostAnalyzeRequest,
)
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.remote import ControlOptions, RemoteControlAdapter
from zcu_tools.gui.app.measure.services.results_comparison import build_comparison
from zcu_tools.gui.app.measure.services.results_index import scan_results
from zcu_tools.gui.app.measure.state import SessionEnv, State
from zcu_tools.gui.app.measure.ui.interactive_frontend import InteractiveFrontend
from zcu_tools.gui.app.measure.ui.main_window import MainWindow
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.form_adapter import Form
from zcu_tools.gui.interactive import PluginDefinition, Session
from zcu_tools.gui.remote.rpc_endpoint import ClientLink
from zcu_tools.gui.remote.wire import Request
from zcu_tools.gui.session.context_types import (
    CommandOrigin,
    EntryRequest,
    PointCreateRequest,
)
from zcu_tools.plotting.plots import Plots
from zcu_tools.resources.entry import ArtifactKey

from tests.experiment._saving_support import SavingCfg, SavingCore, SavingResult
from tests.gui._dialog_fakes import RecordingDialogPresenter
from tests.gui._qt_waits import qt_until
from tests.gui._user_input import ComponentDriver
from tests.gui.app.measure.ui._results_browser_support import navigation_path


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
    req: PostAnalyzeRequest[RunRecord[SavingCfg, SavingResult], object, None],
    *,
    plots: Plots,
) -> NativeFit:
    primary = req.analyze_result
    if not isinstance(primary, NativeFit):
        raise TypeError("Post requires the native primary fit")
    return NativeFit(primary.pi_amp + 1, np.array([0.0, np.nan]), 2 + 3j)


class NativeProducer:
    """Caller-owned callbacks can return native values or fail on the next attempt."""

    def __init__(self) -> None:
        self.pending: deque[object] = deque()
        self.pending_post: deque[object] = deque()

    def primary(
        self,
        req: AnalyzeRequest[RunRecord[SavingCfg, SavingResult], None],
        *,
        plots: Plots,
    ) -> object:
        if self.pending:
            result = self.pending.popleft()
            if isinstance(result, Exception):
                raise result
            return result
        return analyze(req, plots=plots)

    def post(
        self,
        req: PostAnalyzeRequest[RunRecord[SavingCfg, SavingResult], object, None],
        *,
        plots: Plots,
    ) -> object:
        if self.pending_post:
            result = self.pending_post.popleft()
            if isinstance(result, Exception):
                raise result
            return result
        return post_analyze(req, plots=plots)

    def plugin(
        self,
        req: AnalyzeRequest[RunRecord[SavingCfg, SavingResult], None],
        *,
        plots: Plots,
    ) -> PluginDefinition[float, object]:
        return PluginDefinition(
            plugin_id="native",
            seed=1.0,
            commands=(),
            can_finish=lambda state: None,
            build_result=lambda state: self.primary(req, plots=plots),
        )


class NativeFrontend(InteractiveFrontend):
    """A presentation boundary without another result or interactive state store."""

    def __init__(self) -> None:
        super().__init__()
        self._figure = Figure()

    @property
    def figure(self) -> Figure:
        return self._figure

    @property
    def preview_active(self) -> bool:
        return False

    def set_read_only(self, *, read_only: bool) -> None:
        pass

    def cancel_preview(self) -> None:
        pass

    def teardown(self) -> None:
        pass


def frontend(
    plugin: PluginDefinition[float, object],
    session: Session[float],
    env: object,
    finish: Callable[[], bool],
    cancel: Callable[[], bool],
    *,
    plots: Plots,
) -> object:
    return NativeFrontend()


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
    producer: NativeProducer

    def reply(self, method: str, **params: object) -> RpcReply:
        self.remote.route(
            self.link, Request("native", method, {"tab_id": self.tab, **params})
        )
        return RpcReply.model_validate_json(self.link.outbound.get(timeout=2))

    def invoke(self, method: str, **params: object) -> dict[str, JsonValue]:
        reply = self.reply(method, **params)
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
def host(tmp_path: Path, qapp, request) -> Iterator[Host]:
    producer = NativeProducer()
    interactive = getattr(request, "param", "sync") == "interactive"
    definition = Adapter(
        spec=SavingCore.AXES_SPEC,
        make_form=lambda context: Form(SavingCfg, {}),
        run=run,
        guide=None,
        analysis=Interactive(plugin=producer.plugin, frontend=frontend)
        if interactive
        else Analysis(fn=producer.primary),
        post_analysis=PostAnalysis(fn=producer.post),
    )
    registry = Registry()
    registry.register("native", definition)
    state = State(SessionEnv(parameters=None, soc=None, soccfg=None))
    controller = Controller(
        state, registry, None, BaseEventBus(), project_root=str(tmp_path)
    )
    window = (
        MainWindow(
            controller,
            templates=lambda: {},
            dialog_presenter=RecordingDialogPresenter(),
        )
        if interactive
        else None
    )
    if window is not None:
        controller.add_view(window)
    tab = controller.tab_control.new_tab("native")
    controller.setup_control.new_entry(EntryRequest("native"))
    controller.setup_control.new_point(
        PointCreateRequest("p", "", CommandOrigin("gui"))
    )
    captured = controller.context_control.read_context()
    storage = captured.capture_storage()
    metadata = RunMetadata(
        run_id="20261008T000000Z-a1b2c3",
        experiment=SavingCore.AXES_SPEC.tag,
        started_at=None,
        finished_at=None,
        completion="complete",
        snapshot=RunSnapshot(
            entry_id=storage.entry_id,
            point=storage.point,
            roles=storage.roles,
        ),
        provenance=SoftwareProvenance(
            software_versions={},
            git_commit=None,
            git_dirty=None,
            qick_version=None,
            soc_fingerprint=None,
            hostname=None,
        ),
    )
    cfg = SavingCfg()
    state.update_tab_result(
        tab,
        RunRecord(cfg, SavingCore().run(cfg, context=None), metadata=metadata),
        parameters=captured,
    )
    remote = RemoteControlAdapter(controller, ControlOptions(port=0), OwnerScheduler())
    link = ClientLink("native-analysis", token_required=False)
    remote.on_client_open(link)
    remote.route(link, Request("connect", "control.connect", {}))
    reply = json.loads(link.outbound.get(timeout=2))
    assert reply["ok"], reply
    try:
        yield Host(state, controller, remote, link, tab, producer)
    finally:
        controller.close()
        remote.on_client_close(link, on_owner_thread=True)
        if window is not None:
            window.deleteLater()
            qapp.processEvents()


@pytest.mark.parametrize("post", [False, True])
def test_native_analysis_is_identical_in_rpc_and_historical_document(
    host: Host, post: bool, tmp_path: Path
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
    report = host.controller.save_control.save_artifacts(
        host.tab,
        artifacts=(ArtifactRef(pane_name, "results"),),
        origin=CommandOrigin("gui"),
    )
    saved = next(row for row in report.outputs if row.output.format == "json")
    assert saved.status == "saved", saved.error
    reopened = json.loads(saved.output.path.read_text(encoding="utf-8"))
    assert reopened == document
    index = scan_results(tmp_path)
    indexed = next(run for run in index.runs if saved.output.path in run.analysis_paths)
    assert index.read_analysis(indexed.key, saved.output.path.name) == document
    comparison = build_comparison(index, (indexed.key,))
    cells = dict(zip(comparison.columns, comparison.rows[0].cells, strict=True))
    assert {
        key: cells[key] for key in ("pi_amp", "errors", "center.real", "center.imag")
    } == {
        "pi_amp": "2.0" if post else "1.0",
        "errors": "[0.0, null]",
        "center.real": "2.0" if post else "1.0",
        "center.imag": "3.0" if post else "2.0",
    }


def test_native_saved_analysis_is_visible_in_browser_and_comparison(
    host: Host, tmp_path: Path, qapp
) -> None:
    # Public host setup produces saved data; UI observations start at the toolbar.
    token = host.invoke("tab.analyze")["operation_id"]
    host.wait(token)
    report = host.controller.save_control.save_artifacts(
        host.tab,
        artifacts=(
            ArtifactRef("run", "measurement_data"),
            ArtifactRef("analysis", "results"),
        ),
        origin=CommandOrigin("gui"),
    )
    assert all(output.status == "saved" for output in report.outputs), report.outputs
    window = MainWindow(
        host.controller,
        templates=lambda: {},
        dialog_presenter=RecordingDialogPresenter(),
    )
    host.controller.add_view(window)
    window.resize(1600, 1000)
    window.show()
    qapp.processEvents()
    ui = ComponentDriver(window)
    evidence = Path(os.environ.get("NATIVE_BROWSER_EVIDENCE", str(tmp_path)))
    try:
        ui.screenshot(evidence / "01-main-window.png")
        ui.click("Results browser…")
        assert ui.observe(object_name="results.browser").visible
        ui.screenshot(evidence / "02-browser.png")
        branch = ("native",)
        for caption in ("p", "test/saving", "a1b2c3  ·  Unknown time"):
            ui.expand_tree(
                navigation_path(ui, branch), object_name="results.navigation"
            )
            branch = (*branch, caption)
        selected = navigation_path(ui, branch)
        ui.select_tree_row(selected, object_name="results.navigation")
        ui.screenshot(evidence / "03-run.png")
        ui.select_tab("Analysis")
        rows = {
            row.columns
            for row in ui.visible().tree_rows
            if row.tree_name == "results.analysis"
        }
        assert {
            ("pi_amp", "1.0"),
            ("errors", ""),
            ("[0]", "0.0"),
            ("[1]", "—"),
            ("real", "1.0"),
            ("imag", "2.0"),
        } <= rows
        ui.screenshot(evidence / "04-native-analysis.png")
        ui.invoke_context_action(
            action="Add to comparison",
            object_name="results.navigation",
            tree_path=selected,
            evidence=evidence / "05-pin-menu.png",
        )
        ui.select_tab("Comparison (1)")
        assert ui.observe(object_name="results.pin_count").text == "Pinned runs: 1"
        rows = ui.table_rows(object_name="results.comparison_table")
        assert len(rows) == 1
        assert rows[0].columns[-4:-1] == ("1.0", "[0.0, null]", "1.0")
        ui.screenshot(evidence / "06-native-comparison.png")
        ui.select_table_row("native", object_name="results.comparison_table")
        ui.press_key(
            Qt.Key.Key_End,
            object_name="results.comparison_table",
            modifiers=Qt.KeyboardModifier.ControlModifier,
        )
        assert (
            ui.table_rows(object_name="results.comparison_table")[0].columns[-1]
            == "2.0"
        )
        ui.screenshot(evidence / "07-native-comparison-imag.png")
    finally:
        host.controller.close()
        window.dispose()
        qapp.processEvents()


@pytest.mark.parametrize("host", ["interactive"], indirect=True)
def test_interactive_result_is_published_only_after_done_settles(host: Host) -> None:
    assert host.invoke("tab.get_analyze_result") == {"summary": None, "invalid": []}
    started = host.invoke("tab.analyze")
    token = started["operation_id"]
    assert started["interactive"] is True
    assert isinstance(token, int)
    assert (
        host.controller.operation_control.await_operation(token, 0).reason
        != "completed"
    )
    assert host.invoke("tab.get_analyze_result")["summary"] is None

    done = host.invoke(
        "tab.interact", payload={"command": "done"}, include_figure=False
    )
    assert done["operation_id"] == token
    host.wait(token)
    result = host.invoke("tab.get_analyze_result", operation_id=token)
    assert result["summary"] == {
        "pi_amp": 1.0,
        "errors": [0.0, None],
        "center": {"real": 1.0, "imag": 2.0},
    }
    assert result["invalid"] == [{"path": "summary.errors[1]", "reason": "non_finite"}]


def test_failed_reanalysis_keeps_primary_and_post_until_next_success(
    host: Host,
) -> None:
    primary = host.invoke("tab.analyze")["operation_id"]
    host.wait(primary)
    post = host.invoke("tab.post_analyze", operation_id=primary)["operation_id"]
    host.wait(post)
    before_primary = host.invoke("tab.get_analyze_result")["summary"]
    before_post = host.invoke("tab.get_post_analyze_result")["summary"]

    host.producer.pending.append(ValueError("native fit failed"))
    failed = host.invoke("tab.analyze")["operation_id"]
    assert isinstance(failed, int)
    control = host.controller.operation_control
    qt_until(lambda: control.await_operation(failed, 0).reason == "completed")
    outcome = control.await_operation(failed, 0).outcome
    assert outcome is not None and outcome.status == "failed"
    assert "native fit failed" in str(outcome)
    assert host.invoke("tab.get_analyze_result")["summary"] == before_primary
    assert host.invoke("tab.get_post_analyze_result")["summary"] == before_post

    host.producer.pending.append(NativeFit(0.5, np.array([0.0]), 3 + 4j))
    replaced = host.invoke("tab.analyze")["operation_id"]
    host.wait(replaced)
    assert host.invoke("tab.get_analyze_result", operation_id=replaced)["summary"] == {
        "pi_amp": 0.5,
        "errors": [0.0],
        "center": {"real": 3.0, "imag": 4.0},
    }
    assert host.invoke("tab.get_post_analyze_result") == {
        "summary": None,
        "invalid": [],
    }


@pytest.mark.parametrize("post", [False, True])
def test_completed_figures_only_result_has_an_empty_object_summary(
    host: Host,
    post: bool,
) -> None:
    if post:
        primary = host.invoke("tab.analyze")["operation_id"]
        host.wait(primary)
        host.producer.pending_post.append(None)
        token = host.invoke("tab.post_analyze", operation_id=primary)["operation_id"]
    else:
        assert host.invoke("tab.get_analyze_result")["summary"] is None
        host.producer.pending.append(None)
        token = host.invoke("tab.analyze")["operation_id"]
    host.wait(token)
    method = "tab.get_post_analyze_result" if post else "tab.get_analyze_result"
    reply = host.invoke(method, operation_id=token)
    assert reply["summary"] == {}
    assert reply["invalid"] == []


def test_failed_post_analysis_preserves_both_committed_native_panes(host: Host) -> None:
    primary = host.invoke("tab.analyze")["operation_id"]
    host.wait(primary)
    previous = host.invoke("tab.post_analyze", operation_id=primary)["operation_id"]
    host.wait(previous)
    primary_summary = host.invoke("tab.get_analyze_result")["summary"]
    post_summary = host.invoke("tab.get_post_analyze_result")["summary"]
    host.producer.pending_post.append(ValueError("native post failed"))
    token = host.invoke("tab.post_analyze", operation_id=primary)["operation_id"]
    assert isinstance(token, int)
    control = host.controller.operation_control
    qt_until(lambda: control.await_operation(token, 0).reason == "completed")
    outcome = control.await_operation(token, 0).outcome
    assert outcome is not None and outcome.status == "failed"
    assert host.invoke("tab.get_analyze_result")["summary"] == primary_summary
    assert host.invoke("tab.get_post_analyze_result")["summary"] == post_summary


@pytest.mark.parametrize("host", ["interactive"], indirect=True)
def test_interactive_construction_failure_retains_the_previous_native_result(
    host: Host,
) -> None:
    first = host.invoke("tab.analyze")["operation_id"]
    host.invoke("tab.interact", payload={"command": "done"}, include_figure=False)
    host.wait(first)
    previous = host.invoke("tab.get_analyze_result")["summary"]

    host.producer.pending.append(ValueError("interactive native failure"))
    token = host.invoke("tab.analyze")["operation_id"]
    assert isinstance(token, int)
    assert host.invoke("tab.get_analyze_result")["summary"] == previous
    host.invoke("tab.interact", payload={"command": "done"}, include_figure=False)
    terminal = host.controller.operation_control.await_operation(token, 0)
    assert terminal.reason == "completed"
    assert terminal.outcome is not None and terminal.outcome.status == "failed"
    assert host.invoke("tab.get_analyze_result")["summary"] == previous

    host.producer.pending.append(NativeFit(0.75, np.array([0.0]), 0j))
    replacement = host.invoke("tab.analyze")["operation_id"]
    host.invoke("tab.interact", payload={"command": "done"}, include_figure=False)
    host.wait(replacement)
    assert host.invoke("tab.get_analyze_result")["summary"] == {
        "pi_amp": 0.75,
        "errors": [0.0],
        "center": {"real": 0.0, "imag": 0.0},
    }


@pytest.mark.parametrize("post", [False, True])
def test_mcp_stdio_bridge_reads_the_same_live_native_summary(
    host: Host, post: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.mcp.measure._analyze_support import call_stdio, stdio_data
    from tests.mcp.measure._support import make_client

    primary = host.invoke("tab.analyze")["operation_id"]
    host.wait(primary)
    token = primary
    if post:
        token = host.invoke("tab.post_analyze", operation_id=primary)["operation_id"]
        host.wait(token)
    method = "tab.get_post_analyze_result" if post else "tab.get_analyze_result"
    direct = host.invoke(method, operation_id=token)

    def respond(name: str, params: dict[str, object]) -> dict[str, JsonValue]:
        # Delegate the transport boundary to the real registered host, not a wire fixture.
        response = host.reply(name, **params)
        assert response.ok, response.error
        assert response.result is not None
        return response.result

    monkeypatch.setenv("ZCU_MCP_CALL_LOG", "0")
    client = make_client(tmp_path, respond)
    try:
        reply = call_stdio(
            monkeypatch,
            client,
            "rpc_call",
            {"method": method, "params": {"tab_id": host.tab, "operation_id": token}},
        )
        result = stdio_data(reply)
        assert result["summary"] == direct["summary"]
        assert result["invalid"] == [
            {"path": "summary.errors[1]", "reason": "non_finite"}
        ]
    finally:
        client.context.session.close()


@dataclass(frozen=True)
class UnsupportedFit:
    payload: object


@pytest.mark.parametrize("post", [False, True])
def test_unsupported_native_field_is_a_public_rpc_and_save_error(
    host: Host,
    post: bool,
) -> None:
    if post:
        primary = host.invoke("tab.analyze")["operation_id"]
        host.wait(primary)
        host.producer.pending_post.append(UnsupportedFit(object()))
        token = host.invoke("tab.post_analyze", operation_id=primary)["operation_id"]
    else:
        host.producer.pending.append(UnsupportedFit(object()))
        token = host.invoke("tab.analyze")["operation_id"]
    host.wait(token)
    method = "tab.get_post_analyze_result" if post else "tab.get_analyze_result"
    reply = host.reply(method, operation_id=token)
    assert not reply.ok
    assert "summary.payload" in str(reply.error)
    assert "object" in str(reply.error)
    with pytest.raises(TypeError, match=r"summary\.payload.*object"):
        host.controller.save_control.save_artifacts(
            host.tab,
            artifacts=(
                ArtifactRef("post_analysis" if post else "analysis", "results"),
            ),
            origin=CommandOrigin("gui"),
        )
